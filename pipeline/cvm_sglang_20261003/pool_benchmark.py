#!/usr/bin/env python3
"""One bounded real-workload host-pool diagnostic; no remote setup or retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import threading
import time

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'benchmark' / 'frozen_c4_c8_runner.py'
spec = importlib.util.spec_from_file_location('frozen_pool_runner', RUNNER)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def tp_evidence(value, path=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('tp_size', 'tensor_parallel_size'):
                found.append({'path': path + '/' + key, 'value': child})
            found.extend(tp_evidence(child, path + '/' + key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(tp_evidence(child, path + '/' + str(i)))
    return found


def utc_unix(stamp):
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()


def aggregate(replica_results):
    first = min(r['first_request_started_unix'] for r in replica_results)
    last = max(r['last_response_finished_unix'] for r in replica_results)
    wall = last - first
    b.require(wall > 0, 'Nonpositive client wall interval')
    good = sum(r['successful_requests'] for r in replica_results)
    tokens = sum(r['completion_tokens'] for r in replica_results)
    return {'first_request_started_unix': first, 'last_response_finished_unix': last,
            'client_arrival_to_last_response_seconds': wall,
            'successful_requests': good, 'errors': sum(r['errors'] for r in replica_results),
            'requests': sum(r['requests'] for r in replica_results),
            'completion_tokens': tokens, 'output_tokens_per_second': tokens / wall,
            'request_throughput_per_second': good / wall,
            'timing_scope': 'client request invocation to final response; not server arrival',
            'cross_host_combine_requires_synchronized_clocks': True}


def timed_phase(rows, endpoint, timeout, phase, concurrency, folder, api_key,
                barrier=None):
    if barrier is not None:
        barrier.wait(timeout=30)
    started_unix = time.time()
    phase_summary = b.run_phase(rows, endpoint, timeout, phase, concurrency, folder, api_key)
    finished_unix = time.time()
    records = [json.loads(line) for line in (folder / (phase + '_responses.jsonl')).read_text().splitlines()]
    # Preserve epoch timestamps with each private raw record for a later joint audit.
    for record in records:
        record['started_unix'] = utc_unix(record['started_utc'])
        record['finished_unix'] = utc_unix(record['finished_utc'])
    b.write_new(folder / (phase + '_epoch_responses.json'), records)
    result = {**phase_summary, 'phase_started_unix': started_unix,
              'phase_finished_unix': finished_unix,
              'first_request_started_unix': min(r['started_unix'] for r in records),
              'last_response_finished_unix': max(r['finished_unix'] for r in records)}
    result['client_arrival_to_last_response_seconds'] = (
        result['last_response_finished_unix'] - result['first_request_started_unix'])
    result['output_tokens_per_second_from_client_arrival'] = (
        result['completion_tokens'] / result['client_arrival_to_last_response_seconds'])
    b.write_new(folder / (phase + '_summary.json'), result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-urls', nargs='+', required=True)
    ap.add_argument('--host-label', required=True)
    ap.add_argument('--workload', type=Path, default=HERE / 'benchmark' / 'workload_v1')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--concurrency', type=int, choices=(4, 8, 16), default=8)
    ap.add_argument('--expected-tp', type=int, choices=(1, 2), default=2)
    ap.add_argument('--start-at-unix', type=float)
    ap.add_argument('--release-file', type=Path,
                    help='Wait after readiness for an operator-written JSON start_at_unix')
    ap.add_argument('--release-timeout', type=float, default=600)
    ap.add_argument('--assert-servers-idle', action='store_true')
    ap.add_argument('--api-key-env')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args(argv)
    b.require(1 <= len(a.base_urls) <= 8 // a.expected_tp,
              'Replica count exceeds eight GPUs at the selected uniform TP size')
    b.require(a.expected_tp != 1 or a.concurrency <= 8,
              'TP1 profile permits concurrency four or eight only')
    b.require(math.isfinite(a.timeout) and a.timeout > 0, 'Invalid timeout')
    b.require(a.start_at_unix is None or math.isfinite(a.start_at_unix), 'Invalid start epoch')
    b.require(not (a.start_at_unix is not None and a.release_file), 'Choose one barrier type')
    b.require(math.isfinite(a.release_timeout) and a.release_timeout > 0, 'Invalid release timeout')
    b.require(not a.release_file or not a.release_file.exists(), 'Release file must be new')
    pairs = [b.endpoint_urls(url) for url in a.base_urls]
    b.require(len({root for root, _ in pairs}) == len(pairs), 'Duplicate endpoints')
    b.require(not a.execute or a.assert_servers_idle, 'Execution and flush require explicit exclusive idleness')
    manifest, rows = b.load_workload(a.workload, 16)
    # exist_ok=False makes concurrent output ownership atomic.
    a.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    a.output.chmod(0o700)
    folders = []
    for i in range(len(pairs)):
        folder = a.output / ('replica_%02d' % i)
        folder.mkdir(mode=0o700)
        b.write_new(folder / 'phases.jsonl', {'event': 'phase_journal_created'})
        folders.append(folder)
    run = {'schema': 'bounded_teacher_host_pool.v1', 'host_label': a.host_label,
           'hostname': socket.gethostname(), 'created_utc': b.now(), 'base_urls': a.base_urls,
           'requests_per_replica': 16, 'warmup_per_replica': 4,
           'concurrency_per_replica': a.concurrency,
           'required_tensor_parallel_size': a.expected_tp,
           'gpu_count_from_uniform_profile': len(pairs) * a.expected_tp,
           'max_chat_calls': 20 * len(pairs),
           'workload_requests_sha256': manifest['requests_sha256'],
           'ordered_payload_sha256': b.digest([r['payload_sha256'] for r in rows]),
           'harness_sha256': b.file_sha(Path(__file__)), 'frozen_runner_sha256': b.file_sha(RUNNER),
           'start_at_unix': a.start_at_unix, 'execute': a.execute,
           'release_file': str(a.release_file) if a.release_file else None,
           'operator_asserted_servers_idle': a.assert_servers_idle,
           'cache_policy': 'all replicas flushed after all warmups, before measured barrier',
           'automatic_retries': False, 'student_training_allowed': False,
           'full_teacher_collection_allowed': False, 'semantic_or_training_approval': False}
    b.write_new(a.output / 'run_manifest.json', run)
    if not a.execute:
        b.write_new(a.output / 'summary.json', {**run, 'status': 'dry_run_validated', 'chat_calls': 0})
        print(json.dumps({'status': 'dry_run_validated', 'replicas': len(pairs), 'max_chat_calls': run['max_chat_calls']}))
        return
    try:
        api_key = os.environ.get(a.api_key_env) if a.api_key_env else None
        b.require(not a.api_key_env or bool(api_key), 'Missing requested API key variable')
        # Every replica passes all gates before the first warmup is sent.
        for i, (root, _) in enumerate(pairs):
            health = b.http_call(root + '/health', 'GET', None, a.timeout, api_key)
            models = b.http_call(root + '/v1/models', 'GET', None, a.timeout, api_key)
            info = b.http_call(root + '/get_server_info', 'GET', None, a.timeout, api_key)
            b.write_new(folders[i] / 'preflight.json', {'health': health, 'models': models, 'server_info': info})
            b.require(health['http_status'] == 200, 'Health failed for replica ' + str(i))
            body = models.get('body')
            b.require(models['http_status'] == 200 and isinstance(body, dict) and
                      any(x.get('id') == b.MODEL for x in body.get('data', [])), 'Model gate failed')
            dtypes = b.dtype_evidence(info.get('body'))
            b.require(info['http_status'] == 200 and any(
                x['value'].lower() in ('bfloat16', 'bf16', 'torch.bfloat16') for x in dtypes), 'BF16 gate failed')
            tps = tp_evidence(info.get('body'))
            b.require(tps and all(x['value'] == a.expected_tp for x in tps),
                      'All replicas must attest TP' + str(a.expected_tp))
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            futures = [pool.submit(timed_phase, rows[:4], url, a.timeout, 'warmup', 4, folder, api_key)
                       for (_, url), folder in zip(pairs, folders)]
            warms = [future.result() for future in futures]
        b.require(all(r['errors'] == 0 for r in warms), 'Warmup failed; measured pool not dispatched')
        for i, (root, _) in enumerate(pairs):
            result = b.http_call(root + '/flush_cache', 'POST', {}, a.timeout, api_key)
            b.write_new(folders[i] / 'flush.json', result)
            body = result.get('body')
            b.require(result['http_status'] == 200 and body is not False and not
                      (isinstance(body, dict) and (body.get('success') is False or
                       body.get('status') in ('failed', 'error', 'busy'))), 'Cache flush failed')
        ready = {'ready_unix': time.time(), 'start_at_unix': a.start_at_unix,
                 'expected_tp': a.expected_tp, 'concurrency_per_replica': a.concurrency,
                 'all_health_model_BF16_uniform_TP_warmup_flush_passed': True}
        b.write_new(a.output / 'ready.json', ready)
        print(json.dumps({'status': 'ready_for_pool_barrier', **ready}), flush=True)
        if a.release_file:
            deadline = time.monotonic() + a.release_timeout
            while not a.release_file.exists():
                b.require(time.monotonic() < deadline, 'Release timeout; no measured requests dispatched')
                time.sleep(.25)
            release = json.loads(a.release_file.read_text())
            a.start_at_unix = float(release['start_at_unix'])
            b.require(math.isfinite(a.start_at_unix), 'Invalid release epoch')
            b.write_new(a.output / 'release.json', release)
        if a.start_at_unix is not None:
            b.require(time.time() < a.start_at_unix, 'Scheduled start missed; no measured requests dispatched')
            while True:
                remaining = a.start_at_unix - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(0.25, remaining))
        barrier = threading.Barrier(len(pairs))
        phase = 'measured_c' + str(a.concurrency)
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            futures = [pool.submit(timed_phase, rows, url, a.timeout, phase, a.concurrency, folder, api_key, barrier)
                       for (_, url), folder in zip(pairs, folders)]
            measured = [future.result() for future in futures]
        total = aggregate(measured)
        summary = {**run, 'status': 'complete' if total['errors'] == 0 else 'failed_requests',
                   'host': total, 'replicas': measured, 'chat_calls': run['max_chat_calls'],
                   'completed_unix': time.time(), 'TTFT_measured': False,
                   'diagnostic_not_steady_state_peak': True}
        b.write_new(a.output / 'summary.json', summary)
        print(json.dumps({'status': summary['status'], 'host_label': a.host_label, 'host': total}), flush=True)
        if total['errors']:
            raise SystemExit(1)
    except Exception as ex:
        b.write_new(a.output / 'failure.json', {'status': 'failed', 'error_kind': type(ex).__name__,
                    'error_detail': str(ex), 'finished_unix': time.time()})
        print(json.dumps({'status': 'failed', 'error_kind': type(ex).__name__}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
