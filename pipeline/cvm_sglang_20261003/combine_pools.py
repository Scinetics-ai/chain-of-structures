#!/usr/bin/env python3
"""Combine exactly two completed pools on the reference host's clock.

clock_audit.py records peer(CVM2) minus reference(CVM1), so peer timestamps
are corrected by SUBTRACTING the offset. Bounds are conditional on the peer
offset staying inside the audit envelope throughout both measured intervals.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

MATCH = ('workload_requests_sha256', 'ordered_payload_sha256', 'harness_sha256',
         'frozen_runner_sha256', 'required_tensor_parallel_size',
         'concurrency_per_replica', 'requests_per_replica', 'cache_policy')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_pool(folder):
    path = folder / 'summary.json'
    data = load(path)
    require(data.get('schema') == 'bounded_teacher_host_pool.v1', 'Unexpected pool schema')
    require(data.get('status') == 'complete' and data.get('execute') is True,
            'Only complete executed pools accepted')
    require(data.get('semantic_or_training_approval') is False, 'Missing no-approval contract')
    for key in MATCH:
        require(key in data and data[key] is not None, 'Missing matched field: ' + key)
    replicas = data.get('replicas', [])
    require(isinstance(replicas, list) and bool(replicas), 'Missing replica summaries')
    finishes = Counter()
    for r in replicas:
        require(r['errors'] == 0 and r['successful_requests'] == r['requests'], 'Replica had failed requests')
        require(r['requests'] == data['requests_per_replica'], 'Replica workload size differs')
        require(r['concurrency'] == data['concurrency_per_replica'], 'Replica concurrency differs')
        require(type(r['completion_tokens']) is int and r['completion_tokens'] >= 0, 'Invalid token count')
        require(finite(r['first_request_started_unix']) and finite(r['last_response_finished_unix']) and
                r['last_response_finished_unix'] > r['first_request_started_unix'], 'Invalid replica timestamps')
        reasons = r['finish_reasons']
        require(set(reasons) <= {'stop', 'length'} and
                all(type(v) is int and v >= 0 for v in reasons.values()) and
                sum(reasons.values()) == r['requests'], 'Invalid finish-reason accounting')
        finishes.update(reasons)
    first = min(r['first_request_started_unix'] for r in replicas)
    last = max(r['last_response_finished_unix'] for r in replicas)
    tokens = sum(r['completion_tokens'] for r in replicas)
    requests = sum(r['requests'] for r in replicas)
    recorded = data['host']
    require(recorded['completion_tokens'] == tokens and recorded['requests'] == requests and
            recorded['errors'] == 0, 'Host totals disagree with replicas')
    require(abs(recorded['first_request_started_unix'] - first) < 1e-6 and
            abs(recorded['last_response_finished_unix'] - last) < 1e-6,
            'Host timestamps disagree with replicas')
    return data, {'host_label': data['host_label'], 'hostname': data['hostname'],
                  'summary_path': str(path.resolve()), 'summary_sha256': sha(path),
                  'first_request_started_unix': first, 'last_response_finished_unix': last,
                  'wall_seconds': last - first, 'completion_tokens': tokens, 'requests': requests,
                  'finish_reasons': dict(finishes), 'output_tokens_per_second': tokens / (last - first),
                  'normal_stop_request_throughput': finishes['stop'] / (last - first),
                  'length_limited_requests': finishes['length'], 'semantic_or_training_approval': False}


def clock_bound(path):
    value = load(path)
    bound = value.get('best_bound', value)
    lower, upper = bound['offset_lower_seconds'], bound['offset_upper_seconds']
    require(finite(lower) and finite(upper) and lower <= upper, 'Invalid peer-minus-reference clock bounds')
    return value, {'path': str(path.resolve()), 'sha256': sha(path),
                   'reference': value.get('reference'), 'peer': value.get('peer'),
                   'offset_lower_seconds': lower, 'offset_upper_seconds': upper,
                   'sample_local_before': bound.get('local_before'),
                   'sample_local_after': bound.get('local_after')}


def effective_start(folder, data):
    if data.get('release_file'):
        path = folder / 'release.json'
        release = load(path)
        start = release.get('start_at_unix')
        require(finite(start), 'Missing or invalid recorded release epoch')
        return start, {'path': str(path.resolve()), 'sha256': sha(path),
                       'start_at_unix': start}
    start = data.get('start_at_unix')
    require(start is None or finite(start), 'Invalid scheduled start epoch')
    return start, None


def combine(reference_run, peer_run, clock_path, clock_after=None, extra_bound=0.0):
    require(reference_run.resolve() != peer_run.resolve(), 'Two distinct run directories required')
    ref, host1 = read_pool(reference_run)
    peer, host2 = read_pool(peer_run)
    require(ref['hostname'] != peer['hostname'], 'Two distinct hosts required')
    for key in MATCH:
        require(ref[key] == peer[key], 'Pool configuration mismatch: ' + key)
    start1, release1 = effective_start(reference_run, ref)
    start2, release2 = effective_start(peer_run, peer)
    require(start1 == start2, 'Effective release/scheduled start epochs differ')
    require(finite(extra_bound) and extra_bound >= 0, 'Invalid additional clock uncertainty')
    _, before = clock_bound(clock_path)
    clocks = [before]
    if clock_after is not None:
        _, after = clock_bound(clock_after)
        require((before['reference'], before['peer']) == (after['reference'], after['peer']),
                'Before/after clock sign or host mapping differs')
        clocks.append(after)
    lower = min(c['offset_lower_seconds'] for c in clocks) - extra_bound
    upper = max(c['offset_upper_seconds'] for c in clocks) + extra_bound
    midpoint = (lower + upper) / 2
    s1, e1 = host1['first_request_started_unix'], host1['last_response_finished_unix']
    s2, e2 = host2['first_request_started_unix'], host2['last_response_finished_unix']
    start, end = min(s1, s2 - midpoint), max(e1, e2 - midpoint)
    # Conservative endpoint bounds permit offset drift anywhere in the envelope.
    min_wall = max(e1, e2 - upper) - min(s1, s2 - lower)
    max_wall = max(e1, e2 - lower) - min(s1, s2 - upper)
    require(min_wall > 0 and max_wall >= min_wall, 'Clock uncertainty invalidates timing interval')
    overlap_lower = min(e1, e2 - upper) - max(s1, s2 - lower)
    require(overlap_lower > 0, 'Audited intervals do not establish concurrent host execution')
    tokens = host1['completion_tokens'] + host2['completion_tokens']
    finishes = Counter(host1['finish_reasons']) + Counter(host2['finish_reasons'])
    host2['first_request_started_reference_clock_estimate'] = s2 - midpoint
    host2['last_response_finished_reference_clock_estimate'] = e2 - midpoint
    return {'schema': 'bounded_teacher_joint_pool.v1', 'status': 'complete',
            'matched_profile': {key: ref[key] for key in MATCH},
            'hosts': [host1, host2], 'clock_evidence': clocks,
            'effective_common_start_at_unix': start1,
            'release_evidence': [release1, release2],
            'clock_offset_definition': 'peer(CVM2) minus reference(CVM1); subtract offset from peer timestamps',
            'reference_run_is_clock_reference': True, 'peer_run_is_clock_peer': True,
            'clock_assumption': 'Offset remains within the before/after audit envelope plus extra bound throughout both runs; wall-clock jumps are not independently ruled out.',
            'effective_offset_lower_seconds': lower, 'effective_offset_upper_seconds': upper,
            'additional_clock_uncertainty_seconds': extra_bound,
            'joint': {'completion_tokens': tokens, 'requests': host1['requests'] + host2['requests'],
                      'first_request_started_reference_clock_estimate': start,
                      'last_response_finished_reference_clock_estimate': end,
                      'wall_seconds_estimate': end - start,
                      'wall_seconds_lower_bound': min_wall, 'wall_seconds_upper_bound': max_wall,
                      'output_tokens_per_second': tokens / (end - start),
                      'output_tokens_per_second_lower_bound': tokens / max_wall,
                      'output_tokens_per_second_upper_bound': tokens / min_wall,
                      'finish_reasons': dict(finishes), 'length_limited_requests': finishes['length'],
                      'normal_stop_request_throughput': finishes['stop'] / (end - start),
                      'normal_stop_request_throughput_lower_bound': finishes['stop'] / max_wall,
                      'normal_stop_request_throughput_upper_bound': finishes['stop'] / min_wall,
                      'guaranteed_host_overlap_seconds': overlap_lower},
            'aggregation': 'sum actual output tokens / global earliest client invocation to latest response; never sum host throughput',
            'diagnostic_not_steady_state_peak': True, 'semantic_or_training_approval': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-run', type=Path, required=True, help='CVM1 pool run directory')
    ap.add_argument('--peer-run', type=Path, required=True, help='CVM2 pool run directory')
    ap.add_argument('--clock-offset', type=Path, required=True)
    ap.add_argument('--clock-after', type=Path)
    ap.add_argument('--extra-clock-bound-seconds', type=float, default=0.0)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    result = combine(a.reference_run, a.peer_run, a.clock_offset, a.clock_after,
                     a.extra_clock_bound_seconds)
    result['combiner_sha256'] = sha(Path(__file__))
    fd = os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'complete', 'joint': result['joint']}))


if __name__ == '__main__':
    main()
