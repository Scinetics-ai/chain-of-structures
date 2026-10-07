#!/usr/bin/env python3
"""Launch only on idle assigned GPUs; preserve command and process ownership."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--run', required=True)
    p.add_argument('--memory', type=float, default=.85)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--attention-backend', default='flashinfer')
    p.add_argument('--max-mamba-cache-size', type=int)
    p.add_argument('--max-total-tokens', type=int)
    a = p.parse_args()
    devices = a.gpus.split(',')
    if not devices or len(devices) != len(set(devices)) or not all(x.isdigit() for x in devices):
        raise SystemExit('Invalid GPU indices')
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    # Serialize launch checks for all this project's replicas on a host.
    lock = (run.parent/'replica_launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (run/'launch.json').exists():
        raise SystemExit('Run already launched; select a new run path')
    for path in run.parent.glob('*/launch.json'):
        previous = json.loads(path.read_text())
        stat = Path('/proc')/str(previous['pid'])/'stat'
        if stat.exists():
            # A live reservation also protects the CUDA-context startup gap.
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and fields[19] == previous.get('proc_start_ticks'):
                if set(devices) & set(previous['gpus']):
                    raise SystemExit('GPU reserved by live task: '+str(path))
    for d in devices:
        result = subprocess.check_output(['nvidia-smi', '-i', d,
                 '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if result:
            raise SystemExit('Assigned GPU is occupied: '+d)
    with socket.socket() as s:
        s.bind(('127.0.0.1', a.port))
    cmd = [a.python, '-u', '-m', 'sglang.launch_server', '--model-path', a.model,
           '--served-model-name', 'Qwen/Qwen3.8-27B', '--host', '127.0.0.1',
           '--port', str(a.port), '--tp-size', str(len(devices)), '--dtype', 'bfloat16',
           '--context-length', '32768', '--mem-fraction-static', str(a.memory),
           '--max-running-requests', str(a.concurrency), '--chunked-prefill-size', str(a.chunk),
           '--reasoning-parser', 'qwen3', '--attention-backend', a.attention_backend]
    for name in ('max_mamba_cache_size', 'max_total_tokens'):
        value = getattr(a, name)
        if value is not None:
            cmd += ['--'+name.replace('_', '-'), str(value)]
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    # Absolute Python doesn't expose this environment's ninja/compiler helpers.
    env['PATH'] = str(Path(a.python).resolve().parent)+':/usr/local/cuda/bin:'+env.get('PATH', '')
    env.update(CUDA_VISIBLE_DEVICES=a.gpus, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONNOUSERSITE='1')
    # Keep task outputs and compilation caches away from existing training runs.
    env['TMPDIR'] = str(run.parent/'tmp')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    env['XDG_CACHE_HOME'] = str(run.parent/'cache')
    env['TORCH_EXTENSIONS_DIR'] = str(run.parent/'cache'/'torch_extensions')
    env['TRITON_CACHE_DIR'] = str(run.parent/'cache'/'triton')
    log = (run/'server.log').open('ab', buffering=0)
    process = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'hostname': socket.gethostname(), 'pid': process.pid, 'command': cmd,
               'dtype': 'bfloat16', 'tp_size': len(devices), 'port': a.port,
               'gpus': devices, 'started_unix': time.time(), 'full_collection_allowed': False,
               'student_training_allowed': False, 'health_verified': False}
    receipt['proc_start_ticks'] = (Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
    receipt['runtime_paths'] = {name: env.get(name) for name in
                               ('PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'XDG_CACHE_HOME',
                                'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR')}
    (run/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
