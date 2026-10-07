#!/usr/bin/env python3
"""Release this task's two ready diagnostic pools through the private SSH route."""
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path('/data/caiqingran/sglang_teacher_20261003/benchmarks')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
       '-i', '/home/ubuntu/.ssh/id_cvm2', 'ubuntu@10.0.0.9']


def remote(code):
    return subprocess.check_output(SSH + ['python3 -c ' + shlex.quote(code)],
                                   text=True, timeout=10)


def read_local():
    root = BASE / 'cvm1_pool_tp2_c16'
    return {name: json.loads((root / (name+'.json')).read_text())
            for name in ('ready', 'run_manifest')}


left = read_local()
right = json.loads(remote(
    "import json; from pathlib import Path; p=Path(" + repr(str(BASE/'cvm2_pool_tp2_c16')) + "); "
    "print(json.dumps({n:json.loads((p/(n+'.json')).read_text()) for n in ['ready','run_manifest']}))"))
for data in (left, right):
    assert data['ready']['all_health_model_BF16_uniform_TP_warmup_flush_passed']
    manifest = data['run_manifest']
    assert manifest['execute'] and manifest['required_tensor_parallel_size'] == 2
    assert manifest['concurrency_per_replica'] == 16 and len(manifest['base_urls']) == 4
    assert not manifest['full_teacher_collection_allowed']
for key in ('harness_sha256', 'frozen_runner_sha256', 'ordered_payload_sha256',
            'workload_requests_sha256', 'cache_policy'):
    assert left['run_manifest'][key] == right['run_manifest'][key], key
path = BASE/'pool_release_v1.json'
assert not path.exists(), 'Local release already exists'
remote("from pathlib import Path; assert not Path(" + repr(str(path)) + ").exists()")
release = {'start_at_unix': time.time()+30, 'both_hosts_ready_and_matched': True,
           'student_training_allowed': False, 'full_collection_allowed': False}
payload = json.dumps(release)+'\n'
# Publish peer first; both writes must finish before the future start epoch.
remote("from pathlib import Path; import os; p=Path(" + repr(str(path)) + "); "
       "t=p.with_suffix('.tmp'); f=t.open('x'); f.write(" + repr(payload) + "); "
       "f.close(); os.replace(t,p)")
temp = path.with_suffix('.tmp')
with temp.open('x') as f:
    f.write(payload)
temp.replace(path)
assert time.time() < release['start_at_unix'], 'Release arrived too late'
print(json.dumps(release), flush=True)
