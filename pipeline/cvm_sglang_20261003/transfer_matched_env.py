#!/usr/bin/env python3
"""Short-lived local SSH agent; private key never copied to either server."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parent
started = time.time()
data = subprocess.check_output(['ssh-agent', '-s'], text=True)
env = os.environ.copy()
for name in ('SSH_AUTH_SOCK', 'SSH_AGENT_PID'):
    env[name] = re.search(r'^'+name+r'=([^;]+);', data, re.M).group(1)
result = {'started_unix': started, 'private_key_copied': False,
          'authorized_keys_modified': False, 'source': 'lthpc validated A800 environment'}
try:
    subprocess.run(['ssh-add', '/home/ubuntu/.ssh/bioclaw_cvm_ed25519'],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    remote = ('rsync -az --compress-level=3 --info=progress2 --stats --partial '
              '-e "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes '
              '-o UserKnownHostsFile=/data/workspace/caiqingran/agent_dplmbit_20261003/ops/cvm_benchmark_20261003/cvm2_known_hosts" '
              '/data/workspace/caiqingran/conda_envs/sglang_qwen38_teacher_v1/ '
              'ubuntu@124.221.97.180:/data/caiqingran/sglang_teacher_20261003/runtime/')
    with (root/('matched_env_transfer_'+str(int(started))+'.log')).open('w') as log:
        proc = subprocess.run(['ssh', '-A', '-S', 'none', '-o', 'ControlMaster=no',
                               '-o', 'BatchMode=yes', 'lthpc', remote], env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    result['exit_code'] = proc.returncode
finally:
    cleanup = subprocess.run(['ssh-agent', '-k'], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    result['temporary_agent_stopped'] = cleanup.returncode == 0
    result['elapsed_seconds'] = time.time()-started
    (root/'matched_env_transfer_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
if result.get('exit_code') != 0:
    raise SystemExit(result.get('exit_code', 1))
