#!/usr/bin/env python3
"""Bound CVM2-CVM1 clock offset using the existing private-network SSH route."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
a=p.parse_args()
rows=[]
for _ in range(5):
    t0=time.time()
    raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
        '-i','/home/ubuntu/.ssh/id_cvm2','ubuntu@10.0.0.9',
        'python3 -c "import time; print(time.time())"'],text=True)
    t1=time.time()
    remote=float(raw.strip())
    rows.append({'local_before':t0,'local_after':t1,'remote':remote,
                 'rtt_seconds':t1-t0,'offset_midpoint_seconds':remote-(t0+t1)/2,
                 'offset_lower_seconds':remote-t1,'offset_upper_seconds':remote-t0})
best=min(rows,key=lambda r:r['rtt_seconds'])
result={'schema':'cvm_clock_offset_bound.v1','reference':'CVM1','peer':'CVM2',
        'samples':rows,'best_bound':best,
        'method':'Remote wall-clock reading lies between local send and response; RTT bounds offset'}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'offset_lower_seconds':best['offset_lower_seconds'],
                  'offset_upper_seconds':best['offset_upper_seconds']}))
