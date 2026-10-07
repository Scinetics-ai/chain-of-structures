#!/usr/bin/env python3
"""One-shot state snapshot, no server starts or automatic workload dispatch."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--runs', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
root = Path(a.runs)
states = []
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for path in sorted(root.glob('*/launch.json')):
    launch = json.loads(path.read_text())
    entry = {'run': str(path.parent), 'pid': launch['pid'], 'gpus': launch['gpus'],
             'port': launch.get('port'), 'process_alive': False, 'health': None}
    stat = Path('/proc')/str(launch['pid'])/'stat'
    if stat.exists():
        fields = stat.read_text().rsplit(')', 1)[1].split()
        entry['process_alive'] = fields[0] != 'Z' and fields[19] == launch.get('proc_start_ticks')
    if entry['process_alive'] and entry['port']:
        try:
            with opener.open(f"http://127.0.0.1:{entry['port']}/health", timeout=3) as r:
                entry['health'] = r.status
        except Exception as ex:
            entry['health_error'] = type(ex).__name__
    states.append(entry)
receipt = {'hostname': socket.gethostname(), 'unix': time.time(), 'servers': states,
           'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                          '--format=csv'], text=True),
           'production_collection_allowed': False, 'student_training_allowed': False}
Path(a.out).write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
