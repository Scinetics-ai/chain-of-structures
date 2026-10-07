#!/usr/bin/env python3
"""Bounded read-only GPU telemetry for the teacher benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--duration', type=float, default=600)
a = p.parse_args()
if not 0 < a.duration <= 1800:
    raise SystemExit('Duration must be in (0, 1800] seconds')
deadline = time.monotonic() + a.duration
with a.out.open('x') as f:
    while time.monotonic() < deadline:
        r = subprocess.run(['nvidia-smi',
            '--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        f.write(json.dumps({'unix': time.time(), 'returncode': r.returncode,
                            'rows': r.stdout.strip().splitlines()}) + '\n')
        f.flush()
        time.sleep(min(5, max(0, deadline-time.monotonic())))
