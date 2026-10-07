#!/usr/bin/env python3
"""Stop one task-owned replica, with PID start-time verification."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

p=argparse.ArgumentParser()
p.add_argument('--run', required=True)
a=p.parse_args()
root=Path(a.run)
r=json.loads((root/'launch.json').read_text())
pid=r['pid']
stat=Path('/proc')/str(pid)/'stat'
out={'pid':pid,'run':str(root),'time_unix':time.time(),'signal_sent':False}
if not stat.exists():
    out['status']='already_exited'
else:
    fields=stat.read_text().rsplit(')',1)[1].split()
    if fields[19]!=r['proc_start_ticks'] or os.getpgid(pid)!=pid:
        raise SystemExit('PID identity/group mismatch; refusing to signal')
    if fields[0]=='Z':
        out['status']='already_exited'
    else:
        os.killpg(pid,signal.SIGTERM)
        out['signal_sent']=True
        out['status']='term_sent'
        # Model runner normally cascades to its schedulers on TERM.
        for _ in range(100):
            if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
                out['status']='parent_exited_after_term'
                break
            time.sleep(.1)
        else:
            fields=stat.read_text().rsplit(')',1)[1].split()
            if fields[19]!=r['proc_start_ticks']:
                raise SystemExit('PID changed during shutdown; refusing SIGKILL')
            os.killpg(pid,signal.SIGKILL)
            out['status']='kill_sent_after_term_timeout'
(root/'stop.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
