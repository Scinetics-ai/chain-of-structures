#!/usr/bin/env python3
"""Full file identity audit. No model inference or data approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    model = Path(a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((model/'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values()))
    names += ['config.json', 'model.safetensors.index.json', 'tokenizer.json',
              'tokenizer_config.json', 'chat_template.jinja']
    receipt = {'hostname': socket.gethostname(), 'model': str(model.resolve()),
               'started_unix': time.time(), 'files': {}, 'complete': False}
    for name in names:
        path = model/name
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(16*1024*1024), b''):
                h.update(block)
        receipt['files'][name] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(receipt, indent=2)+'\n')
        os.replace(tmp, out)
    receipt['complete'] = True
    receipt['elapsed_seconds'] = time.time()-receipt['started_unix']
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'hostname': socket.gethostname(), 'complete': True,
                      'files': len(names), 'elapsed_seconds': receipt['elapsed_seconds']}))


if __name__ == '__main__':
    main()
