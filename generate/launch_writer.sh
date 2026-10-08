#!/bin/bash
set -euo pipefail
NR=${1:?node rank}
W=/data/caiqingran/enzymecot/test_eval_20261002_d2like_s9000
PY=/data/caiqingran/env/py/bin/python
mkdir -p "$W/logs" "$W/status"
for g in {0..7}; do
 sh=$((NR*8+g))
 if [ -s "$W/status/shard$sh.json" ]; then echo "Refusing duplicate shard $sh";exit 1;fi
 CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" -u "$W/code/eval_worker.py" "$sh" "$W" > "$W/logs/worker_$sh.log" 2>&1 < /dev/null &
 echo "shard=$sh gpu=$g pid=$!"
 sleep 1
done
