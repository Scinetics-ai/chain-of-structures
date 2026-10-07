#!/usr/bin/env bash
# Run only the explicitly named A0 protocol experiment; never advances to B/C.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'usage: run_protocol.sh CONFIG TRAIN_JSONL VALIDATION_JSONL RUN_DIRECTORY [INITIAL_CHECKPOINT]' >&2
  exit 2
fi
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
config=$1
train=$2
validation=$3
run=$4
initialization=()
if [ "$#" -eq 5 ]; then
  test -f "$5"
  initialization=(--init-checkpoint "$5")
fi
python=/data/workspace/caiqingran/conda_envs/dplmqwen35/bin/python
test -f "$config"
test -f "$train"
test -f "$validation"
mkdir -p "$run"
exec 9>"$run/launch.lock"
flock -n 9 || { echo 'This run already has a launcher.' >&2; exit 1; }
test ! -e "$run/run_identity.json" || { echo 'Use a new run directory.' >&2; exit 1; }
export PYTHONPATH="$release/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "$$" > "$run/launcher.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run/started_utc.txt"
set +e
"$python" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m agent_dplmbit.train_protocol --config "$config" --train "$train" \
  --validation "$validation" --output "$run" "${initialization[@]}"
status=$?
set -e
printf '%s\n' "$status" > "$run/exit_code.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run/finished_utc.txt"
exit "$status"
