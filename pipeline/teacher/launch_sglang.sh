#!/usr/bin/env bash
# Explicit launch only. Never interrupts training or starts corpus generation.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo 'usage: launch_sglang.sh PYTHON MODEL_DIRECTORY GPU_IDS RUN_DIRECTORY [--check-only]' >&2
  exit 2
fi
teacher_python=$1
teacher_model=$2
teacher_devices=$3
teacher_run=$4
if [ "$#" -eq 5 ] && [ "$5" != '--check-only' ]; then exit 2; fi
[[ "$teacher_devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPU_IDS must be comma-separated GPU indices.' >&2; exit 2; }
test -x "$teacher_python"
test -d "$teacher_model"
mkdir -p "$teacher_run"
exec 9>"$teacher_run/server.lock"
flock -n 9 || { echo 'A server already holds this run lock.' >&2; exit 1; }
"$teacher_python" - "$teacher_model" "$teacher_run" "$teacher_devices" <<'PY'
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from packaging.version import Version

model, run = map(Path, sys.argv[1:3])
devices = sys.argv[3].split(',')
if len(devices) != len(set(devices)):
    raise SystemExit('Duplicate GPU indices')
errors = []
index_path = model / 'model.safetensors.index.json'
if not index_path.is_file():
    raise SystemExit('Missing model.safetensors.index.json; incomplete weights directory')
index = json.loads(index_path.read_text())
shards = sorted(set(index['weight_map'].values()))
missing = [s for s in shards if not (model / s).is_file()]
if missing:
    errors.append('Missing weight shards: ' + repr(missing))
if 'Qwen3.8-27B' not in (model / 'README.md').read_text():
    errors.append('Model card does not identify the requested Qwen3.8-27B')
try:
    version = md.version('sglang')
    if Version(version) < Version('0.5.19'):
        errors.append('Use an isolated SGLang >=0.5.19 runtime; existing old environments are not approved')
except md.PackageNotFoundError:
    version = None
    errors.append('SGLang not installed in this interpreter')
import torch
cuda_runtime = torch.version.cuda
if cuda_runtime is None:
    errors.append('PyTorch has no CUDA runtime')
elif int(cuda_runtime.split('.')[0]) >= 13:
    errors.append('This lthpc driver 550 deployment requires a CUDA 12 compatible build; do not upgrade the live driver')
gpu_evidence = {}
for device in devices:
    result = subprocess.run(['nvidia-smi', '-i', device, '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    pids = [s.strip() for s in result.stdout.splitlines() if s.strip()]
    gpu_evidence[device] = pids
    if pids:
        errors.append(f'GPU {device} occupied; no job will be stopped')
receipt = {
    'time_utc': datetime.now(timezone.utc).isoformat(), 'model_id': 'Qwen/Qwen3.8-27B',
    'model_path': str(model.resolve()), 'shard_count': len(shards),
    'weight_bytes': sum((model/s).stat().st_size for s in shards if (model/s).is_file()),
    'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
    'full_weight_hash_verified': False, 'sglang': version, 'torch': torch.__version__,
    'cuda_runtime': cuda_runtime, 'gpu_processes': gpu_evidence, 'errors': errors,
    'ready_for_launch': not errors, 'generation_quality_validated': False,
}
(run / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
if errors:
    raise SystemExit(1)
PY
if [ "${5:-}" = '--check-only' ]; then exit 0; fi
IFS=',' read -ra device_array <<< "$teacher_devices"
export CUDA_VISIBLE_DEVICES="$teacher_devices"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
echo "$$" > "$teacher_run/launcher.pid"
exec "$teacher_python" -u -m sglang.launch_server \
  --model-path "$teacher_model" --served-model-name Qwen/Qwen3.8-27B \
  --host 127.0.0.1 --port "${TEACHER_PORT:-31038}" \
  --tp-size "${#device_array[@]}" --dtype bfloat16 \
  --context-length "${TEACHER_CONTEXT_LENGTH:-32768}" \
  --mem-fraction-static "${TEACHER_MEMORY_FRACTION:-0.85}" \
  --max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-16}" \
  --chunked-prefill-size 4096 --reasoning-parser qwen3 \
  >"$teacher_run/server.log" 2>&1
