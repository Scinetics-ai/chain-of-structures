#!/bin/bash
# Wait for the rebuilt stage means, verify the GPUs really are empty, then launch stage A.
# The verification is not ceremony: a refused or duplicated launch still fires, and a second trainer
# on an occupied card dies on OOM 20 minutes in rather than at the start.
set -u
D=/data/caiqingran/enzymecot
M=$D/cot_split_v2/qwen35_stage_means_v2.pt
GPUS=2,3,4,5
for i in $(seq 1 120); do
  [ -f "$M" ] && break
  echo "  waiting for stage means ($i)"; sleep 15
done
[ -f "$M" ] || { echo "ABORT: stage means never appeared"; exit 1; }
echo "stage means ready: $(stat -c%s $M) bytes"
sleep 5
for g in $(echo $GPUS | tr ',' ' '); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  echo "  GPU $g uses ${u} MiB"
  [ "$u" -lt 1000 ] || { echo "ABORT: GPU $g is not empty (${u} MiB) -- something else is running"; exit 1; }
done
if pgrep -f "[d]plm2bit_textcond_hybrid.py" >/dev/null; then
  echo "ABORT: a dplm2bit trainer is already running:"; pgrep -fa "[d]plm2bit_textcond_hybrid.py" | head -3; exit 1
fi
echo "launching stage A at $(date +%H:%M:%S)"
exec bash $D/scripts/run_stage.sh A q35_4stage_A $GPUS 20000 29851
