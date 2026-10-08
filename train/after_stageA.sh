#!/bin/bash
# After stage A: let the writer WRITE, parse what it wrote, and fit the noise profile for all four
# stages.  Until this runs, stages 4 and 5 have no fitted profile and --ground_noise only works
# because --ground_noise_fallback 3 lends them stage 3's -- which is a placeholder, not a measurement.
#
# Usage: after_stageA.sh <CKPT> <GPU>
set -u
CKPT=${1:-/data/caiqingran/enzymecot/runs/q35_4stage_A/best.pt}
GPU=${2:-6}
D=/data/caiqingran/enzymecot
S=$D/cot_split_v2
[ -f "$CKPT" ] || { echo "ABORT: $CKPT missing"; exit 1; }
export SPLIT_DIR=$S
export PYTHONPATH=/data/caiqingran/code/dplm:/data/caiqingran/code/dplm/src:/data/caiqingran/code/dplm/vendor/openfold
export QWEN_PATH=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
export DOCS_PT=$S/qwen35_9b_docs_v2.pt        # the v2 text, so stage 1 is the prompt it trained on
P=/data/caiqingran/env/py/bin/python

echo "=== 1 let the writer generate, stage 1 as the only prompt  $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=$GPU $P -u $D/scripts/gen_cot_docs_v2.py \
  --lora "$CKPT" --rows $S/test_rows_cotfull.json \
  --out $D/gen_stageA --batch 8 --max_new 1800 --max_pos 9999 || exit 1

echo "=== 2 fit the profile for all four stages  $(date +%H:%M:%S) ==="
$P -u $D/scripts/refit_noise_v2.py \
  --ref $S/site_pairs_v2.pt \
  --gen $D/gen_stageA_pairs_0.pt \
  --rows $S/test_rows_cotfull.json || exit 1

echo
echo "Paste the printed SITE_NOISE_PROFILE / SITE_NOISE_BETA into the trainer, then drop"
echo "--ground_noise_fallback from the stage B line so a missing profile is fatal again."
echo "Read-out thresholds fixed before the run: stage 4 pos must beat 0.30 (picking 24 of the 80"
echo "unnamed positions in the +-4 window at random), stage 4 exact ceiling is 0.098 (background),"
echo "stage 5 exact = pos x 0.156 means identity is only the prior. deep2: s2 0.232 / s3 0.185."
