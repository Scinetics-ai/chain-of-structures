#!/bin/bash
# 16 GPUs across cvm-2 (node 0, master 10.0.0.9) and cvm-1 (node 1).
# Usage: [SCRATCH=1] run_stage_2node.sh <A|B> <NODE_RANK> [NAME] [STEPS] [PORT]
#
# SCRATCH=1 drops --init_checkpoint so the writer is trained from the base Qwen. Inheriting
# stage1_q35 is what made s4 and s5 start at ~0.95 while s2 and s3 sat at ~0.45: that checkpoint was
# trained on the OLD text, where only steps 2 and 3 named residues. peft initialises lora_B at zero,
# so a scratch adapter is a no-op at step 0 and the model learns all five spans.
#
# --text_embed_pt loads the precomputed text_hash table. Without it every rank spends 17 minutes at
# startup on ~5000 md5 calls per row, each followed by a scalar index-add on a torch tensor.
#
# Effective batch is the world size (--batch is 1 per rank): deep2 ran 4 GPUs = batch 4, this runs 16.
# Learning rates are scaled by sqrt(4)=2 (cond 3e-4 -> 6e-4, qwen 1e-4 -> 2e-4). --lr is left alone
# because stage A freezes the DPLM backbone, so nothing reads it.
#
# Verified on these boxes: 16-rank cross-node all_reduce returns 136 for world=16, a 600 MB reduce
# takes 196 ms. NCCL_IB_DISABLE=1 is mandatory (rxe0 segfaults NCCL); NCCL_SOCKET_IFNAME=eth0 pins it
# to the 10.0.0.0/24 LAN measured at 953 MB/s.
set -u
STAGE=${1:?A or B}
NR=${2:?node rank}
NAME=${3:-q35_4stage_${STAGE}_16g}
STEPS=${4:-20000}
PORT=${5:-29861}
SCRATCH=${SCRATCH:-0}
D=/data/caiqingran/enzymecot
S=$D/cot_split_v2
OUT=$D/runs/$NAME
mkdir -p "$OUT" "$D/logs"

case "$STAGE" in
  A) UNFREEZE=0; INIT=$D/stage1_q35/best.pt ;;
  # INIT_OVERRIDE lets a restart resume from this stage's own checkpoint instead of stage A's, which
  # is what it must do after an interruption: stage A's best.pt would silently discard every joint step.
  B) UNFREEZE=8; INIT=${INIT_OVERRIDE:-$D/runs/q35_4stage_A_16g/best.pt} ;;
  *) echo "stage must be A or B"; exit 1 ;;
esac
if [ "$SCRATCH" = "1" ]; then
  INIT_ARG=""
  echo "SCRATCH=1: no --init_checkpoint, the writer starts from the base model"
else
  [ -f "$INIT" ] || { echo "ABORT: init checkpoint missing: $INIT"; exit 1; }
  INIT_ARG="--init_checkpoint $INIT"
fi
for f in train_dplm2_tokens.pt val_dplm2_tokens.pt test_dplm2_tokens.pt \
         train_no_retrieval_tokens.pt val_no_retrieval_tokens.pt test_no_retrieval_tokens.pt; do
  [ -e "$OUT/$f" ] || [ ! -f "$D/tokcache/$f" ] || ln -sf "$D/tokcache/$f" "$OUT/$f"
done

export PYTHONPATH=/data/caiqingran/code/dplm:/data/caiqingran/code/dplm/src:/data/caiqingran/code/dplm/vendor/openfold
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NVLS_ENABLE=0
export MR_TEXT_DROPOUT=0.1
export EPOCH_SAMPLER=1
export MR_TRIM_PAD=1
export DPLM_GRAD_CKPT=1
export DPLM_CKPT_MIN_TOKENS=1100
export MR_EVAL_SKIP_TEST=1
export MEM_PREFLIGHT_IDX=1235,47023,38450,14311,14315,14314,14313,14312
# stage A never wrote a periodic checkpoint because CKPT_EVERY defaults to 0 and this was not
# exported; only best.pt existed, so there was no way back to a specific step.
export CKPT_EVERY=2000
Q=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
[ -n "$Q" ] || { echo "ABORT: no Qwen snapshot"; exit 1; }
MEANS=$S/qwen35_stage_means_v2.pt
[ -f "$MEANS" ] || { echo "ABORT: $MEANS missing"; exit 1; }
HASH=$S/text_embed_hash.pt
[ -f "$HASH" ] || { echo "ABORT: $HASH missing -- startup would spend 17 min hashing"; exit 1; }

echo "node_rank=$NR stage=$STAGE name=$NAME steps=$STEPS unfreeze_last=$UNFREEZE world=16 batch=16"
echo "init=${INIT_ARG:-<scratch>}"

/data/caiqingran/env/py/bin/python -m torch.distributed.run \
  --nnodes=2 --node_rank=$NR --nproc_per_node=8 \
  --master_addr=10.0.0.9 --master_port=$PORT \
  $D/scripts/dplm2bit_textcond_hybrid.py \
  --train_rows  $S/train_rows_cotfull.json \
  --val_rows    $S/val_rows_cotfull.json \
  --test_rows   $S/test_rows_cotfull.json \
  --split_json  $S/split_seqcluster.json \
  --out_dir     $OUT \
  --model_name  /data/caiqingran/models/dplm2_bit_650m \
  --struct_tokenizer_name $D/dplm2_struct_tokenizer_hydra_wrapper \
  $INIT_ARG \
  --per_token_text --max_text_tokens 576 \
  --qwen_path $Q \
  --text_embed_pt $HASH \
  --qwen_docs $S/qwen35_9b_docs_v2.pt \
  --qwen_hidden_mean $MEANS \
  --nce_gamma -1 --nce_ratio 0.1 --nce_bank $S/text_bank_stage1_q35_len997.pt \
  --text_lambda 0.7277 \
  --qwen_lr 2e-4 --qwen_lora_r 32 --qwen_freeze_steps 1000 \
  --site_pairs $S/site_pairs_v2.pt --ground_mode gated \
  --ground_cumulative --ground_noise --ground_noise_fallback 3 \
  --core_pos_json $S/core_pos.json \
  --resid_scale 0 --ctx_corrupt 0.15 \
  --max_len 9999 --steps $STEPS --batch 1 --lr 5e-5 --cond_lr 6e-4 --unfreeze_last $UNFREEZE \
  --train_eval_limit 512 --val_limit 252 --test_limit 1350 \
  --eval_interval 1000 --loss_log_interval 50 \
  --eval_batch_size 4 --mr_eval_limit 256 \
  --generate_batch_size 1 --sample_max_iter 500 --sample_strategy annealing@1.1:0.1 \
  --skip_step1_eval --skip_final_generate --no_retrieval \
  --tf32 \
  --use_mmdit_text --mmdit_layers 1 --mmdit_text_tokens 4 \
  --objective multiround --fullmask_gamma 1.0 --high_noise_gamma 0.5 \
  --struct_loss_gamma 1.0 --aa_loss_gamma 1.0 --diffusion_weighting constant \
  --seed 10320 \
  --use_deep_per_token_cross --deep_cross_layers 8 \
  --wandb_project protein_design_dplm2 --wandb_run $NAME \
  2>&1 | tee -a $D/logs/${NAME}_n${NR}.log
