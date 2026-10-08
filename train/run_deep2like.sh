#!/bin/bash
# deep2 的配方，只把 step4/5 的离散注入加上。逐项对照 run_q35_joint.sh(deep2 实际用的):
#   相同: --lr 5e-5 --cond_lr 3e-4 --qwen_lr 1e-4 --qwen_lora_r 32 --qwen_freeze_steps 1000
#         --ground_mode gated --max_len 9999 --batch 1 --unfreeze_last 8
#         --use_deep_per_token_cross --deep_cross_layers 8  4 卡(有效批量 4)
#         【不加】--ground_noise / --ground_cumulative —— deep2 没有，训的是干净对子
#   只改三处: site_pairs / qwen_docs / qwen_hidden_mean 全部 v1 -> v2
#            (后两个必须跟着换，否则 step4/5 的文本不点名残基，写手学不会在那两段报位置)
set -u
NAME=${1:-q35_deep2like_4stage}
STEPS=${2:-25210}
PORT=${3:-29851}
D=/data/caiqingran/enzymecot
S=$D/cot_split_v2
OUT=$D/runs/$NAME
INIT=$D/stage1_q35/best.pt
PY=/data/caiqingran/env/py/bin/python
Q=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
mkdir -p "$OUT" "$D/logs"
for f in "$INIT" "$S/site_pairs_v2.pt" "$S/qwen35_9b_docs_v2.pt" "$S/qwen35_stage_means_v2.pt" \
         "$S/text_bank_stage1_q35_len997.pt" "$S/text_embed_hash.pt" "$S/core_pos.json"; do
  [ -s "$f" ] || { echo "ABORT: 缺 $f"; exit 1; }
done
[ -n "$Q" ] || { echo "ABORT: Qwen 快照找不到"; exit 1; }
for f in train_dplm2_tokens.pt val_dplm2_tokens.pt test_dplm2_tokens.pt \
         train_no_retrieval_tokens.pt val_no_retrieval_tokens.pt test_no_retrieval_tokens.pt; do
  [ -e "$OUT/$f" ] || [ ! -f "$D/tokcache/$f" ] || ln -sf "$D/tokcache/$f" "$OUT/$f"
done
export PYTHONPATH=$D/../code/dplm:$D/../code/dplm/src:$D/../code/dplm/vendor/openfold
export PYTHONPATH=/data/caiqingran/code/dplm:/data/caiqingran/code/dplm/src:/data/caiqingran/code/dplm/vendor/openfold
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_NVLS_ENABLE=0
export MR_TEXT_DROPOUT=0.1 EPOCH_SAMPLER=1 MR_TRIM_PAD=1
export DPLM_GRAD_CKPT=1 DPLM_CKPT_MIN_TOKENS=1100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MR_EVAL_SKIP_TEST=1
echo "name=$NAME steps=$STEPS 4卡(有效批量4) site_pairs=v2 docs=v2 ground_noise=无"
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup $PY -m torch.distributed.run \
  --standalone --nproc_per_node=4 --master_port=$PORT \
  $D/scripts/dplm2bit_textcond_hybrid.py \
  --train_rows  $S/train_rows_cotfull.json \
  --val_rows    $S/val_rows_cotfull.json \
  --test_rows   $S/test_rows_cotfull.json \
  --split_json  $S/split_seqcluster.json \
  --out_dir     $OUT \
  --model_name  /data/caiqingran/models/dplm2_bit_650m \
  --struct_tokenizer_name $D/dplm2_struct_tokenizer_hydra_wrapper \
  --init_checkpoint $INIT \
  --per_token_text --max_text_tokens 576 \
  --qwen_path $Q \
  --text_embed_pt $S/text_embed_hash.pt \
  --qwen_docs $S/qwen35_9b_docs_v2.pt \
  --qwen_hidden_mean $S/qwen35_stage_means_v2.pt \
  --nce_gamma -1 --nce_ratio 0.1 --nce_bank $S/text_bank_stage1_q35_len997.pt \
  --text_lambda 0.7277 \
  --qwen_lr 1e-4 --qwen_lora_r 32 --qwen_freeze_steps 1000 \
  --site_pairs $S/site_pairs_v2.pt --ground_mode gated \
  --core_pos_json $S/core_pos.json \
  --resid_scale 0 --ctx_corrupt 0.15 \
  --max_len 9999 --steps $STEPS --batch 1 --lr 5e-5 --cond_lr 3e-4 --unfreeze_last 8 \
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
  > $D/logs/${NAME}.log 2>&1 < /dev/null &
sleep 8
echo "已发起, 日志 $D/logs/${NAME}.log"
