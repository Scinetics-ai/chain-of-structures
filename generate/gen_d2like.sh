#!/bin/bash
# test 管线 阶段4/4: 用 stage B 的权重 + 写手自己产的对子/文档，生成序列
# 用法: gen_ours.sh <NODE_RANK 0|1>
# 和 run_stage_2node.sh 逐项对齐，只改这些：steps=0 / generate_only_test / 不加 skip_final_generate /
# site_pairs 与 qwen_docs 换成【写手自己产的】/ 不加 --ground_noise（注入的已是真实写手对子，
# 再叠合成噪声就是双重加噪）/ test_rows 按卡分片
set -u
NR=${1:?node rank 0 or 1}
D=/data/caiqingran/enzymecot
S=$D/cot_split_v2
W=$D/test_eval_20261002_d2like_s9000
R=$D/runs/q35_deep2like_4stage_16g
CK=$R/FINAL_d2like_step9000_val0.7807.pt
PY=/data/caiqingran/env/py/bin/python
OUT=$W/gen_ours
Q=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
MEANS=$S/qwen35_stage_means_v2.pt
HASH=$S/text_embed_hash.pt
mkdir -p $OUT/logs
for f in "$CK" "$W/test_pairs_merged.pt" "$W/test_docs_merged.pt" "$MEANS" "$HASH" \
         "$S/train_rows_cotfull.json" "$S/split_seqcluster.json"; do
  [ -s "$f" ] || { echo "ABORT: 缺 $f"; exit 1; }
done
[ -n "$Q" ] || { echo "ABORT: Qwen 快照找不到"; exit 1; }

# 按长度贪心均衡分 16 片（成本随长度走，等条数会让短片的卡空等）
$PY - "$S/test_rows_cotfull.json" "$OUT" <<'PY'
import json, os, sys
rows = json.load(open(sys.argv[1])); out = sys.argv[2]
rows.sort(key=lambda r: -int(r.get("length") or 0))
sh = [[] for _ in range(16)]; load = [0]*16
for r in rows:
    j = load.index(min(load)); sh[j].append(r); load[j] += int(r.get("length") or 0)
os.makedirs(out, exist_ok=True)
for g in range(16):
    json.dump(sh[g], open("%s/test_shard%d.json" % (out, g), "w"))
print("  分片: 每片 %d-%d 条, 残基 %d-%d" % (min(len(x) for x in sh), max(len(x) for x in sh), min(load), max(load)))
PY

export PYTHONPATH=/data/caiqingran/code/dplm:/data/caiqingran/code/dplm/src:/data/caiqingran/code/dplm/vendor/openfold
export MR_TRIM_PAD=1
export MR_EVAL_SKIP_TEST=1
for g in 0 1 2 3 4 5 6 7; do
  SH=$((NR * 8 + g)); O=$OUT/shard$SH; mkdir -p $O
  for sp in train val test; do
    [ -e "$O/${sp}_dplm2_tokens.pt" ] || ln -sf $D/tokcache/${sp}_dplm2_tokens.pt $O/ 2>/dev/null || true
    [ -e "$O/${sp}_no_retrieval_tokens.pt" ] || ln -sf $D/tokcache/${sp}_no_retrieval_tokens.pt $O/ 2>/dev/null || true
  done
  CUDA_VISIBLE_DEVICES=$g setsid nohup $PY -u $D/scripts/dplm2bit_textcond_hybrid.py \
    --train_rows $S/train_rows_cotfull.json --val_rows $S/val_rows_cotfull.json \
    --test_rows $OUT/test_shard$SH.json --split_json $S/split_seqcluster.json \
    --out_dir $O --model_name /data/caiqingran/models/dplm2_bit_650m \
    --struct_tokenizer_name $D/dplm2_struct_tokenizer_hydra_wrapper \
    --init_checkpoint $CK \
    --per_token_text --max_text_tokens 576 --qwen_path $Q \
    --text_embed_pt $HASH --qwen_docs $W/test_docs_merged.pt --qwen_hidden_mean $MEANS \
    --nce_gamma -1 --nce_ratio 0.1 --nce_bank $S/text_bank_stage1_q35_len997.pt \
    --text_lambda 0.7277 --qwen_lora_r 32 \
    --site_pairs $W/test_pairs_merged.pt --ground_mode gated \
    --core_pos_json $S/core_pos.json \
    --resid_scale 0 --ctx_corrupt 0.15 \
    --max_len 9999 --steps 0 --batch 1 --unfreeze_last 8 \
    --train_limit 8 --train_eval_limit 8 --val_limit 8 --test_limit 100000 \
    --eval_interval 100000 --loss_log_interval 100000 --skip_step1_eval \
    --eval_batch_size 4 \
    --tf32 --gen_bf16 --generate_only_test --multiround_generate \
    --generate_batch_size 4 --sample_max_iter 500 --sample_strategy annealing@1.1:0.1 \
    --no_retrieval --use_mmdit_text --mmdit_layers 1 --mmdit_text_tokens 4 \
    --objective multiround --fullmask_gamma 1.0 --high_noise_gamma 0.5 \
    --struct_loss_gamma 1.0 --aa_loss_gamma 1.0 --diffusion_weighting constant \
    --use_deep_per_token_cross --deep_cross_layers 8 \
    --seed 10320 > $OUT/logs/g$SH.log 2>&1 < /dev/null &
  sleep 2
done
sleep 6
echo "node$NR: 启动 $(pgrep -f dplm2bit_textcond_hybrid | wc -l) 个, 分片 $((NR*8))-$((NR*8+7))"
