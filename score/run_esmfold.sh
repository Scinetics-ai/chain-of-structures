#!/bin/bash
# 四卡并行跑 ESMFold 自洽评测。每卡一个分片(f1..f4)。
# ★HF_HOME 必须是 home_migration 那个 —— 权重只在那里, 指错会去联网然后失败。
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs out
source /home/caiqingran/miniconda3/etc/profile.d/conda.sh
conda activate /data/workspace/caiqingran/conda_envs/dplmqwen35
export PYTHONPATH=/data/workspace/caiqingran/protein_design/DISCO_repo/src:/data/workspace/caiqingran/protein_design/DISCO_repo:${PYTHONPATH:-}
export HF_HOME=/data/workspace/caiqingran/home_migration/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
S=/data/workspace/caiqingran/protein_design/DISCO_repo/text_adapter/eval_direct_esmfold_consistency_local.py
for i in 1 2 3 4; do
  gpu=$((i-1))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -u "$S" \
    --run run_f$i --split test \
    --out_csv out/esmfold_sc_f$i.csv --summary_json out/esmfold_sc_f$i.json \
    > logs/esmfold_f$i.log 2>&1 < /dev/null &
  sleep 3
done
echo "  四个 ESMFold 任务已发"
