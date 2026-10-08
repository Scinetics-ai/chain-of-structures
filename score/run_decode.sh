#!/bin/bash
# 四卡并行把 4 段注入臂的结构 token 解成 PDB。每张卡包一个分片(f1..f4), 互不干扰。
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs
source /home/caiqingran/miniconda3/etc/profile.d/conda.sh
conda activate /data/workspace/caiqingran/conda_envs/dplmqwen35
export PYTHONPATH=/data/workspace/caiqingran/protein_design/DISCO_repo/src:/data/workspace/caiqingran/protein_design/DISCO_repo:${PYTHONPATH:-}
G=/data/workspace/caiqingran/protein_design/DISCO_repo/runs/EVAL_KEEP_deep4b/gen
for i in 1 2 3 4; do
  gpu=$((i-1))
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup python -u decode_struct_to_pdb.py \
    $G/f$i /data/workspace/caiqingran/metrics_deep4b/pdb_f$i \
    > logs/decode_f$i.log 2>&1 < /dev/null &
  sleep 2
done
echo "  四个解码任务已发"
