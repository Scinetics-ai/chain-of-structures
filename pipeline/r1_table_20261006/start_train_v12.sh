#!/bin/bash
# v12: 发两机 16 卡的 LoRA SFT。每台机器一条 ssh 命令(feedback_blocked_ssh_launch_fires_late)。
#
# ★v12 和 v11 的三个差别:
#   1 R1 这一轮现在**有候选表**(模板检索 + p/L x EC 先验), 所以 R1 的输入变长了
#   2 验证集拆成两份, **分开报, 永远不要平均**:
#       val_overlap —— 和 test_overlap(占测试集 97%)同分布, **决策/早停看这个**
#       val_strict  —— 全新反应的诚实读数, 表质量低 10 分(51.5% vs 61.0%)
#     v11 那个反应不交的 val 比 test 难得多, 拿它做决策会判错。
#   3 必须先在【每台机器】各跑一次 cache, 再发 torchrun ——
#     在训练进程里建缓存会让其余 rank 在屏障上等到 NCCL 超时(实测 13.5 分钟 > 默认 10 分钟)。
# 用法: start_train_v12.sh cache <模型路径> [W=6]            # 先在每台跑这个
#       start_train_v12.sh <node_rank> <master_addr> <模型路径> [W=6]
set -u
TR=sft_v12_train.jsonl; VA=sft_v12_val_overlap.jsonl; VA2=sft_v12_val_strict.jsonl
cd "$(dirname "$0")" || exit 1
for f in "$TR" "$VA" "$VA2"; do
  [ -s "$f" ] || { echo "★缺数据文件 $f"; exit 1; }
done
PY=/data/caiqingran/env/py/bin/python
if [ "${1:-}" = cache ]; then
  MODEL=$2; W=${3:-6}
  mkdir -p logs
  echo "  建缓存(本机单进程)..."
  $PY -u train_sft_v12.py --train "$TR" --val "$VA" --val2 "$VA2" \
    --model "$MODEL" --out "out_v12_w$W" --answer-weight "$W" \
    --deps /data/caiqingran/pydeps_train --build-cache-only 2>&1 | tail -6
  exit $?
fi
NR=$1; MASTER=$2; MODEL=$3; W=${4:-6}
mkdir -p logs out_v12_w$W
exec 9>/tmp/train_v12.lock
flock -n 9 || { echo "★已有训练在发, 退出"; exit 0; }
# ★NCCL_IB_DISABLE=1 必须有(feedback_cvm2_nccl_ib_segfault): cvm-2 上坏的软 RoCE 设备 rxe0
#   会让 NCCL 段错误 —— 单卡好、多卡全崩。另外两机关 GDR、关 NVLS。
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
# ★降显存碎片: 实测每卡从 40GB 涨到 57GB(上限 72GB), 偏紧。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ★前置闸: 等显存【真的】还回来再起。被杀的进程会从 pgrep 里消失但显存还攥着
#   (这次实测: 杀完立刻读是 0 MiB, 过一会儿冒出个进程占着 62.66GB, 新任务直接 CUDA OOM)。
for i in $(seq 1 60); do
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  if [ "${BUSY:-0}" -eq 0 ] && [ "${USED:-999999}" -lt 2000 ]; then break; fi
  echo "  等显存释放: 占卡进程 ${BUSY} / 已用 ${USED} MiB"
  sleep 10
done
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [ "${BUSY:-0}" -ne 0 ]; then echo "★仍有 ${BUSY} 个进程占卡, 不发训练"; exit 1; fi
setsid nohup $PY -m torch.distributed.run \
  --nnodes 2 --nproc_per_node 8 --node_rank "$NR" \
  --master_addr "$MASTER" --master_port 29532 \
  train_sft_v12.py --train "$TR" --val "$VA" --val2 "$VA2" \
  --model "$MODEL" --out "out_v12_w$W" --answer-weight "$W" \
  --deps /data/caiqingran/pydeps_train \
  > "logs/train_node${NR}.log" 2>&1 < /dev/null &
sleep 10
echo "  node_rank=$NR 已发 | 日志 logs/train_node${NR}.log"
tail -4 "logs/train_node${NR}.log" 2>/dev/null
