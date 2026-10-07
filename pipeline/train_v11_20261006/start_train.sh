#!/bin/bash
# 发两机 16 卡的 LoRA SFT。每台机器一条 ssh 命令(feedback_blocked_ssh_launch_fires_late:
# 长 ssh 命令在负载高的机器上会被掐在中间, 导致"旧的杀了新的没起来")。
# 用法: start_train.sh <node_rank> <master_addr> <模型路径> [W=6]
set -u
NR=$1; MASTER=$2; MODEL=$3; W=${4:-6}
cd "$(dirname "$0")" || exit 1
mkdir -p logs out_w$W
exec 9>/tmp/train_v11.lock
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
PY=/data/caiqingran/env/py/bin/python
setsid nohup $PY -m torch.distributed.run \
  --nnodes 2 --nproc_per_node 8 --node_rank "$NR" \
  --master_addr "$MASTER" --master_port 29531 \
  train_sft_v11.py --train sft_v11_train.jsonl --val sft_v11_val.jsonl \
  --model "$MODEL" --out "out_w$W" --answer-weight "$W" \
  --deps /data/caiqingran/pydeps_train \
  > "logs/train_node${NR}.log" 2>&1 < /dev/null &
sleep 10
echo "  node_rank=$NR 已发 | 日志 logs/train_node${NR}.log"
tail -4 "logs/train_node${NR}.log" 2>/dev/null
