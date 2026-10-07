#!/bin/bash
# v12 R1 离线配对评测。★绝不挂在训练机上(feedback_never_run_eval_on_the_training_machine:
# 显存够 ≠ 不影响)。cvm-1/cvm-2 在训练时, 这个只能跑在 lthpc。
#
# 要对着比的三条线(都已算好, 见 lookup_only_baseline.py / stratified_headroom.py):
#   表内随机(test_overlap)      位置 13.0%  氨基酸@选对 43.7%  整对  5.7%
#   查表-only 最强手写 DP        位置 34.7%  氨基酸@选对 67.4%  整对 23.4%   ← **要打赢这条**
#   天花板(表覆盖)              对整档召回 59.5%
# 分层看(test_overlap 的 gold 位置):
#   tplD 0-1  占 47.7%  查表 tplAA 已对 57.5% ⇒ agent 最多持平
#   tplD >=2  占 52.3%  查表 tplAA 只对 19.3%/0% ⇒ **agent 的战场在这里**
#
# ★max-new 取 3200: 1200 会把后面几轮截断(v11 实测 32 条里 17 条), JSON 合法率假塌到 3%;
#   v12 的参考输出最大 3,031 token, 2400 只覆盖 99.76%。生成遇 EOS 就停, 抬高上限零代价。
# 用法: run_eval_v12.sh <base模型> <adapter目录> <评测集> <每组条数> [卡数=4]
set -u
BASE=$1; AD=$2; VAL=$3; PER=$4; NG=${5:-4}
cd "$(dirname "$0")" || exit 1
mkdir -p logs out
for f in "$VAL" eval_pairs_v12.py; do [ -s "$f" ] || { echo "★缺 $f"; exit 1; }; done
[ -d "$AD" ] || { echo "★adapter 目录不存在: $AD"; exit 1; }
# ★前置闸: 本机不许有别的进程占卡
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
[ "${BUSY:-0}" -ne 0 ] && { echo "★本机有 ${BUSY} 个进程占卡, 不发评测"; exit 1; }
OV=${OV:-test_overlap_pids.txt}; ST=${ST:-test_strict_pids.txt}
for f in "$OV" "$ST"; do [ -s "$f" ] || { echo "★缺分组文件 $f"; exit 1; }; done
# ★强制先冒烟: 统计管线已用"完美预测"验过, 但【真模型路径】(加载/批生成/左填充)从没跑过。
#   先用 8 条跑一遍, 过了再发全量 —— 否则崩在模型路径上会白等一轮。
#   设 SKIP_SMOKE=1 可跳过(只在刚刚冒烟过的情况下)。
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "  冒烟 8 条(真模型路径)..."
  head -8 "$VAL" > /tmp/ev_smoke.jsonl
  CUDA_VISIBLE_DEVICES=0 timeout 1800 python -u eval_pairs_v12.py \
    --base "$BASE" --adapter "$AD" --val /tmp/ev_smoke.jsonl \
    --per-round 8 --batch 2 --max-new 3200 \
    --group-pids "test_overlap=$OV,test_strict=$ST" \
    --out /tmp/ev_smoke_out.jsonl > logs/ev_smoke.log 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "★冒烟失败(exit $RC), 不发全量。日志尾部:"
    grep -vE "it/s\]|%\|" logs/ev_smoke.log | tail -15
    exit 1
  fi
  OK=$(grep -c . /tmp/ev_smoke_out.jsonl 2>/dev/null || echo 0)
  [ "${OK:-0}" -lt 1 ] && { echo "★冒烟没产出记录, 不发全量"; tail -10 logs/ev_smoke.log; exit 1; }
  echo "  冒烟过了($OK 条), 发全量"
fi
for i in $(seq 0 $((NG-1))); do
  ( exec 9>"logs/.ev.$i"
    flock -n 9 || { echo "分片 $i 锁被占"; exit 0; }
    CUDA_VISIBLE_DEVICES=$i setsid nohup python -u eval_pairs_v12.py \
      --base "$BASE" --adapter "$AD" --val "$VAL" \
      --per-round "$PER" --batch 8 --max-new 3200 \
      --group-pids "test_overlap=$OV,test_strict=$ST" \
      --shard "$i" --num-shards "$NG" --out "out/ev_s$i.jsonl" \
      > "logs/ev_s$i.log" 2>&1 < /dev/null &
    sleep 3 )
done
sleep 10
N=$(printf "eval_pairs_v1%s.py" "2")
echo "  起了 $(pgrep -fc "$N") 个分片 | 每组 $PER 条 | max-new 3200"
echo "  汇总: python3 -c \"见 README; 或直接看各分片日志尾部的分层表\""
