#!/bin/bash
# 等 lthpc 反向隧道回来。★隧道是从 lthpc 那侧发起的, 本机只能等 ——
#   ~/.ssh/config 里 lthpc = 127.0.0.1:2222, 没有备用路由(已查)。
# 回来之后它会打印一行, 然后就可以发评测了(评测绝不能挂在训练机上)。
set -u
while true; do
  if timeout 15 ssh -o ConnectTimeout=10 -o BatchMode=yes lthpc 'echo UP' >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] ★lthpc 隧道回来了"
    timeout 60 ssh -o ConnectTimeout=20 lthpc 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo; echo "占卡进程 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || echo 0)"'
    exit 0
  fi
  sleep 120
done
