#!/bin/bash
# 持续观测训练: 每 5 分钟采一次, 用【所有窗口的累计】算真实平均, 不看单个窗口。
# ★为什么不能看单个窗口(今天踩的): 按长度分桶后, 不同 20 步窗口的序列长度差 4 倍,
#   单窗口的 s/步 和 tok/s 都不可比。只有"累计 token / 累计时间"是可比的。
set -u
H=${1:-cvm-1}; N=${2:-0}; D=/data/caiqingran/train_v11
for i in $(seq 1 240); do
  R=$(timeout 70 ssh -o ConnectTimeout=25 "$H" "cd $D && python3 - <<'PY'
import re, time, os
p='logs/train_node$N.log'
txt=open(p, errors='ignore').read()
steps=re.findall(r'步 (\d+)/(\d+)', txt)
sp=[float(x) for x in re.findall(r'([0-9.]+)s/步', txt)]
tk=[float(x) for x in re.findall(r'总 ([0-9.]+)k tok/s', txt)]
ans=re.findall(r'答案 CE ([0-9.]+) 准确 ([0-9.]+)%', txt)
pro=re.findall(r'散文 CE ([0-9.]+) 准确 ([0-9.]+)%', txt)
if not steps: print('NOSTEP'); raise SystemExit
cur, tot = int(steps[-1][0]), int(steps[-1][1])
# 真实平均: 每个窗口 20 步, 窗口内 s/步 x 20 = 窗口耗时; tok/s x 窗口耗时 = 窗口 token
secs=sum(s*20 for s in sp); toks=sum(t*1000*s*20 for t, s in zip(tk, sp))
a0,a1=ans[0] if ans else ('-','-'); an,am=ans[-1] if ans else ('-','-')
p0,p1=pro[0] if pro else ('-','-'); pn,pm=pro[-1] if pro else ('-','-')
print(f'步 {cur}/{tot} | 窗口数 {len(sp)} | 真实均值 {secs/max(len(sp)*20,1):.1f}s/步 '
      f'{toks/max(secs,1)/1000:.1f}k tok/s | 整轮预计 {tot*secs/max(len(sp)*20,1)/3600:.1f}h')
print(f'  答案 CE {a0}->{an} 准确 {a1}%->{am}% | 散文 CE {p0}->{pn} 准确 {p1}%->{pm}%')
PY" 2>/dev/null)
  echo "$(date '+%H:%M:%S') $R"
  case "$R" in *"步 "*) :;; *) echo "  (还没有步数日志或探针失败)";; esac
  sleep 300
done
