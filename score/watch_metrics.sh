#!/bin/bash
# 盯 lthpc 的 ESMFold 指标。每 10 分钟采一次【累计】量(不看单片), 跑完自动算总表。
set -u
for i in $(seq 1 200); do
  R=$(timeout 90 ssh -o ConnectTimeout=25 lthpc 'cd /data/workspace/caiqingran/metrics_deep4b && python3 - <<PY
import csv, glob, statistics as st, subprocess
rows=[]
for f in sorted(glob.glob("out/esmfold_sc_f*.csv")):
    rows += [r for r in csv.DictReader(open(f)) if str(r.get("ok","")).lower()=="true"]
alive=subprocess.run(["bash","-c","pgrep -fc eval_direct_esmfold_consistency_loca[l] || echo 0"],
                     capture_output=True,text=True).stdout.strip()
if not rows:
    print("NOROWS"); raise SystemExit
g=lambda k:[float(r[k]) for r in rows if r.get(k) not in ("",None)]
rm,tm,pl=g("sc_rmsd"),g("sc_tm"),g("plddt")
f=lambda a,t: sum(1 for x in a if x<t)/len(a)
print(f"n={len(rows)}/2801 进程{alive} | pLDDT {st.mean(pl)*100:.1f} | "
      f"scTM 中位 {st.median(tm):.3f} 均值 {st.mean(tm):.3f} >0.5 {1-f(tm,0.5):.1%} >0.8 {1-f(tm,0.8):.1%} | "
      f"scRMSD 中位 {st.median(rm):.2f} 均值 {st.mean(rm):.2f} <2A {f(rm,2):.1%} <5A {f(rm,5):.1%}")
PY' 2>/dev/null)
  echo "$(date '+%H:%M:%S') $R"
  case "$R" in *"进程0"*) echo "  ★ESMFold 已全部结束"; break;; esac
  sleep 600
done
