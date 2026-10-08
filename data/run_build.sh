#!/bin/bash
# The whole 4-stage data build, on cvm-2, start to finish.  Waits for the two transfers, then runs
# the four steps in order and stops at the first one whose own assertion fails.
set -u
D=/data/caiqingran/enzymecot
export RUNS_V2=$D/runs_v2
export SPLIT_DIR=$D/cot_split_v2
export CIF_ROOT=$D/cifraw
export DS_DIR=""                      # every record, not only the deepseek intersection
export SHELLS_OUT=$SPLIT_DIR/shells_all.json
PY310=/data/caiqingran/env/bl310/bin/python     # has gemmi + torch
PY5=/data/caiqingran/env/py/bin/python          # transformers 5.x
say () { echo; echo "############ $* : $(date +%H:%M:%S)"; }

say "0 waiting for the cif files only -- the shell step does not need Qwen"
for i in $(seq 1 180); do
  NC=$(find $CIF_ROOT -name '*_model_0.cif' 2>/dev/null | wc -l)
  echo "  t=${i}0s  cif $NC/56022"
  [ "$NC" -ge 56022 ] && break
  sleep 10
done
NC=$(find $CIF_ROOT -name '*_model_0.cif' | wc -l)
[ "$NC" -ge 56022 ] || { echo "ABORT: only $NC cif files"; exit 1; }

say "1 shells for all 56022 records"
$PY310 -u $D/scripts/shells_v1.py || exit 1

say "2 cross-check against the lthpc run (gemmi 0.6.7 there, 0.7.5 here)"
$PY310 - <<'PY' || exit 1
import json, os
a = json.load(open("/data/caiqingran/enzymecot/shells_lthpc_22693.json"))
b = json.load(open(os.environ["SHELLS_OUT"]))
common = sorted(set(a) & set(b))
print(f"  compared {len(common)} records")
d2 = sum(1 for i in common if a[i]["shell2"] != b[i]["shell2"])
d5 = sum(1 for i in common if a[i]["anchor"] != b[i]["anchor"])
print(f"  shell2 differs: {d2}   anchor differs: {d5}")
if d2 or d5:
    ex = [i for i in common if a[i]["shell2"] != b[i]["shell2"] or a[i]["anchor"] != b[i]["anchor"]][:2]
    for i in ex:
        print(f"    {i} lthpc shell2 {a[i]['shell2'][:4]}")
        print(f"    {i} cvm2  shell2 {b[i]['shell2'][:4]}")
    print("  the two gemmi versions disagree -- the cvm-2 result is the one we use, but say so")
else:
    print("  identical: the gemmi version does not change the layers")
PY

say "3 render the step4/step5 sentence"
export SHELLS=$SHELLS_OUT
$PY310 -u $D/scripts/patch_text_v2.py --write || exit 1

say "4 extract the pairs"
export PAIRS_OUT=$SPLIT_DIR/site_pairs_v2.pt
$PY310 -u $D/scripts/build_pairs_v2.py --write || exit 1

say "5 tokenise the documents"
for i in $(seq 1 180); do
  QS=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ 2>/dev/null | head -1)
  QB=0; [ -n "$QS" ] && QB=$(du -sb "$D/hf/models--Qwen--Qwen3.5-9B" 2>/dev/null | cut -f1)
  echo "  waiting for qwen: $((QB/1024/1024/1024)) GB"
  [ "$QB" -gt 18000000000 ] && break
  sleep 20
done
export QWEN_PATH=$(ls -d $D/hf/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
[ -n "$QWEN_PATH" ] || { echo "ABORT: no qwen snapshot"; exit 1; }
echo "  QWEN_PATH=$QWEN_PATH"
export DOCS_OUT=$SPLIT_DIR/qwen35_9b_docs_v2.pt
PYT=$PY310
$PY310 -c "
import os
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(os.environ['QWEN_PATH'])" >/dev/null 2>&1 || PYT=$PY5
echo "  tokenizer env: $PYT"
$PYT -u $D/scripts/build_docs_v2.py --write || exit 1

say "DONE"
ls -la $SPLIT_DIR/shells_all.json $SPLIT_DIR/site_pairs_v2.pt $SPLIT_DIR/qwen35_9b_docs_v2.pt
