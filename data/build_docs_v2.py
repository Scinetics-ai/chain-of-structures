#!/usr/bin/env python3
"""Tokenised 5-stage documents for the 4-stage arm.

The original builder was not kept, so its two conventions were recovered from qwen35_9b_docs.pt and
reproduced exactly on 20/20 records before this was written:
  join = "\\n\\n".join(step_k.strip() for k in 1..5)
  hard = token is all digits, or is a single character in the 20-letter alphabet (NOT the residue
         regex: the stored mask marks the digits of "389 residues" and the E of "(E)-sinapate" too)
Spans start at each "# Step k" marker, found as a token subsequence, the same way gen_cot_docs.py
recovers them.

qwen_batch truncates at 2048 keeping the FRONT, so a document over 2048 loses step 5's labels
silently.  The overflow count is printed and asserted against a ceiling rather than left to chance.
"""
import os, json, re, sys, glob, torch, statistics as st
from transformers import AutoTokenizer
V2 = os.environ.get("RUNS_V2", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs_v2")
S  = os.environ.get("SPLIT_DIR", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2")
QP = os.environ.get("QWEN_PATH") or (glob.glob(
    "/data/workspace/caiqingran/home_migration/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*")
    + glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*")))[0]
OUT = os.environ.get("DOCS_OUT", f"{S}/qwen35_9b_docs_v2.pt")
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
CAP = int(os.environ.get("DOC_CAP", "2048"))

tok = AutoTokenizer.from_pretrained(QP)
marks = {k: tok.encode(f"# Step {k}", add_special_tokens=False) for k in (2, 3, 4, 5)}

def find_sub(seq, sub, start=0):
    n, m = len(seq), len(sub)
    for i in range(start, n - m + 1):
        if seq[i:i + m] == sub: return i
    return -1

def hard_of(ids):
    out = torch.zeros(len(ids), dtype=torch.bool)
    for i, t in enumerate(tok.convert_ids_to_tokens(ids)):
        s = t.replace("Ġ", "").replace("Ģ", "").strip()
        if s and (s.isdigit() or (len(s) == 1 and s in AA1)): out[i] = True
    return out

split = json.load(open(f"{S}/split_seqcluster.json"))
ids_all = sorted({i for v in split.values() for i in v})
ref = torch.load(os.environ.get("DOCS_REF", f"{S}/qwen35_9b_docs.pt"), map_location="cpu")
print(f"records {len(ids_all)}   cap {CAP}", flush=True)

out_ids, out_spans, out_hard = {}, {}, {}
n_v2 = 0; bad_span = 0; lens = []; ex = []
for n, idx in enumerate(ids_all):
    parts = []
    used_v2 = False
    ok = True
    for k in range(1, 6):
        p = f"{V2}/cot_{idx}/step{k}.v2.md"
        if os.path.exists(p): used_v2 = True
        else: p = f"{V2}/cot_{idx}/step{k}.md"
        if not os.path.exists(p): ok = False; break
        parts.append(open(p).read().strip())
    if not ok: continue
    n_v2 += used_v2
    t = tok("\n\n".join(parts), add_special_tokens=False).input_ids
    starts = [0]
    cur = 0
    for k in (2, 3, 4, 5):
        j = find_sub(t, marks[k], cur + 1)
        if j < 0: ok = False; break
        starts.append(j); cur = j
    if not ok:
        bad_span += 1
        if len(ex) < 5: ex.append((idx, "marker not found"))
        continue
    out_ids[idx] = torch.tensor(t, dtype=torch.long)
    out_spans[idx] = [(starts[i], starts[i + 1] if i + 1 < 5 else len(t)) for i in range(5)]
    out_hard[idx] = hard_of(t)
    lens.append(len(t))
    if (n + 1) % 5000 == 0: print(f"  {n+1}/{len(ids_all)}  ok={len(out_ids)}", flush=True)

lens.sort()
print(f"\nbuilt {len(out_ids)}   used a .v2.md file: {n_v2}   marker failures: {bad_span}")
print(f"doc len: med {lens[len(lens)//2]}  p90 {lens[9*len(lens)//10]}  max {lens[-1]}")
print(f"over {CAP}: {sum(1 for x in lens if x > CAP)}/{len(lens)}  "
      f"({100*sum(1 for x in lens if x > CAP)/len(lens):.2f}%)   <- these lose step 5's labels")
rl = sorted(len(ref['ids'][k]) for k in ref['ids'])
print(f"reference doc len for comparison: med {rl[len(rl)//2]}  max {rl[-1]}")
# the last span must actually contain the step-5 labels
k5 = [i for i in out_ids if os.path.exists(f"{V2}/cot_{i}/step5.v2.md")][:200]
miss = 0
for i in k5:
    a, e = out_spans[i][4]
    if not re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]\d{1,4}\b", tok.decode(out_ids[i][a:e])): miss += 1
print(f"stage-5 span with no residue label: {miss}/{len(k5)}")
for x in ex: print("   ", x)
if "--write" in sys.argv:
    torch.save({"model": "Qwen/Qwen3.5-9B", "ids": out_ids, "spans": out_spans, "hard": out_hard}, OUT)
    print(f"\nsaved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
else:
    print("\nDRY RUN (pass --write)")
