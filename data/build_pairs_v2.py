#!/usr/bin/env python3
"""site_pairs for the 4-stage arm.  Same extractor as build_pairs.py, reading step{k}.v2.md when it
exists and step{k}.md otherwise, writing to a NEW file so the 2-stage arm keeps its own table.

Two guards that build_pairs.py does not have, because until now stages 4 and 5 were always empty and
an empty result could not be wrong:
  * stage 4/5 parsed sets must equal shells_v1.json exactly  -- a stray label is a poisoned position
  * the four stages must be pairwise disjoint in position    -- an overlap means a layer leaked
"""
import json, os, re, sys, glob, statistics as st
import torch
V2  = os.environ.get("RUNS_V2", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs_v2")
S   = os.environ.get("SPLIT_DIR", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2")
OUT = os.environ.get("PAIRS_OUT", f"{S}/site_pairs_v2.pt")
AA1 = "ACDEFGHIKLMNPQRSTVWY"
RES = re.compile(rf"\b([{AA1}])(\d{{1,4}})\b")
SH  = json.load(open(os.environ.get("SHELLS", f"{S}/shells_v1.json")))

def obs_of(t):
    i = t.find("## OBSERVATION")
    return t[i:] if i >= 0 else t

def pairs_of(idx, k):
    p = f"{V2}/cot_{idx}/step{k}.v2.md"
    if not os.path.exists(p): p = f"{V2}/cot_{idx}/step{k}.md"
    if not os.path.exists(p): return []
    seen, out = set(), []
    for m in RES.finditer(obs_of(open(p).read())):
        aa, pos = m.group(1), int(m.group(2)) - 1     # text is 1-based, tensors are 0-based
        if pos < 0 or (aa, pos) in seen: continue
        seen.add((aa, pos)); out.append((aa, pos))
    return out

split = json.load(open(f"{S}/split_seqcluster.json"))
ids = sorted({i for v in split.values() for i in v})
print(f"records {len(ids)}   shells available {len(SH)}", flush=True)

table, counts = {}, {k: [] for k in (2, 3, 4, 5)}
bad_eq = bad_ov = 0; ex = []
for n, idx in enumerate(ids):
    per = {}
    for k in (2, 3, 4, 5):
        p = pairs_of(idx, k)
        if p: per[k] = p
        counts[k].append(len(p))
    if per: table[idx] = per
    # guard 1: stage 4/5 must be exactly the layer we rendered
    if idx in SH:
        for k, f in ((4, "shell2"), (5, "anchor")):
            want = sorted((a, int(q) - 1) for a, q in SH[idx][f] if a in AA1)
            got = sorted(per.get(k, []))
            if got != want:
                bad_eq += 1
                if len(ex) < 6: ex.append((idx, k, "want %d got %d" % (len(want), len(got)),
                                           sorted(set(got) - set(want))[:4]))
    # guard 2: positions must not repeat across stages
    pos = [q for k in per for _, q in per[k]]
    if len(pos) != len(set(pos)):
        bad_ov += 1
        if len(ex) < 8: ex.append((idx, -1, "position reused across stages", None))
    if (n + 1) % 10000 == 0: print(f"  {n+1}/{len(ids)}", flush=True)

print(f"\nrecords with at least one pair: {len(table)}")
for k in (2, 3, 4, 5):
    c = counts[k]
    print(f"  stage {k}: median {int(st.median(c)):3d}  mean {st.mean(c):5.1f}  "
          f"empty {sum(1 for x in c if x == 0):6d}  ({100*sum(1 for x in c if x>0)/len(c):.1f}% non-empty)")
print(f"\nguard: stage4/5 parse != shells_v1      {bad_eq}")
print(f"guard: a position used at two stages     {bad_ov}")
for x in ex: print("   ", x)
assert bad_eq == 0, "the rendered labels and the parsed labels disagree -- do not train on this"
if "--write" in sys.argv:
    torch.save({"pairs": table, "aa_alphabet": AA1}, OUT)
    print(f"\nsaved {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
else:
    print("\nDRY RUN (pass --write)")
