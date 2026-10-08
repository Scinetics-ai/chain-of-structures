#!/usr/bin/env python3
"""Fit SITE_NOISE_PROFILE / SITE_NOISE_BETA for ALL FOUR stages against the writer we now have.

The harness only ever carried stages 2 and 3, because stages 4 and 5 named no residues.  Now they do,
and a stage with pairs but no profile is returned UNCHANGED by corrupt_site_pairs -- injected as an
ORACLE, with no error.  The trainer now crashes on that rather than reporting a writer accuracy it
never measured; this is the script that removes the reason to crash.

Run it on stage A's generated pair table (gen_cot_docs writes one), then paste the printed dict into
SITE_NOISE_PROFILE / SITE_NOISE_BETA before the joint stage.
"""
import json, os, argparse, torch
ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference pairs, e.g. site_pairs_v2.pt")
ap.add_argument("--gen", required=True, help="pairs parsed from what the writer generated")
ap.add_argument("--rows", required=True, help="test rows json, for the id list")
a = ap.parse_args()

ref = torch.load(a.ref, map_location="cpu")["pairs"]
gen = torch.load(a.gen, map_location="cpu")["pairs"]
TEST = [r["idx"] for r in json.load(open(a.rows))]
print(f"ref {len(ref)}  gen {len(gen)}  test {len(TEST)}")
hit = len([i for i in TEST if i in gen])
print(f"test ids present in the generated table: {hit}/{len(TEST)} = {hit/len(TEST):.1%}")
assert hit > 0.5 * len(TEST), "the generated table barely covers the test split -- wrong file?"

print("\nSITE_NOISE_PROFILE = {")
beta = {}
for stage in (2, 3, 4, 5):
    ex = pos = w5 = tot = ng = nr = 0
    per = []
    for i in TEST:
        t = (ref.get(i) or {}).get(stage) or []
        g = (gen.get(i) or {}).get(stage) or []
        if not t:
            continue
        gd = {p: aa for aa, p in g}
        gp = sorted(gd)
        nr += len(t); ng += len(g)
        h = 0
        for aa, p in t:
            tot += 1
            ex += (gd.get(p) == aa)
            h += (p in gd)
            w5 += any(abs(p - q) <= 5 for q in gp)
        pos += h
        per.append(h / len(t))
    if not per:
        print(f"    # stage {stage}: NO reference pairs on the test split -- nothing to fit")
        continue
    n = len(per); m = sum(per) / n
    v = sum((x - m) ** 2 for x in per) / max(n - 1, 1)
    if 0 < v < m * (1 - m):
        k = m * (1 - m) / v - 1
        aa_, bb_ = m * k, (1 - m) * k
    else:
        aa_ = bb_ = float("nan")
    beta[stage] = (aa_, bb_)
    print(f'    {stage}: {{"exact": {ex/tot:.3f}, "pos": {pos/tot:.3f}, '
          f'"within5": {w5/tot:.3f}, "ratio": {ng} / {nr}}},   # {ng/nr:.3f}, n={n} records')
print("}")
print("\nSITE_NOISE_BETA = {" + ", ".join(f"{k}: ({v[0]:.3f}, {v[1]:.3f})" for k, v in beta.items()) + "}")
print("\nper-record position rate, and how often the writer misses a record entirely:")
for stage in (2, 3, 4, 5):
    per = []
    for i in TEST:
        t = (ref.get(i) or {}).get(stage) or []
        if not t: continue
        gd = {p for _, p in ((gen.get(i) or {}).get(stage) or [])}
        per.append(sum(1 for _, p in t if p in gd) / len(t))
    if per:
        z = sum(1 for x in per if x == 0)
        print(f"  stage {stage}: mean {sum(per)/len(per):.3f}   empty records {z}/{len(per)} = {z/len(per):.1%}")
