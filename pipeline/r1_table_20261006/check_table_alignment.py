# -*- coding: utf-8 -*-
"""对齐体检: 同一套造表规则下, train / val_overlap / val_strict / test_overlap / test_strict
   的表质量是否可比。**val 必须和它要代理的那个 test 对齐, 否则训练决策会判错。**

★为什么要单独做这个:
  v11 的 val 是反应不交的, 而 test 里 97% 是反应重叠的 ⇒ val 比 test 难, 判错。
  第一版修法(按 parent>=4 的反应抽 val_overlap)又把 val 抽成**比 train 还容易**
  (覆盖 70.4% vs 61.5%) —— 抽样条件和表质量正相关([[feedback_pilot_sample_is_biased]])。
  所以必须把五组放一起、用同一套库和同一个判据量一次。
★库只用 train; 打分一律留一反应(leave-one-reaction-out), 五组一视同仁。
用法: check_table_alignment.py <core_ec.json> <splits目录> <test_overlap_pids> <test_strict_pids>
"""
import collections
import json
import math
import sys

import numpy as np

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
NB, WIN, BUDGET, TPL_TRUST = 40, 3, 100, 5
ce = json.load(open(sys.argv[1]))
spl_d, tov_f, tst_f = sys.argv[2], sys.argv[3], sys.argv[4]


def rd(fp):
    try:
        return {x.strip() for x in open(fp)} - {""}
    except OSError:
        return set()


GROUP = {}
for nm in ("train", "val_overlap", "val_strict"):
    for q in rd(f"{spl_d}/{nm}_pids.txt"):
        GROUP[q] = nm
for q in rd(tov_f):
    GROUP.setdefault(q, "test_overlap")
for q in rd(tst_f):
    GROUP.setdefault(q, "test_strict")
ok = {q: g for q, g in GROUP.items() if q in ce and ce[q].get("core") and ce[q].get("ec")}
print(f"  分组 {collections.Counter(ok.values())}")


def lvs_of(ec):
    f = ec.split(".")
    return [".".join(f[:k]) for k in (4, 3, 2, 1)]


pos_tot = collections.defaultdict(lambda: np.zeros(NB))
pos_rxn = collections.defaultdict(lambda: np.zeros(NB))
aab_tot = collections.defaultdict(collections.Counter)
aab_rxn = collections.defaultdict(collections.Counter)
lib = collections.defaultdict(list)
for q, g in ok.items():
    if g != "train":
        continue
    v = ce[q]; L = v["length"]; rk = q.split("_")[0]; ls = lvs_of(v["ec"])
    fr = []
    for pos, aa in zip(v["core"], v["aa"]):
        if not (1 <= pos <= L) or aa not in AA20:
            continue
        b = min(NB - 1, int((pos - 1) / L * NB))
        for e in ls:
            pos_tot[e][b] += 1; pos_rxn[(e, rk)][b] += 1
        aab_tot[(ls[1], b)][aa] += 1; aab_rxn[(ls[1], b, rk)][aa] += 1
        fr.append((pos - 1) / L)
    if fr:
        lib[ls[1]].append((np.asarray(fr), list(v["aa"]), L, rk))
print(f"  库(只用 train): EC3 类 {len(lib):,} / 模板 {sum(len(x) for x in lib.values()):,}")

st = collections.defaultdict(lambda: collections.defaultdict(float))
tdl = collections.defaultdict(list)
for q, g in ok.items():
    v = ce[q]; L = v["length"]; rk = q.split("_")[0]; ls = lvs_of(v["ec"])
    core = [p for p in v["core"] if 1 <= p <= L]
    if not core:
        continue
    hist = None
    for e in ls:
        h = pos_tot[e] - pos_rxn[(e, rk)]
        if h.sum() > 5:
            hist = h + 1.0; break
    if hist is None:
        hist = np.ones(NB)
    bi = np.minimum(NB - 1, (np.arange(L) / L * NB).astype(int))
    pe = hist[bi] / hist.sum()
    z = (pe - pe.mean()) / (pe.std() + 1e-9)
    cand = [t for t in lib.get(ls[1], []) if t[3] != rk]
    tpl_at = {}
    if cand:
        t = min(cand, key=lambda x: abs(x[2] - L))
        tp = np.clip((t[0] * L).round().astype(int), 0, L - 1)
        ts = np.zeros(L)
        for j, aa in zip(tp, t[1]):
            lo, hi = max(0, j - WIN), min(L, j + WIN + 1)
            idx = np.arange(lo, hi)
            ts[lo:hi] = np.maximum(ts[lo:hi], 1.0 - np.abs(idx - j) / (WIN + 1))
            tpl_at.setdefault(int(j) + 1, aa if aa in AA20 else "-")
        score = (ts - ts.mean()) / (ts.std() + 1e-9) + z
    else:
        score = z
    order = np.argsort(-score)[:min(BUDGET, L)]
    pool = set(int(i) + 1 for i in order)
    tgt = [p for p in core if p in pool]
    s = st[g]
    s["n"] += 1
    s["cov"] += len(tgt) / len(core)
    s["has_tpl"] += 1.0 if cand else 0.0
    s["ntgt"] += len(tgt)
    aa_at = {p: a for p, a in zip(v["core"], v["aa"])}
    anchors = sorted(tpl_at)
    for p in tgt:
        s["pn"] += 1
        if anchors:
            qn = min(anchors, key=lambda x: abs(x - p))
            td = min(9, int(math.log1p(abs(qn - p)) * 2.2))
            tdl[g].append(td)
            if td < TPL_TRUST:
                s["tshow"] += 1
                s["thit"] += int(tpl_at[qn] == aa_at[p])
        else:
            tdl[g].append(9)
        b = min(NB - 1, int((p - 1) / L * NB))
        bb = aab_tot.get((ls[1], b), collections.Counter()) - aab_rxn.get(
            (ls[1], b, rk), collections.Counter())
        if bb:
            s["eshow"] += 1
            s["ehit"] += int(bb.most_common(1)[0][0] == aa_at[p])

print(f"\n  {'分组':<14}{'parent':>8}{'表覆盖':>9}{'目标均值':>9}{'有模板':>8}"
      f"{'tplAA给出':>10}{'tplAA命中':>10}{'ecAA命中':>10}{'tplD中位':>9}")
for g in ("train", "val_overlap", "test_overlap", "val_strict", "test_strict"):
    s = st.get(g)
    if not s or not s["n"]:
        continue
    n = s["n"]
    print(f"  {g:<14}{int(n):>8,}{s['cov']/n:8.1%}{s['ntgt']/n:9.2f}{s['has_tpl']/n:7.1%}"
          f"{s['tshow']/max(1,s['pn']):9.1%}{s['thit']/max(1,s['tshow']):9.1%}"
          f"{s['ehit']/max(1,s['eshow']):9.1%}{int(np.median(tdl[g])):8d}")
print("\n  判读: val_overlap 必须贴近 test_overlap, val_strict 必须贴近 test_strict。")
print("        train 与 val_overlap 的差 = 模型会过度信任表的程度(越小越好)。")
