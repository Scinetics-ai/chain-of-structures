# -*- coding: utf-8 -*-
"""定两个设计参数, 用数不用拍。

★参数1: 模板库要不要排除【同一条反应】的兄弟 parent?
  推理时两种情形都存在:
    test_strict(反应不交, 157 parent) —— 没有同反应兄弟 ⇒ 只有"排除"口径和推理一致
    test_overlap(反应重叠, 5,269)     —— 有兄弟 ⇒ "不排除"口径才和推理一致
  若训练时用"不排除", 模型会学到【照抄兄弟模板】这个捷径, 而该捷径在 strict 上不存在
  ⇒ strict 上会崩([[feedback_unpredictable_target_makes_the_baseline_unbeatable]] 的反面:
     这次是目标太可预测, 把捷径学成了本事)。这里把两种口径都量出来再决定。

★参数2: 让 R1 承诺几个位置?
  个数不可预测(EC 条件化 MAE 5.60 vs 常数 5.63) ⇒ 必须由【预算】定死, 不让它猜。
  而错位置是有害的(只给位置不给内容 EC3 0.0337 < 什么都不给 0.0667) ⇒ 精度比召回值钱。
  这里给整条 预算 k -> 精度/召回 曲线, 外加"按 L 比例定预算"的版本。
用法: r1_design_params.py <r1_core_ec.json>
"""
import collections
import json
import sys

import numpy as np

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
NB = 40
ce = json.load(open(sys.argv[1]))
pids = [p for p, v in ce.items() if v.get("core") and v.get("length") and v.get("ec")]
rxn = collections.defaultdict(list)
for p in pids:
    rxn[p.split("_")[0]].append(p)
keys = sorted(rxn)
rng = np.random.default_rng(0)
rng.shuffle(keys)
half = len(keys) // 2
fit_keys = set(keys[:half])
fit = [p for k in keys[:half] for p in rxn[k]]
ev = [p for k in keys[half:] for p in rxn[k]]
print(f"  parent {len(pids):,} | 反应 {len(keys):,} ⇒ 拟合 {len(fit):,} / 评估 {len(ev):,}")

pos_ec = collections.defaultdict(lambda: np.ones(NB))
lib = collections.defaultdict(list)      # EC3 -> [(fractions, aas, L, rxn_key)]
for p in fit:
    v = ce[p]
    L = v["length"]
    f = [".".join(v["ec"].split(".")[:k]) for k in (4, 3, 2, 1)]
    fr = []
    for pos, aa in zip(v["core"], v["aa"]):
        if 1 <= pos <= L and aa in AA20:
            b = min(NB - 1, int((pos - 1) / L * NB))
            for e in f:
                pos_ec[e][b] += 1
            fr.append((pos - 1) / L)
    if fr:
        lib[f[1]].append((np.asarray(fr), v["aa"], L, p.split("_")[0]))

# ★评估侧的"同反应兄弟"要从【评估侧自己】找 —— 但评估反应不在拟合库里,
#   所以这里改成: 在【拟合侧内部】做一次留一法, 同反应兄弟是否可用, 两种口径各量一次。
def score(L, e3, e_lvl, excl_rxn):
    cand = [t for t in lib.get(e3, []) if (excl_rxn is None or t[3] != excl_rxn)]
    ts = None
    if cand:
        t = min(cand, key=lambda x: abs(x[2] - L))
        tp = np.clip((t[0] * L).round().astype(int), 0, L - 1)
        ts = np.zeros(L)
        for j in tp:
            lo, hi = max(0, j - 3), min(L, j + 4)
            ts[lo:hi] = np.maximum(ts[lo:hi], 1.0 - np.abs(np.arange(lo, hi) - j) / 4.0)
    bi = np.minimum(NB - 1, (np.arange(L) / L * NB).astype(int))
    src = pos_ec[e_lvl]
    pe = src[bi] / src.sum()
    z = (pe - pe.mean()) / (pe.std() + 1e-9)
    if ts is None:
        return z
    return (ts - ts.mean()) / (ts.std() + 1e-9) + z


print("\n  ===== 参数1: 模板库含不含同反应兄弟 (在拟合侧做留一, 这样两种口径可比) =====")
print(f"  {'口径':<24}{'命中@k':>9}{'富集':>8}{'前100覆盖':>11}{'有模板比例':>11}")
for mode in ("含同反应兄弟", "排除同反应兄弟"):
    hit = tot = 0
    cov, haz = [], []
    rndhit = 0
    for p in fit[:6000]:
        v = ce[p]
        L = v["length"]
        core = [x for x in v["core"] if 1 <= x <= L]
        if not core:
            continue
        g, k = set(core), len(core)
        f = [".".join(v["ec"].split(".")[:j]) for j in (4, 3, 2, 1)]
        lvl = next((e for e in f if sum(pos_ec[e]) > NB + 5), f[-1])
        rk = p.split("_")[0]
        # 留一: 自己一定排除; 兄弟按 mode
        cand = [t for t in lib.get(f[1], []) if t[3] != rk] if mode == "排除同反应兄弟" else \
               [t for t in lib.get(f[1], [])]
        haz.append(1.0 if cand else 0.0)
        save = lib.get(f[1], [])
        lib[f[1]] = cand
        s = score(L, f[1], lvl, None)
        lib[f[1]] = save
        o = np.argsort(-s)
        hit += len(g & set(int(i) + 1 for i in o[:k])); tot += k
        cov.append(len(g & set(int(i) + 1 for i in o[:min(100, L)])) / k)
        rndhit += k * k / L
    print(f"  {mode:<24}{hit/tot:8.2%}{hit/max(1e-9,rndhit):7.2f}x{np.mean(cov):10.1%}{np.mean(haz):10.1%}")

print("\n  ===== 参数2: 承诺预算 k -> 精度/召回 (评估侧, 排除同反应口径) =====")
print(f"  {'预算':<14}{'位置精度':>9}{'位置召回':>9}{'富集':>8}{'期望对的个数':>13}{'期望错的个数':>13}")
med_core = int(np.median([len(ce[p]["core"]) for p in ev]))
for kk in (3, 4, 6, 8, 10, 14, 20, "L/50", "L/25"):
    hit = tot_pred = tot_gold = 0
    rnd = 0.0
    for p in ev:
        v = ce[p]
        L = v["length"]
        core = [x for x in v["core"] if 1 <= x <= L]
        if not core:
            continue
        g = set(core)
        f = [".".join(v["ec"].split(".")[:j]) for j in (4, 3, 2, 1)]
        lvl = next((e for e in f if sum(pos_ec[e]) > NB + 5), f[-1])
        n = max(1, int(L / 50)) if kk == "L/50" else (max(1, int(L / 25)) if kk == "L/25" else kk)
        s = score(L, f[1], lvl, p.split("_")[0])
        o = np.argsort(-s)[:min(n, L)]
        h = len(g & set(int(i) + 1 for i in o))
        hit += h; tot_pred += len(o); tot_gold += len(g)
        rnd += len(o) * len(g) / L
    pr, rc = hit / tot_pred, hit / tot_gold
    npred = tot_pred / len(ev)
    print(f"  {str(kk):<14}{pr:8.2%}{rc:9.2%}{hit/max(1e-9,rnd):7.2f}x"
          f"{pr*npred:12.2f}{(1-pr)*npred:13.2f}")
print(f"\n  评估侧 |核心| 中位 = {med_core}")
print("  读法: '期望错的个数' 是直接的代价(错位置+内容比什么都不给更差 0.0337<0.0667);")
print("        '期望对的个数' 是收益(真核心把 EC3 0.067->0.805)。")
