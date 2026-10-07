# -*- coding: utf-8 -*-
"""R1 该被多给什么信息? —— 把每个候选信息源单独量出"它在哪个通道值多少"。

★为什么要分通道量(project_blind_position_gate_failed 的实测教训):
  给 R1 加 DPLM 工具报告时, 位置 2.13x -> 1.75x(**变差**), 而氨基酸 0.0000 -> 0.0563(变好);
  报告打乱更差(1.23x)。⇒ 给 R1 加信息**不是一律有益**, 必须分「位置」和「氨基酸」两个通道
  各报一次, 并且和各自的基线比。只报一个合计数会把一个有害项说成有益。

★切分: 按【反应】对半(parent_id 前缀 = 反应 id)。按 parent 切会泄漏同源
  (project_split_leaks_homologs / project_only_99_independent_families)。
★降级阶梯必须记录用了哪一级(feedback_silent_success_is_the_failure):
  EC4 -> EC3 -> EC2 -> EC1 -> 全局, 每级命中多少条要打出来。
★两个区间分开报(feedback_no_retrieval_generation): EC3 见过 / EC3 没见过。

位置通道基线: 随机(~4.7%); 要打的参照是配体距离 8.27x(project_blind_position_gate_failed)。
氨基酸通道基线: 恒答最常见催化残基(project_stage2a_sits_at_the_constant_prior 实测 34.11%)。
用法: r1_info_sources.py <r1_core_ec.json> [geom.jsonl]
"""
import collections
import json
import sys

import numpy as np

import os
RD = int(os.environ.get("GEOM_ROUND", "2"))

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
NB = 40                      # p/L 分桶数, 与 r1_control_seqprior 一致
BUDGET = 100                 # 候选表行数预算(与 R2-R4 同构)

ce = json.load(open(sys.argv[1]))
geom = {}
if len(sys.argv) > 2:
    for line in open(sys.argv[2]):
        d = json.loads(line)
        if d["round"] == RD:   # ★闸1: 真无条件草稿是 round 1, 代理口径是 round 2
            geom[d["parent_id"]] = d

GEOMONLY = "--paired" in sys.argv
pids = [p for p, v in ce.items() if v.get("core") and v.get("length") and v.get("ec")]
if GEOMONLY:          # ★配对口径: 所有臂都只在有草稿几何的 parent 上算, 否则跨组比会夸大
    pids = [p for p in pids if p in geom and geom[p]["length"] == ce[p]["length"]]
    print(f"  ★配对口径: 只留有草稿几何的 parent ⇒ {len(pids):,}")
rxn = collections.defaultdict(list)
for p in pids:
    rxn[p.split("_")[0]].append(p)
keys = sorted(rxn)
rng = np.random.default_rng(0)
rng.shuffle(keys)
half = len(keys) // 2
fit_p = [p for k in keys[:half] for p in rxn[k]]
ev_p = [p for k in keys[half:] for p in rxn[k]]
print(f"  parent {len(pids):,} | 反应 {len(keys):,} ⇒ 拟合 {len(fit_p):,} / 评估 {len(ev_p):,} (反应不交)")


def ec_levels(ec):
    f = ec.split(".")
    return [".".join(f[:k]) for k in (4, 3, 2, 1)]


# ---------- 在拟合侧估计 ----------
pos_glob = np.ones(NB)                                   # 加 1 平滑
pos_ec = collections.defaultdict(lambda: np.ones(NB))
aa_glob = collections.Counter()
aa_ec = collections.defaultdict(collections.Counter)
aa_ec_bucket = collections.defaultdict(collections.Counter)
n_ec = collections.defaultdict(list)                     # EC -> 核心大小
for p in fit_p:
    v = ce[p]
    L = v["length"]
    lv = ec_levels(v["ec"])
    for pos, aa in zip(v["core"], v["aa"]):
        if not (1 <= pos <= L) or aa not in AA20:
            continue
        b = min(NB - 1, int((pos - 1) / L * NB))
        pos_glob[b] += 1
        aa_glob[aa] += 1
        for e in lv:
            pos_ec[e][b] += 1
            aa_ec[e][aa] += 1
        aa_ec_bucket[(lv[1], b)][aa] += 1
    for e in lv:
        n_ec[e].append(len(v["core"]))
AA_CONST = aa_glob.most_common(1)[0][0]
print(f"  恒答基线的那个残基 = {AA_CONST} (拟合侧占 {aa_glob[AA_CONST]/sum(aa_glob.values()):.1%})")

# ---------- 评估 ----------
res = collections.defaultdict(lambda: [0, 0])            # 名字 -> [命中, 总数]
cov = collections.defaultdict(list)                      # 名字 -> 前100行覆盖
tier_used = collections.Counter()
seen_ec3 = set(k for k in pos_ec if len(k.split(".")) == 3 and sum(pos_ec[k]) > NB + 5)
comp_f1 = collections.defaultdict(list)
cnt_err = collections.defaultdict(list)
by_regime = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))


def topk(score, L, k):
    return set(int(i) + 1 for i in np.argsort(-score)[:k])


for p in ev_p:
    v = ce[p]
    L, core = v["length"], [x for x in v["core"] if 1 <= x <= L]
    if not core:
        continue
    gold = set(core)
    k = len(core)
    lv = ec_levels(v["ec"])
    lvl = next((i for i, e in enumerate(lv) if sum(pos_ec[e]) > NB + 5), 4)
    tier_used[["EC4", "EC3", "EC2", "EC1", "全局"][lvl]] += 1
    reg = "EC3见过" if lv[1] in seen_ec3 else "EC3没见过"

    bidx = np.minimum(NB - 1, ((np.arange(L)) / L * NB).astype(int))
    pri_g = pos_glob[bidx] / pos_glob.sum()
    pe = pos_ec[lv[lvl]] if lvl < 4 else pos_glob
    pri_e = pe[bidx] / pe.sum()
    arms = {"R 随机": rng.random(L),
            "S  p/L 全局先验": pri_g,
            "E  p/L × EC(降级)": pri_e}
    g = geom.get(p)
    if g is not None and g["length"] == L:
        ca = np.asarray(g["ca"], dtype=np.float32)
        dcen = np.linalg.norm(ca - ca.mean(0), axis=-1)
        zg = -(dcen - dcen.mean()) / (dcen.std() + 1e-6)
        ze = (pri_e - pri_e.mean()) / (pri_e.std() + 1e-6)
        arms["G  草稿 dCen"] = zg
        arms["E+G 先验+草稿"] = ze + zg
    for name, sc in arms.items():
        sc = np.asarray(sc, dtype=np.float64)
        hit = len(gold & topk(sc, L, k))
        res[name][0] += hit
        res[name][1] += k
        by_regime[reg][name][0] += hit
        by_regime[reg][name][1] += k
        cov[name].append(len(gold & topk(sc, L, min(BUDGET, L))) / k)

    # 氨基酸通道: 给定 gold 位置, 预测那里的残基
    ae = aa_ec[lv[lvl]] if lvl < 4 else aa_glob
    best_e = ae.most_common(1)[0][0] if ae else AA_CONST
    for pos, aa in zip(core, v["aa"]):
        if aa not in AA20:
            continue
        b = min(NB - 1, int((pos - 1) / L * NB))
        bb = aa_ec_bucket.get((lv[1], b))
        for name, pred in (("恒答 " + AA_CONST, AA_CONST),
                           ("EC 条件化 top-1", best_e),
                           ("EC×位置桶 top-1", bb.most_common(1)[0][0] if bb else best_e)):
            res["AA|" + name][0] += int(pred == aa)
            res["AA|" + name][1] += 1
            by_regime[reg]["AA|" + name][0] += int(pred == aa)
            by_regime[reg]["AA|" + name][1] += 1

    # 组成通道(与位置无关): 预测催化残基的多重集
    gold_c = collections.Counter(a for a in v["aa"] if a in AA20)
    for name, src in (("全局频率表", aa_glob), ("EC 条件化", ae if ae else aa_glob)):
        tot = sum(src.values()) or 1
        pr = sorted(src, key=lambda a: -src[a])
        pred_c = collections.Counter()
        for a in pr:                                     # 按期望个数取整
            c = int(round(src[a] / tot * k))
            if c > 0:
                pred_c[a] = c
        inter = sum((gold_c & pred_c).values())
        pn, gn = sum(pred_c.values()), sum(gold_c.values())
        comp_f1[name].append(0.0 if not (pn and gn) else 2 * inter / (pn + gn))
    # 个数
    cnt_err["常数(拟合侧中位)"].append(abs(k - int(np.median(n_ec["1"] + n_ec["2"] + n_ec["3"] +
                                                        n_ec["4"] + n_ec["5"] + n_ec["6"] + n_ec["7"]))))
    cnt_err["EC 条件化中位"].append(abs(k - int(np.median(n_ec[lv[lvl]])) if lvl < 4 else k))

print(f"  降级阶梯用量 {dict(tier_used)}   EC3 见过的类 {len(seen_ec3):,}")
rnd = res["R 随机"][0] / max(1, res["R 随机"][1])
print(f"\n  ===== 位置通道 (取前 k=|核心| 个; 要打的参照是配体距离 8.27x) =====")
print(f"  {'臂':<22}{'命中':>8}{'富集':>8}{'前100行覆盖':>12}")
for nm in ("R 随机", "S  p/L 全局先验", "E  p/L × EC(降级)", "G  草稿 dCen", "E+G 先验+草稿"):
    if nm not in res:
        continue
    h, t = res[nm]
    c = np.mean(cov[nm]) if cov[nm] else float("nan")
    print(f"  {nm:<22}{h/t:7.2%}{h/t/rnd:7.2f}x{c:11.1%}")
print(f"\n  ===== 氨基酸通道 (给定正确位置, 预测该放什么) =====")
for nm in ("AA|恒答 " + AA_CONST, "AA|EC 条件化 top-1", "AA|EC×位置桶 top-1"):
    h, t = res[nm]
    print(f"  {nm[3:]:<22}{h/t:7.2%}   (n={t:,})")
print(f"\n  ===== 组成通道 (与位置无关: 预测催化残基的多重集, F1) =====")
for nm, v in comp_f1.items():
    print(f"  {nm:<22}{np.mean(v):7.3f}")
print(f"\n  ===== 个数通道 (|核心| 的绝对误差) =====")
for nm, v in cnt_err.items():
    print(f"  {nm:<22}{np.mean(v):7.2f}")
print(f"\n  ===== 两个区间分开报 =====")
for reg, d in sorted(by_regime.items()):
    print(f"  --- {reg} ---")
    for nm in ("E  p/L × EC(降级)", "G  草稿 dCen", "AA|EC 条件化 top-1", "AA|恒答 " + AA_CONST):
        if nm in d and d[nm][1]:
            h, t = d[nm]
            print(f"      {nm:<22}{h/t:7.2%}  (n={t:,})")
