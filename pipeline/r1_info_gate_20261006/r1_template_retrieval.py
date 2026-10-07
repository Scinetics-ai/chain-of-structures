# -*- coding: utf-8 -*-
"""再加一个臂: 模板检索 —— 同 EC3 最近邻, 把【整套相对位置】一起迁过来。

★为什么单独测这个: 已实测「内容 >> 间距 > 绝对位置(≈0)」
  (project_r1_format_settled_tokens_plus_spacing / project_injection_interface_settled)。
  p/L 边缘先验扔掉了间距; 模板检索保留【联合模式】。所以它有可能比边缘先验强,
  必须分开量, 不能用边缘先验的数代表"检索这条路"。
★检索是允许的(feedback_no_retrieval_generation, 2026-09-23 用户撤销了禁令),
  判据是不许把【本条的答案】当输入 —— 这里只用拟合侧(反应不交)的其它酶, 合规。
★必须报两个区间: EC3 见过 / 没见过。没见过时检索会退化到无用甚至有害。
用法: r1_template_retrieval.py <r1_core_ec.json> <geom.jsonl>
"""
import collections
import json
import sys

import numpy as np

import os
RD = int(os.environ.get("GEOM_ROUND", "2"))

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
BUDGET = 100
ce = json.load(open(sys.argv[1]))
geom = {}
for line in open(sys.argv[2]):
    d = json.loads(line)
    if d["round"] == RD:   # ★闸1: 真无条件草稿是 round 1
        geom[d["parent_id"]] = d
pids = [p for p, v in ce.items()
        if v.get("core") and v.get("length") and v.get("ec")
        and p in geom and geom[p]["length"] == v["length"]]
rxn = collections.defaultdict(list)
for p in pids:
    rxn[p.split("_")[0]].append(p)
keys = sorted(rxn)
rng = np.random.default_rng(0)
rng.shuffle(keys)
half = len(keys) // 2
fit = [p for k in keys[:half] for p in rxn[k]]
ev = [p for k in keys[half:] for p in rxn[k]]
print(f"  配对口径 parent {len(pids):,} | 反应 {len(keys):,} ⇒ 拟合 {len(fit):,} / 评估 {len(ev):,}")

# 模板库: EC3 -> [(fractions, aas, L)]
lib = collections.defaultdict(list)
for p in fit:
    v = ce[p]
    L = v["length"]
    f = [(x - 1) / L for x in v["core"] if 1 <= x <= L]
    if f:
        lib[".".join(v["ec"].split(".")[:3])].append((np.asarray(f), v["aa"], L))
seen = set(lib)
print(f"  模板库 EC3 类 {len(lib):,}  模板条数 {sum(len(x) for x in lib.values()):,}")

res = collections.defaultdict(lambda: [0, 0])
cov = collections.defaultdict(list)
byreg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
sp_err = collections.defaultdict(list)

for p in ev:
    v = ce[p]
    L = v["length"]
    core = [x for x in v["core"] if 1 <= x <= L]
    if not core:
        continue
    gold, k = set(core), len(core)
    e3 = ".".join(v["ec"].split(".")[:3])
    reg = "EC3见过" if e3 in seen else "EC3没见过"
    g = geom[p]
    ca = np.asarray(g["ca"], dtype=np.float32)
    dcen = np.linalg.norm(ca - ca.mean(0), axis=-1)
    zg = -(dcen - dcen.mean()) / (dcen.std() + 1e-6)

    # 模板: 同 EC3 里长度最接近的一条
    tpl = None
    if e3 in lib:
        tpl = min(lib[e3], key=lambda t: abs(t[2] - L))
    arms = {}
    if tpl is not None:
        tp = np.clip((tpl[0] * L).round().astype(int), 0, L - 1)
        s = np.zeros(L)
        for j in tp:                                  # ±3 的软窗, 容忍一点配准误差
            lo, hi = max(0, j - 3), min(L, j + 4)
            s[lo:hi] = np.maximum(s[lo:hi], 1.0 - np.abs(np.arange(lo, hi) - j) / 4.0)
        arms["T  模板检索(同EC3最近长度)"] = s
        arms["T+G 模板+草稿几何"] = (s - s.mean()) / (s.std() + 1e-6) + zg
        # 间距: 模板的相邻间隔 vs 真实的相邻间隔
        gs = np.diff(sorted(core))
        ts = np.diff(sorted(tp + 1))
        m = min(len(gs), len(ts))
        if m:
            sp_err["模板间距 vs 真实间距(中位绝对误差)"].append(
                float(np.median(np.abs(gs[:m] - ts[:m]))))
    arms["G  草稿几何(参照)"] = zg
    arms["R  随机(参照)"] = rng.random(L)
    for nm, sc in arms.items():
        sc = np.asarray(sc, dtype=np.float64)
        order = np.argsort(-sc)
        hit = len(gold & set(int(i) + 1 for i in order[:k]))
        res[nm][0] += hit; res[nm][1] += k
        byreg[reg][nm][0] += hit; byreg[reg][nm][1] += k
        cov[nm].append(len(gold & set(int(i) + 1 for i in order[:min(BUDGET, L)])) / k)
    # 氨基酸: 模板在对齐位置上的残基
    if tpl is not None:
        ta = {int(j) + 1: a for j, a in zip(tp, tpl[1])}
        for pos, aa in zip(core, v["aa"]):
            if aa not in AA20:
                continue
            best, bd = None, 99
            for q, a in ta.items():
                if abs(q - pos) < bd:
                    best, bd = a, abs(q - pos)
            res["AA|模板对齐残基"][0] += int(best == aa); res["AA|模板对齐残基"][1] += 1
            byreg[reg]["AA|模板对齐残基"][0] += int(best == aa)
            byreg[reg]["AA|模板对齐残基"][1] += 1

rnd = res["R  随机(参照)"][0] / max(1, res["R  随机(参照)"][1])
print(f"\n  ===== 位置通道 =====\n  {'臂':<26}{'命中':>8}{'富集':>8}{'前100行覆盖':>12}")
for nm in ("R  随机(参照)", "T  模板检索(同EC3最近长度)", "G  草稿几何(参照)", "T+G 模板+草稿几何"):
    if nm in res and res[nm][1]:
        h, t = res[nm]
        print(f"  {nm:<26}{h/t:7.2%}{h/t/rnd:7.2f}x{np.mean(cov[nm]):11.1%}")
if res["AA|模板对齐残基"][1]:
    h, t = res["AA|模板对齐残基"]
    print(f"\n  ===== 氨基酸通道 =====\n  模板对齐残基 top-1    {h/t:7.2%}  (n={t:,})")
for nm, v in sp_err.items():
    print(f"\n  {nm}: {np.median(v):.1f} 残基")
print(f"\n  ===== 两个区间 =====")
for reg, d in sorted(byreg.items()):
    print(f"  --- {reg} ---")
    for nm in ("T  模板检索(同EC3最近长度)", "T+G 模板+草稿几何", "G  草稿几何(参照)", "AA|模板对齐残基"):
        if nm in d and d[nm][1]:
            h, t = d[nm]
            print(f"      {nm:<26}{h/t:7.2%}  (n={t:,})")
