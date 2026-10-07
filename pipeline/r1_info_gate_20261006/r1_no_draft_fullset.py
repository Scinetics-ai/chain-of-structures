# -*- coding: utf-8 -*-
"""闸1 的结论之后: 把【不需要草稿】的信息源在全量 parent 上量准。

★闸1 判决(2026-10-06, 真无条件草稿 n=2,925 配对):
    草稿几何 dCen  0.95x  <- **比随机还差, 零信号**
    模板检索        5.90x
    模板+草稿       4.57x  <- **加草稿把 5.90 拉到 4.57, 净负作用**
  之前那个 3.24x 是用【第2轮草稿】当代理算的, 而第2轮草稿是注入 R1 核心后重新生成的,
  几何已经向核心聚拢 ⇒ 那是泄漏。我当时引 identity_pins_do_not_localise_in_3d(p=0.665)
  论证"代理偏得不多", **那个推理是错的**。草稿这条路到此为止。

★所以这里只量不需要草稿的三样, 在全部 parent 上:
    S  p/L 全局先验            (一张 40 桶查表)
    E  p/L x EC(降级阶梯)       (EC4->EC3->EC2->EC1->全局, 每级用量要打出来)
    T  模板检索(同 EC3 最近长度) (检索允许: feedback_no_retrieval_generation 2026-09-23 撤销禁令,
                                判据是端到端、不许把本条的答案当输入)
    T+E 两者相加
★必须报两个区间(EC3 见过 / 没见过)并自己先报检索基线, 否则数字没意义。
★氨基酸通道单独报: R1 实测只有 4.9%, 连恒答基线都不如, 这是最该补的通道。
用法: r1_no_draft_fullset.py <r1_core_ec.json>
"""
import collections
import json
import sys

import numpy as np

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
NB = 40
BUDGET = 100

ce = json.load(open(sys.argv[1]))
pids = [p for p, v in ce.items() if v.get("core") and v.get("length") and v.get("ec")]
rxn = collections.defaultdict(list)
for p in pids:
    rxn[p.split("_")[0]].append(p)
keys = sorted(rxn)
rng = np.random.default_rng(0)
rng.shuffle(keys)
half = len(keys) // 2
fit = [p for k in keys[:half] for p in rxn[k]]
ev = [p for k in keys[half:] for p in rxn[k]]
print(f"  parent {len(pids):,} | 反应 {len(keys):,} ⇒ 拟合 {len(fit):,} / 评估 {len(ev):,} (反应不交)")


def lv(ec):
    f = ec.split(".")
    return [".".join(f[:k]) for k in (4, 3, 2, 1)]


pos_glob = np.ones(NB)
pos_ec = collections.defaultdict(lambda: np.ones(NB))
aa_glob = collections.Counter()
aa_ec = collections.defaultdict(collections.Counter)
aa_ecb = collections.defaultdict(collections.Counter)
n_ec = collections.defaultdict(list)
lib = collections.defaultdict(list)
for p in fit:
    v = ce[p]
    L = v["length"]
    ls = lv(v["ec"])
    fr = []
    for pos, aa in zip(v["core"], v["aa"]):
        if not (1 <= pos <= L) or aa not in AA20:
            continue
        b = min(NB - 1, int((pos - 1) / L * NB))
        pos_glob[b] += 1
        aa_glob[aa] += 1
        for e in ls:
            pos_ec[e][b] += 1
            aa_ec[e][aa] += 1
        aa_ecb[(ls[1], b)][aa] += 1
        fr.append((pos - 1) / L)
    if fr:
        lib[ls[1]].append((np.asarray(fr), v["aa"], L))
    for e in ls:
        n_ec[e].append(len(v["core"]))
CONST = aa_glob.most_common(1)[0][0]
seen3 = set(lib)
print(f"  模板库 EC3 类 {len(lib):,} / 模板 {sum(len(x) for x in lib.values()):,}"
      f" | 恒答基线残基 {CONST} (拟合侧 {aa_glob[CONST]/sum(aa_glob.values()):.1%})")

ARMS = ["R  随机", "S  p/L 全局先验", "E  p/L x EC(降级)",
        "T  模板检索(同EC3)", "T+E 模板+先验"]
AAARMS = ["恒答 " + CONST, "EC 条件化", "EC x 位置桶", "模板对齐残基"]
res = collections.defaultdict(lambda: [0, 0])
cov = collections.defaultdict(list)
reg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
tier = collections.Counter()
comp = collections.defaultdict(list)
cnt = collections.defaultdict(list)
glob_n = int(np.median(sum((n_ec[str(k)] for k in range(1, 8)), [])))

for p in ev:
    v = ce[p]
    L = v["length"]
    core = [x for x in v["core"] if 1 <= x <= L]
    if not core:
        continue
    gold, k = set(core), len(core)
    ls = lv(v["ec"])
    li = next((i for i, e in enumerate(ls) if sum(pos_ec[e]) > NB + 5), 4)
    tier[["EC4", "EC3", "EC2", "EC1", "全局"][li]] += 1
    rg = "EC3见过" if ls[1] in seen3 else "EC3没见过"

    bi = np.minimum(NB - 1, (np.arange(L) / L * NB).astype(int))
    pg = pos_glob[bi] / pos_glob.sum()
    pe_src = pos_ec[ls[li]] if li < 4 else pos_glob
    pe = pe_src[bi] / pe_src.sum()
    zpe = (pe - pe.mean()) / (pe.std() + 1e-9)
    tpl = min(lib[ls[1]], key=lambda t: abs(t[2] - L)) if ls[1] in lib else None
    ts = None
    if tpl is not None:
        tp = np.clip((tpl[0] * L).round().astype(int), 0, L - 1)
        ts = np.zeros(L)
        for j in tp:
            lo, hi = max(0, j - 3), min(L, j + 4)
            ts[lo:hi] = np.maximum(ts[lo:hi], 1.0 - np.abs(np.arange(lo, hi) - j) / 4.0)
    sc = {"R  随机": rng.random(L), "S  p/L 全局先验": pg, "E  p/L x EC(降级)": pe}
    if ts is not None:
        sc["T  模板检索(同EC3)"] = ts
        sc["T+E 模板+先验"] = (ts - ts.mean()) / (ts.std() + 1e-9) + zpe
    for nm, s in sc.items():
        o = np.argsort(-np.asarray(s, dtype=np.float64))
        h = len(gold & set(int(i) + 1 for i in o[:k]))
        res[nm][0] += h; res[nm][1] += k
        reg[rg][nm][0] += h; reg[rg][nm][1] += k
        cov[nm].append(len(gold & set(int(i) + 1 for i in o[:min(BUDGET, L)])) / k)

    ae = aa_ec[ls[li]] if li < 4 else aa_glob
    be = ae.most_common(1)[0][0] if ae else CONST
    ta = None if tpl is None else {int(j) + 1: a for j, a in zip(tp, tpl[1])}
    for pos, aa in zip(core, v["aa"]):
        if aa not in AA20:
            continue
        b = min(NB - 1, int((pos - 1) / L * NB))
        bb = aa_ecb.get((ls[1], b))
        preds = {"恒答 " + CONST: CONST, "EC 条件化": be,
                 "EC x 位置桶": bb.most_common(1)[0][0] if bb else be}
        if ta:
            preds["模板对齐残基"] = min(ta.items(), key=lambda q: abs(q[0] - pos))[1]
        for nm, pd in preds.items():
            res["AA|" + nm][0] += int(pd == aa); res["AA|" + nm][1] += 1
            reg[rg]["AA|" + nm][0] += int(pd == aa); reg[rg]["AA|" + nm][1] += 1

    gc = collections.Counter(a for a in v["aa"] if a in AA20)
    for nm, src in (("全局频率表", aa_glob), ("EC 条件化", ae if ae else aa_glob)):
        tot = sum(src.values()) or 1
        pc = collections.Counter()
        for a in sorted(src, key=lambda x: -src[x]):
            c = int(round(src[a] / tot * k))
            if c:
                pc[a] = c
        it = sum((gc & pc).values()); pn, gn = sum(pc.values()), sum(gc.values())
        comp[nm].append(0.0 if not (pn and gn) else 2 * it / (pn + gn))
    cnt["常数"].append(abs(k - glob_n))
    cnt["EC 条件化"].append(abs(k - (int(np.median(n_ec[ls[li]])) if li < 4 else glob_n)))

print(f"  降级阶梯 {dict(tier)}")
rb = res["R  随机"][0] / max(1, res["R  随机"][1])
print(f"\n  ===== 位置通道 (取前 k=|核心|; 参照: 天然配体距离 8.27x 但那用了答案) =====")
print(f"  {'臂':<22}{'命中':>8}{'富集':>8}{'前100行覆盖':>12}{'n':>10}")
for nm in ARMS:
    if nm in res and res[nm][1]:
        h, t = res[nm]
        print(f"  {nm:<22}{h/t:7.2%}{h/t/rb:7.2f}x{np.mean(cov[nm]):11.1%}{t:10,}")
print(f"\n  ===== 氨基酸通道 (给定正确位置, 预测放什么; R1 实测只有 4.9%) =====")
for nm in AAARMS:
    key = "AA|" + nm
    if key in res and res[key][1]:
        h, t = res[key]
        print(f"  {nm:<22}{h/t:7.2%}{h/t/(res['AA|恒答 '+CONST][0]/max(1,res['AA|恒答 '+CONST][1])):7.2f}x{t:21,}")
print(f"\n  ===== 组成通道 (与位置无关, 多重集 F1) =====")
for nm, v2 in comp.items():
    print(f"  {nm:<22}{np.mean(v2):7.3f}")
print(f"\n  ===== 个数通道 (|核心| 绝对误差) =====")
for nm, v2 in cnt.items():
    print(f"  {nm:<22}{np.mean(v2):7.2f}")
print(f"\n  ===== 两个区间必须分开报 =====")
for rg in ("EC3见过", "EC3没见过"):
    d = reg.get(rg)
    if not d:
        continue
    tot = d["R  随机"][1]
    print(f"  --- {rg}  (位置 n={tot:,}) ---")
    for nm in ARMS + ["AA|" + x for x in AAARMS]:
        if nm in d and d[nm][1]:
            h, t = d[nm]
            print(f"      {nm:<22}{h/t:7.2%}  (n={t:,})")
