# -*- coding: utf-8 -*-
"""mac-F1 为什么只有 0.148? 按每个 EC 类的支持度拆开看。

★不解释, 量。mac-F1 是各类 F1 的等权平均, 所以只要有大量只含 1-2 条样本的类、
  而模型在它们上面失败, mac 就会被拖到很低, 而 mic 几乎不受影响。
  这里把类按支持度分桶, 报每桶的类数、占总记录比、平均 F1、F1=0 的类数,
  并给出"只算支持度>=N 的类"时 mac-F1 怎么变。
"""
import glob
import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

lab = json.load(open("gt_ec4.json"))
keep = {x.strip() for x in open("subset_2435.txt") if x.strip()}
files = sorted(glob.glob("/data/workspace/caiqingran/protein_design/DISCO_repo/runs/"
                         "EVAL_KEEP_deep4b/clean/cotfull_f*_maxsep.csv"))
yt, yp = [], []
for p in files:
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        head, *parts = ln.split(",")
        _, _, idx = head.partition("|")
        idx = (idx or head).strip()
        if idx not in keep:
            continue
        w = lab.get(idx)
        if not w:
            continue
        g = set()
        for x in parts:
            x = x.strip()
            if not x or "/" not in x:
                continue
            a, _, b = x.rpartition("/")
            try:
                if float(b) <= 0.01:
                    continue
            except ValueError:
                continue
            a = a.replace("EC:", "").strip()
            if a:
                g.add(a)
        yt.append(set(w)); yp.append(g)

mlb = MultiLabelBinarizer()
Yt = mlb.fit_transform(yt)
cls = set(mlb.classes_)
Yp = mlb.transform([{x for x in s if x in cls} for s in yp])
sup = Yt.sum(0)
f1 = f1_score(Yt, Yp, average=None, zero_division=0)
pr = precision_score(Yt, Yp, average=None, zero_division=0)
rc = recall_score(Yt, Yp, average=None, zero_division=0)

print(f"  类别 {len(sup)} 个 | 记录 {len(yt):,} 条 | mac-F1 {f1.mean():.3f} | "
      f"mic-F1 {f1_score(Yt, Yp, average='micro', zero_division=0):.3f}")
print()
h = ("支持度", "类数", "占类数", "占记录", "平均F1", "F1=0的类")
print(f"  {h[0]:<10} {h[1]:>6} {h[2]:>8} {h[3]:>8} {h[4]:>8} {h[5]:>9}")
for lo, hi, name in ((1, 1, "1 条"), (2, 2, "2 条"), (3, 5, "3-5 条"),
                     (6, 10, "6-10 条"), (11, 30, "11-30 条"), (31, 10 ** 9, ">30 条")):
    m = (sup >= lo) & (sup <= hi)
    if not m.any():
        continue
    print(f"  {name:<10} {m.sum():>6} {m.sum()/len(sup):>7.1%} "
          f"{sup[m].sum()/sup.sum():>7.1%} {f1[m].mean():>8.3f} {int((f1[m]==0).sum()):>9}")
print()
z = f1 == 0
print(f"  F1=0 的类共 {int(z.sum())} 个 = {z.mean():.1%} 的类, 但它们只占 "
      f"{sup[z].sum()/sup.sum():.1%} 的记录")
print(f"  这些类的支持度中位 {int(np.median(sup[z])) if z.any() else 0} 条")
print()
print(f"  {'只算支持度 >= N 的类':<22} {'类数':>6} {'mac-F1':>8}")
for n in (1, 2, 3, 5, 10, 20):
    m = sup >= n
    print(f"  {'N=' + str(n):<22} {m.sum():>6} {f1[m].mean():>8.3f}")
print()
print("  支持度最高的 8 个类:")
for i in np.argsort(-sup)[:8]:
    print(f"    {mlb.classes_[i]:<12} n={int(sup[i]):<4} F1={f1[i]:.3f}  "
          f"P={pr[i]:.3f} R={rc[i]:.3f}")
