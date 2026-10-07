# -*- coding: utf-8 -*-
"""可行性闸: 表那 5 列里的信息榨干了吗? —— 直接回答"agent 到底需不需要"。

★逻辑:
  手写规则 DP(先 tplD 再 prior) 拿到 test_overlap 整对 23.4% / 位置精确 34.7%。
  如果**一个只读表的小分类器**能明显超过它并逼近 oracle(表覆盖 59.5%),
  说明表里信息够、手写规则只是不够优 ⇒ **agent 的化学推理是多余的**。
  如果它卡在 DP 附近, 说明**表的列已经榨干** ⇒ 剩下的余量只能来自表外信息(反应化学),
  那才是 agent 的位置。
★只用表里有的列(模型看得到的): tplD, pL, prior, 有没有 tplAA, tplAA==ecAA, 行在表中的序位。
  **不许用任何来自答案的特征** —— 那就是循环。
★按【反应】切开拟合/评估(parent 前缀 = 反应 id), 否则同源会让分类器虚高
  (project_split_leaks_homologs)。
用法: table_info_ceiling.py <eval_heldout.jsonl> <overlap_pids>
"""
import collections
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

ev_f, ov_f = sys.argv[1], sys.argv[2]
OV = {x.strip() for x in open(ov_f)} - {""}

recs = []
for ln in open(ev_f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    d = json.loads(ln)
    if d["parent_id"] not in OV:
        continue
    u = json.loads(d["messages"][1]["content"])
    gold = {p["position_1based"] for p in json.loads(d["messages"][2]["content"])["picks"]}
    rows = []
    for i, r in enumerate(u["candidate_table"]):
        t = r.split()
        pos = int(t[0])
        rows.append({"pos": pos, "tplAA": t[1], "tplD": int(t[2]), "pL": int(t[3]),
                     "ecAA": t[4], "prior": int(t[5]), "ord": i,
                     "y": 1 if pos in gold else 0})
    recs.append({"rxn": d["parent_id"].split("_")[0], "k": len(gold), "rows": rows,
                 "L": d["length"], "nband": d["n_target_band"]})
print(f"  test_overlap 记录 {len(recs):,} | 反应 {len({r['rxn'] for r in recs}):,}")

rxn = sorted({r["rxn"] for r in recs})
rng = np.random.default_rng(0)
rng.shuffle(rxn)
half = set(rxn[:len(rxn) // 2])
fit = [r for r in recs if r["rxn"] in half]
ev = [r for r in recs if r["rxn"] not in half]
print(f"  按反应切开: 拟合 {len(fit):,} / 评估 {len(ev):,}")


def feats(rows, L):
    X = []
    for r in rows:
        X.append([r["tplD"], r["pL"], r["prior"], r["ord"] / max(1, len(rows)),
                  1.0 if r["tplAA"] != "-" else 0.0,
                  1.0 if (r["tplAA"] != "-" and r["tplAA"] == r["ecAA"]) else 0.0,
                  r["pos"] / max(1, L)])
    return np.asarray(X, dtype=np.float32)


Xf = np.concatenate([feats(r["rows"], r["L"]) for r in fit])
yf = np.concatenate([np.asarray([q["y"] for q in r["rows"]]) for r in fit])
print(f"  拟合样本 {len(yf):,} 行, 正例 {yf.mean():.2%}")

MODELS = {
    "LR 逻辑回归": LogisticRegression(max_iter=2000, C=1.0),
    "GB 梯度提升": GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                          subsample=0.7, random_state=0),
}
fitted = {}
for nm, m in MODELS.items():
    m.fit(Xf, yf)
    fitted[nm] = m
    print(f"  {nm} 训好")


def eval_arm(score_fn, name):
    hit = tot_pred = tot_gold = tot_band = 0
    for r in ev:
        k = r["k"]
        if not k:
            continue
        sc = score_fn(r)
        idx = np.argsort(-sc)[:k]
        g = {q["pos"] for q in r["rows"] if q["y"]}
        sel = [r["rows"][i]["pos"] for i in idx]
        hit += len(g & set(sel)); tot_pred += len(sel); tot_gold += len(g)
        tot_band += r["nband"]
    return (name, hit / max(1, tot_pred), hit / max(1, tot_gold), hit / max(1, tot_band))


arms = []
arms.append(eval_arm(lambda r: rng.random(len(r["rows"])), "R 表内随机"))
arms.append(eval_arm(lambda r: np.asarray([-(q["tplD"] * 10 + q["prior"])
                                           for q in r["rows"]], dtype=float), "DP 手写(先tplD再prior)"))
for nm, m in fitted.items():
    arms.append(eval_arm(lambda r, m=m: m.predict_proba(feats(r["rows"], r["L"]))[:, 1], nm))
arms.append(eval_arm(lambda r: np.asarray([q["y"] for q in r["rows"]], dtype=float),
                     "O oracle(=表覆盖)"))

print(f"\n  {'臂':<26}{'位置精确':>9}{'位置召回(表内)':>16}{'对整档召回':>12}")
for nm, pr, rc, bd in arms:
    print(f"  {nm:<26}{pr:8.2%}{rc:15.2%}{bd:11.2%}")
dp = [a for a in arms if a[0].startswith("DP")][0]
best_ml = max([a for a in arms if a[0][:2] in ("LR", "GB")], key=lambda a: a[1])
orc = [a for a in arms if a[0].startswith("O ")][0]
print(f"\n  ★判读: 手写 DP {dp[1]:.2%} → 最好的小分类器 {best_ml[1]:.2%} "
      f"(+{100*(best_ml[1]-dp[1]):.1f} 分) → oracle {orc[1]:.2%}")
gap_ml = orc[1] - best_ml[1]
print(f"  只读表的分类器离 oracle 还差 {100*gap_ml:.1f} 分 ⇒ "
      f"{'表的列已基本榨干, 剩下的余量只能来自反应化学 = agent 的位置' if gap_ml > 0.25 else '★表里还有不少没榨干的信息, 先调规则比上 agent 便宜'}")
for nm, m in fitted.items():
    if hasattr(m, "coef_"):
        names = ["tplD", "pL", "prior", "ord", "有tplAA", "两列一致", "pos/L"]
        print(f"  {nm} 的权重: " + "  ".join(
            f"{a}{b:+.2f}" for a, b in zip(names, m.coef_[0])))
    elif hasattr(m, "feature_importances_"):
        names = ["tplD", "pL", "prior", "ord", "有tplAA", "两列一致", "pos/L"]
        print(f"  {nm} 的重要度: " + "  ".join(
            f"{a}{b:.2f}" for a, b in zip(names, m.feature_importances_)))
