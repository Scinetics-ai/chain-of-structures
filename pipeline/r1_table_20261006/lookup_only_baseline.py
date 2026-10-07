# -*- coding: utf-8 -*-
"""查表-only 基线: 不要 agent, 只用【表里有的信息】取前 k 个。这是 agent 必须打赢的那条线。

★论文现在明确承诺要报这条(feedback_retrieval_baseline_is_now_a_published_threat:
  指纹最近邻 46.62% 打赢模型 38.83%, 所以必须自己先报, 别让审稿人来报)。
★只能用表里有的: 行给的是 `prior` 的 0-9 【桶】, 不是精确排名 ⇒ 桶内并列按随机打破,
  多个种子平均。用精确排名会高估基线能力(模型也看不到精确排名)。
★氨基酸用 `tplAA` 那一列(对齐不可信时是 '-', 此时退回该档最常见的催化残基)。
★两个组分开报(test_overlap 97% / test_strict 3%), 永远不要合在一起。
用法: lookup_only_baseline.py <eval_heldout.jsonl> <test_overlap_pids> <test_strict_pids> [seeds=20]
"""
import collections
import json
import random
import sys

ev_f, ov_f, st_f = sys.argv[1:4]
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 20


def rd(p):
    return {x.strip() for x in open(p)} - {""}


OV, ST = rd(ov_f), rd(st_f)
acc = collections.defaultdict(lambda: collections.defaultdict(float))

rows_all = []
for ln in open(ev_f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    d = json.loads(ln)
    u = json.loads(d["messages"][1]["content"])
    gold = {p["position_1based"]: p["aa"]
            for p in json.loads(d["messages"][2]["content"])["picks"]}
    tbl = []
    for r in u["candidate_table"]:
        t = r.split()
        tbl.append({"pos": int(t[0]), "tplAA": t[1], "tplD": int(t[2]),
                    "pL": int(t[3]), "ecAA": t[4], "prior": int(t[5])})
    g = "test_overlap" if d["parent_id"] in OV else (
        "test_strict" if d["parent_id"] in ST else "其他")
    # ★有些输入(直接用 SFT 记录当评测集)没有 n_target_band。
    #   **不许拿别的数代替** —— 那会把"对整档召回"错标成"位置召回"。缺了就记 None, 后面打 '-'。
    rows_all.append((g, gold, tbl, d.get("n_target_band")))
print(f"  评测记录 {len(rows_all):,} | 组 {collections.Counter(g for g,_,_,_ in rows_all)}")

for seed in range(SEEDS):
    rng = random.Random(1000 + seed)
    for g, gold, tbl, nband in rows_all:
        k = len(gold)
        # ★把手写规则扫到最强, 否则是赢了一个假基线
        #   (feedback_unpredictable_target_makes_the_baseline_unbeatable 的反面教训)
        RULES = {
            "R 表内随机": lambda r: (rng.random(),),
            "P prior": lambda r: (r["prior"], rng.random()),
            "D tplD": lambda r: (r["tplD"], rng.random()),
            "PD prior+tplD": lambda r: (r["prior"], r["tplD"], rng.random()),
            "DP tplD+prior": lambda r: (r["tplD"], r["prior"], rng.random()),
            "PDA prior+tplD+有tplAA": lambda r: (r["prior"], r["tplD"],
                                                 0 if r["tplAA"] != "-" else 1, rng.random()),
            "AGREE 两列一致优先": lambda r: (0 if (r["tplAA"] != "-" and r["tplAA"] == r["ecAA"])
                                        else 1, r["prior"], r["tplD"], rng.random()),
            "SUM prior+tplD 相加": lambda r: (r["prior"] + r["tplD"], rng.random()),
        }
        # ★两条 oracle 臂: 不是基线, 是【天花板】—— 用来算 agent 还剩多少余量
        #   O1 位置全对+氨基酸仍取 tplAA ⇒ 隔离出"位置选对了的话 tplAA 够不够用"
        #   O2 位置全对+氨基酸也全对 ⇒ 这张表能达到的上限(= 表对本档的覆盖)
        ORACLE = {"O1 oracle位置+tplAA": 1, "O2 oracle位置+oracle氨基酸": 2}
        for arm, key in list(RULES.items()) + [(x, None) for x in ORACLE]:
            if arm == "R 表内随机":
                sel = rng.sample(tbl, min(k, len(tbl)))
            elif arm in ORACLE:
                sel = [r for r in tbl if r["pos"] in gold][:k]
            else:
                sel = sorted(tbl, key=key)[:k]
            hit = [r for r in sel if r["pos"] in gold]
            right = 0
            for r in hit:
                if arm == "O2 oracle位置+oracle氨基酸":
                    right += 1                      # 氨基酸也由 oracle 给
                else:
                    aa = r["tplAA"] if r["tplAA"] != "-" else r["ecAA"]
                    right += int(aa == gold[r["pos"]])
            a = acc[(g, arm)]
            a["n_pred"] += len(sel); a["n_gold"] += k
            a["pos_hit"] += len(hit); a["pair_hit"] += right
            if nband is not None:
                a["n_band"] += nband
            else:
                a["no_band"] += 1
            a["n"] += 1

print(f"\n  {'组':<13}{'臂':<26}{'位置精确':>9}{'位置召回':>9}{'氨基酸@选对':>12}"
      f"{'整对精确':>9}{'对整档的召回':>13}")
for g in ("test_overlap", "test_strict"):
    arms = sorted({k[1] for k in acc})
    base = [x for x in arms if not x.startswith("O")]
    best = max(base, key=lambda x: acc[(g, x)]["pair_hit"] / max(1, acc[(g, x)]["n_gold"]))
    for arm in arms:
        a = acc.get((g, arm))
        if not a or not a["n_pred"]:
            continue
        print(f"  {g:<13}{arm:<26}{a['pos_hit']/a['n_pred']:8.2%}"
              f"{a['pos_hit']/a['n_gold']:9.2%}{a['pair_hit']/max(1,a['pos_hit']):11.2%}"
              f"{a['pair_hit']/a['n_gold']:9.2%}"
              + (f"{a['pos_hit']/a['n_band']:12.2%}" if a["n_band"] else "          -")
              + ("   ←最强手写" if arm == best else
                 ("   ←天花板" if arm.startswith("O") else "")))
    print()
print(f"  ({SEEDS} 个种子平均; 桶内并列按随机打破 —— 模型看到的也只是 0-9 的桶)")
print("  ★读法: agent 要打赢的是【最强】那一行(实测是 DP: 先 tplD 再 prior)。")
print("        'tplD 优先' 比 'prior 优先' 好, 因为 tplD 小的时候 tplAA 才可信 ——")
print("        这也解释了为什么 DP 的氨基酸@选对 (67.4%) 远高于 P 的 (56.2%)。")
print("        '对整档的召回' 分母是整个催化核心, 不是表里那部分 —— 表覆盖是任务上界,")
print("        两个分母都要报, 否则会把'表给的'说成'模型做的'。")
