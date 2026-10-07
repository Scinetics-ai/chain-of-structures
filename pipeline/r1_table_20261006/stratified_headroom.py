# -*- coding: utf-8 -*-
"""按 tplD 分层算余量: 表在哪一层已经够用, agent 该在哪一层发力。

★为什么必须分层(feedback_stratify_on_external_reference):
  最强手写规则 DP 的"氨基酸@选对 67.43%"是**挑容易的挑出来的** ——
  它专取 tplD 小的行, 而 oracle 位置下 tplAA 只有 36.95%。
  不分层就会把"表给的简单子集"说成"方法的本事"。
  tplD 是**输入特征**, 不是被测量的东西 ⇒ 用它分层是合法的。

★对每个【真实核心位置】问三件事:
  1 它在表里吗(不在 ⇒ 任何 agent 都选不到, 这是任务上界)
  2 它那行的 tplD 多大(决定 tplAA 可不可信)
  3 tplAA / ecAA 在那里对不对(查表能不能供出氨基酸)
用法: stratified_headroom.py <eval_heldout.jsonl> <overlap_pids> <strict_pids>
"""
import collections
import json
import sys

ev_f, ov_f, st_f = sys.argv[1:4]


def rd(p):
    return {x.strip() for x in open(p)} - {""}


OV, ST = rd(ov_f), rd(st_f)
BANDS = [("tplD 0-1 (对得很准)", 0, 1), ("tplD 2-4 (中)", 2, 4),
         ("tplD 5-9 / 无模板", 5, 9)]
S = collections.defaultdict(lambda: collections.Counter())

for ln in open(ev_f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    d = json.loads(ln)
    g = "test_overlap" if d["parent_id"] in OV else (
        "test_strict" if d["parent_id"] in ST else "其他")
    u = json.loads(d["messages"][1]["content"])
    gold = {p["position_1based"]: p["aa"]
            for p in json.loads(d["messages"][2]["content"])["picks"]}
    row = {}
    for r in u["candidate_table"]:
        t = r.split()
        row[int(t[0])] = {"tplAA": t[1], "tplD": int(t[2]), "ecAA": t[4],
                          "prior": int(t[5])}
    for pos, aa in gold.items():
        r = row[pos]                       # gold 已经是【本档 ∩ 表】, 所以必在表里
        nm = next(b for b, lo, hi in BANDS if lo <= r["tplD"] <= hi)
        c = S[(g, nm)]
        c["n"] += 1
        c["tplAA对"] += int(r["tplAA"] == aa)
        c["tplAA给了"] += int(r["tplAA"] != "-")
        c["ecAA对"] += int(r["ecAA"] == aa)
        c["prior前2成"] += int(r["prior"] <= 1)
        c["prior桶和"] += r["prior"]

print(f"  {'组':<13}{'tplD 层':<22}{'占比':>7}{'tplAA给了':>10}{'tplAA对':>9}"
      f"{'ecAA对':>8}{'落在prior前2成':>15}")
for g in ("test_overlap", "test_strict"):
    tot = sum(S[(g, b)]["n"] for b, _, _ in BANDS)
    if not tot:
        continue
    for b, _, _ in BANDS:
        c = S[(g, b)]
        if not c["n"]:
            continue
        print(f"  {g:<13}{b:<22}{c['n']/tot:6.1%}{c['tplAA给了']/c['n']:9.1%}"
              f"{c['tplAA对']/c['n']:8.1%}{c['ecAA对']/c['n']:7.1%}"
              f"{c['prior前2成']/c['n']:14.1%}")
    print(f"  {'':<13}{'合计':<22}{1.0:6.1%}"
          f"{sum(S[(g,b)]['tplAA给了'] for b,_,_ in BANDS)/tot:9.1%}"
          f"{sum(S[(g,b)]['tplAA对'] for b,_,_ in BANDS)/tot:8.1%}"
          f"{sum(S[(g,b)]['ecAA对'] for b,_,_ in BANDS)/tot:7.1%}"
          f"{sum(S[(g,b)]['prior前2成'] for b,_,_ in BANDS)/tot:14.1%}")
    print()
print("  ★读法: tplAA对 高的那一层, 查表已经够用, agent 顶多持平;")
print("        tplAA对 低的那一层才是 agent 的战场 —— 那里只能靠化学推理。")
print("        '落在prior前2成' 低 = 手写规则根本排不到它 ⇒ 位置通道的余量也在这一层。")
