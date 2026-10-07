# -*- coding: utf-8 -*-
"""分析 R1 régime 闸。全部用【配对符号检验】—— 五个臂在同一条草稿上, 配对是天然的。

★口径沿用项目已有参照, 可直接对照:
  天然结构本身                  最近一对 CA 6.87Å  <8Å 80.6%  <12Å 100%
  钉天然结构bit@天然位置         6.08Å  67.7%  77.4%
  只钉身份@天然位置(共生成)      12.56Å  33.3%
  只钉身份@随机位置(共生成)      14.96Å  12.5%
★先看遵守率: 遵守率不够高, 后面的几何比较没有意义。
★几何闸: 相邻 CA 中位不在 3.6-4.0 的样本整条作废。
用法: analyze_regime.py <out.jsonl>
"""
import collections
import json
import sys

import numpy as np
from scipy.stats import binomtest

ARMS = ["CO_native", "CO_pocket", "IP_native", "IP_pocket", "IP_shuf"]
NAT_REF = 6.87

rows = [json.loads(l) for l in open(sys.argv[1])]
iters = sorted({r["max_iter"] for r in rows})
print(f"  判据用【非局部对】(|i-j|>=4)的最近距离 —— 含相邻残基的集合上 min_CA 恒为 3.8Å, 没有区分力\n")
print(f"  记录 {len(rows)} 条 | 步数 {iters} | parent {len({r['parent_id'] for r in rows})}\n")

for it in iters:
    sub = [r for r in rows if r["max_iter"] == it]
    by = collections.defaultdict(dict)
    for r in sub:
        by[r["parent_id"]][r["arm"]] = r
    pids = [p for p, d in by.items() if len(d) == len(ARMS)
            and all(3.6 <= d[a]["ca_step_median"] <= 4.0 for a in ARMS)]
    drop = len(by) - len(pids)
    print(f"=== max_iter={it}  配对完整且几何合法的 parent {len(pids)}"
          f"{f' (几何闸作废 {drop} 条)' if drop else ''} ===")
    if not pids:
        continue
    print(f"  {'臂':<11} {'遵守率':>7} {'非局部中位CA':>12} {'<8Å对占比':>9} {'<12Å对占比':>10} "
          f"{'非局部最近CA':>12} {'回转半径':>8} {'plddt':>6}")
    M = {}
    for a in ARMS:
        med = np.array([by[p][a]["site"]["med_CA_nl"] for p in pids])
        mn = np.array([by[p][a]["site"]["min_CA_nl"] for p in pids])
        f8 = np.array([by[p][a]["site"]["frac_nl_lt8"] for p in pids])
        f12 = np.array([by[p][a]["site"]["frac_nl_lt12"] for p in pids])
        gy = np.array([by[p][a]["site"]["gyr"] for p in pids])
        co = np.array([by[p][a]["compliance"] for p in pids])
        pl = np.array([by[p][a]["plddt_med"] for p in pids])
        M[a] = med                               # 主判据用【中位】, 对 k 稳健
        print(f"  {a:<11} {co.mean():6.1%} {np.median(med):10.2f} {np.mean(f8):8.1%} "
              f"{np.mean(f12):9.1%} {np.median(mn):10.2f} {np.median(gy):8.1f} {np.median(pl):6.0f}")
    d0 = np.array([by[pids[0]][ARMS[0]]["draft_ref"]["draft_at_pocket"]["min_CA_nl"]])
    dp = np.array([by[p][ARMS[0]]["draft_ref"]["draft_at_pocket"]["med_CA_nl"] for p in pids])
    dn = np.array([by[p][ARMS[0]]["draft_ref"]["draft_at_native"]["med_CA_nl"] for p in pids])
    print(f"  {'(编辑前草稿)':<11} {'-':>6} 口袋处 {np.median(dp):10.2f} {'':>8} {'':>9} "
          f"天然位置处 {np.median(dn):.2f}")
    print("  ★不要拿这几个数去对 6.87 / 12.56 那组参照: 那组是钉【3 个】位点测的,")
    print("    最近一对会随 k 变大机械变小。臂间【配对】比较才有效(同草稿、同 k)。")
    print()
    print(f"  {'配对比较':<28} {'胜:负':>9} {'Δ中位':>8} {'p(符号)':>10}")
    for x, y, why in (("IP_pocket", "CO_native", "提议 vs 现在的流水线"),
                      ("IP_pocket", "IP_native", "同régime: 口袋 vs 天然位置"),
                      ("CO_pocket", "CO_native", "现régime: 口袋 vs 天然位置"),
                      ("IP_pocket", "IP_shuf", "内容对照: 真残基型 vs 打乱"),
                      ("IP_native", "CO_native", "同位置: 换régime")):
        a, b = M[x], M[y]
        win = int(np.sum(a < b)); los = int(np.sum(a > b))
        p = binomtest(win, win + los, 0.5).pvalue if win + los else 1.0
        print(f"  {x:>10} vs {y:<10} {win:>4}:{los:<4} {np.median(a)-np.median(b):+8.2f} "
              f"{p:10.2g}   {why}")
    print()
