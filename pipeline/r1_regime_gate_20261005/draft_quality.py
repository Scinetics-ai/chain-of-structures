# -*- coding: utf-8 -*-
"""无条件草稿的局部几何到底能不能读出"口袋"? —— 和生产的第2轮草稿逐项对照。

★为什么这是前置闸: 「位置由草稿的几何给」这个方案的第一步就是"在草稿里找口袋"。
  若无条件草稿的局部几何本身不可信(有塌缩), 那"最聚拢的一簇"找到的是塌缩区不是口袋,
  方案在第一步就断了。
★必须精确复现生产的生成参数, 否则不是对照(feedback_two_machines_are_not_a_control):
  生产 = --bf16 --max-iter 100, temperature 1.0, 且【不覆盖】
  unmasking_strategy / sampling_strategy(用 generate 签名默认 annealing@1.1:0.1)。
  我第一版写了 annealing@2.2:1.0(更热), 那个读数作废。
★判据: 非局部(|i-j|>=4) CA-CA 最近距离 与 <4.0Å 的对数。
  第2轮草稿(天然核心已按天然位置注入, n=120, L 180-260) 的实测:
    非局部最近 中位 3.88Å(四分位 3.57-4.21) | <4.0Å 对数 中位 1(四分位 0-2) | 回转 16.2
用法: draft_quality.py <core_ec.json> [n] [--hot]
"""
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
import geom as G
from dplm_local import load

core_ec = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HOT = "--hot" in sys.argv

m = load()
ca = G.atom_idx("CA", m)
cm = json.load(open(core_ec))
pool = [(p, v) for p, v in cm.items() if 180 <= v["length"] <= 260]
pool.sort(key=lambda kv: kv[0])
import random
random.Random(1234).shuffle(pool)
pool = pool[:N]
print(f"  无条件草稿 n={len(pool)}  采样={'annealing@2.2:1.0(热, 作废口径)' if HOT else '生产默认'}", flush=True)

rows = []
for i, (pid, v) in enumerate(pool, 1):
    L = v["length"]
    st = torch.full((1, L + 2), m.struct_mask_id, dtype=torch.long, device="cuda")
    st[:, 0] = m.struct_bos_id; st[:, -1] = m.struct_eos_id
    aa = torch.full((1, L + 2), m.aa_mask_id, dtype=torch.long, device="cuda")
    aa[:, 0] = m.aa_bos_id; aa[:, -1] = m.aa_eos_id
    torch.manual_seed(0)
    kw = {"unmasking_strategy": "stochastic1.0", "sampling_strategy": "annealing@2.2:1.0"} if HOT else {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = m.generate(input_tokens=torch.cat([st, aa], 1), max_iter=100, temperature=1.0, **kw)
    pos, pl = G.decode(m, o["final_struct_feature"], L)
    c = pos[0, :L, ca]
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    ii, jj = np.triu_indices(L, 4)
    nl = d[ii, jj]
    rows.append((L, float(np.median(np.linalg.norm(c[1:] - c[:-1], axis=-1))), float(nl.min()),
                 int(np.sum(nl < 4.0)), int(np.sum(nl < 4.5)),
                 float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())), float(np.median(pl[0]))))
    if i % 10 == 0:
        print(f"    {i}/{len(pool)}", flush=True)

a = np.array(rows)
q = lambda col, p: np.percentile(a[:, col], p)
print()
print(f"  {'指标':<22} {'中位':>8} {'四分位':>16}")
for nm, col in (("相邻CA距离", 1), ("非局部最近CA", 2), ("<4.0Å的非局部对数", 3),
                ("<4.5Å的非局部对数", 4), ("整体回转半径", 5), ("plddt", 6)):
    print(f"  {nm:<22} {np.median(a[:, col]):8.2f} {q(col,25):7.2f}-{q(col,75):<8.2f}")
print()
print("  对照(生产第2轮草稿, 天然核心已注入, n=120):")
print("    非局部最近CA 中位 3.88 (3.57-4.21) | <4.0Å 对数 中位 1 (0-2) | 回转 16.2")
print()
print("  判读: 若无条件草稿的 <4.0Å 对数远多于 1 ⇒ 局部几何不可信,")
print("        「在草稿里找口袋」这一步就不成立, 不是调判据能救的。")
