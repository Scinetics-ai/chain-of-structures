# -*- coding: utf-8 -*-
"""R1 闸: 「位置由草稿的口袋给、身份由 agent 给」可行吗? 本地 H100, 生产机不碰。

★要判的那件事:
  我们现在的流水线是【带钉子的全局共生成】—— 每轮把前几档注入后重新生成整张草稿。
  而那正是实测"只钉身份不定位"的那个 régime: 天然位置 12.56Å vs 随机位置 14.96Å, p=0.665。
  régime 表另一头: "给完整合法链、掩掉口袋" 是生成器最强的一格(32/32 合法, 几何 0.80Å),
  而它恰好做不了身份(全对 1/32, 逐残基 30.7%) —— 身份正是 agent 强的那部分。
  所以要比的不是"位置怎么挑", 是【两个 régime 本身】。

★五个臂(全部只钉身份、结构自由, 同一条无条件草稿上做):
  CO_native   空画布 + 天然位置钉身份 + 全局共生成   ← 现在的流水线
  CO_pocket   空画布 + 口袋位置钉身份 + 全局共生成
  IP_native   保留草稿, 只在天然位置±W 重算结构     ← 换 régime
  IP_pocket   保留草稿, 只在口袋位置±W 重算结构     ← 提议的做法
  IP_shuf     同 IP_pocket 但把残基型打乱            ← 内容空对照
★空对照必须用【错配/打乱】不是随机(project_structure_bit_pins_reproduce_native_geometry:
  随机 bit 反而好于错配, 用随机会低估难度)。
★口袋只用草稿几何挑, 不碰天然信息。
★判据沿用项目口径(最近一对 CA), 可直接和 6.87 / 12.56 / 14.96 比。
★多步扩散能不能修回来 = max_iter 这个旋钮, 所以同一批跑两个步数。
用法: r1_regime_experiment.py <core_ec.json> <out.jsonl> [--limit N] [--iters 100,400] [--seed S]
"""
import argparse
import json
import random
import time

import numpy as np
import torch

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import geom
from dplm_local import load

W = 2            # IP 臂: 在每个目标位置两侧各放开 W 个残基的结构
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def aa_ids(m):
    t = m.tokenizer
    d = {}
    for c in AA20:
        i = t.convert_tokens_to_ids(c)
        back = t.convert_ids_to_tokens(i)
        assert back == c, f"★氨基酸 token 往返不一致: {c} -> {i} -> {back}"
        d[c] = i
    return d


def blank(m, L, B, dev):
    """全掩码画布, 两半等长(chunk(2,dim=1) 硬编码)。"""
    st = torch.full((B, L + 2), m.struct_mask_id, dtype=torch.long, device=dev)
    st[:, 0] = m.struct_bos_id; st[:, -1] = m.struct_eos_id
    aa = torch.full((B, L + 2), m.aa_mask_id, dtype=torch.long, device=dev)
    aa[:, 0] = m.aa_bos_id; aa[:, -1] = m.aa_eos_id
    return torch.cat([st, aa], 1)


def gen(m, x, pm, it, seed):
    """★必须和生产的生成参数逐字一致, 否则不是对照(feedback_two_machines_are_not_a_control):
    生产 = `--bf16 --max-iter 100`, temperature 1.0, 且【不覆盖】
    unmasking_strategy / sampling_strategy(用签名默认 annealing@1.1:0.1)。
    我第一版自作主张写了 annealing@2.2:1.0(更热), 结果草稿里多出一堆塌缩,
    还差点让我得出"无条件草稿没有可用口袋"的错结论 —— 那个读数已作废。"""
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m.generate(input_tokens=x, max_iter=it, temperature=1.0, partial_masks=pm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("core_ec"); ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--iters", default="100,400")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmin", type=int, default=150); ap.add_argument("--lmax", type=int, default=320)
    ap.add_argument("--kmax", type=int, default=14)
    a = ap.parse_args()
    iters = [int(x) for x in a.iters.split(",")]

    cm = json.load(open(a.core_ec))
    pool = [(p, v) for p, v in cm.items()
            if a.lmin <= v["length"] <= a.lmax and 3 <= len(v["core"]) <= a.kmax
            and all(c in AA20 for c in v["aa"])]
    pool.sort(key=lambda kv: kv[0])
    random.Random(1234).shuffle(pool)
    pool = pool[:a.limit]
    print(f"  候选 {len(pool)} 条 (长度 {a.lmin}-{a.lmax}, 核心 3-{a.kmax} 个)", flush=True)

    m = load()
    AID = aa_ids(m)
    dev = "cuda"
    ARMS = ["CO_native", "CO_pocket", "IP_native", "IP_pocket", "IP_shuf"]
    w = open(a.out_jsonl, "w")
    t0 = time.time()

    for n, (pid, v) in enumerate(pool, 1):
        L, core, caa = v["length"], v["core"], v["aa"]
        k = len(core)
        nat0 = np.array([p - 1 for p in core if 1 <= p <= L])
        if len(nat0) < 3:
            continue
        caa = [caa[i] for i, p in enumerate(core) if 1 <= p <= L]

        # ---- 0) 无条件草稿 ----
        d0 = gen(m, blank(m, L, 1, dev), None, 100, a.seed)
        tok0 = d0["output_tokens"]
        pos0, pl0 = geom.decode(m, d0["final_struct_feature"], L)
        step0 = geom.ca_step_median(pos0[0], m)
        if not (3.6 <= step0 <= 4.0):
            print(f"  [{n}] {pid} 草稿相邻CA {step0:.2f}Å 不合法 ⇒ 跳过", flush=True)
            continue
        poc0, gyr_poc = geom.pick_pocket(pos0[0], m, k)
        draft_aa = "".join(m.tokenizer.convert_ids_to_tokens(
            tok0[0, L + 3:2 * L + 3].tolist()))

        ref = {"draft_at_pocket": geom.site_metrics(pos0[0], m, poc0),
               "draft_at_native": geom.site_metrics(pos0[0], m, nat0)}

        for it in iters:
            B = len(ARMS)
            st0 = tok0[:, :L + 2].repeat(B, 1).clone()
            aa0 = tok0[:, L + 2:].repeat(B, 1).clone()
            x = blank(m, L, B, dev)
            pm = torch.zeros_like(x, dtype=torch.bool)
            shuf = list(caa); random.Random(a.seed + n).shuffle(shuf)
            tgt = {"CO_native": (nat0, caa), "CO_pocket": (poc0, caa),
                   "IP_native": (nat0, caa), "IP_pocket": (poc0, caa),
                   "IP_shuf": (poc0, shuf)}
            for bi, arm in enumerate(ARMS):
                idx0, res = tgt[arm]
                if arm.startswith("CO"):
                    # 空画布: 只把目标位置的氨基酸设为给定, 其余全生成
                    for j, p0 in enumerate(idx0):
                        x[bi, L + 2 + p0 + 1] = AID[res[j]]
                        pm[bi, L + 2 + p0 + 1] = True
                else:
                    # 保留草稿: 氨基酸全给定(目标位换成指定残基), 结构只在目标位±W 放开
                    x[bi, :L + 2] = st0[bi]
                    x[bi, L + 2:] = aa0[bi]
                    for j, p0 in enumerate(idx0):
                        x[bi, L + 2 + p0 + 1] = AID[res[j]]
                    # 全部先标成"给定", 再只把要重算的那几段放开。
                    # bos/eos 不用管: get_non_special_symbol_mask 本来就排除特殊符号。
                    pm[bi, :] = True
                    for p0 in idx0:
                        lo, hi = max(0, p0 - W), min(L - 1, p0 + W)
                        pm[bi, 1 + lo:1 + hi + 1] = False               # 这一段结构重算
            out = gen(m, x, pm, it, a.seed + 7)
            pos, pl = geom.decode(m, out["final_struct_feature"], L)
            got = out["output_tokens"]
            for bi, arm in enumerate(ARMS):
                idx0, res = tgt[arm]
                seq = "".join(m.tokenizer.convert_ids_to_tokens(
                    got[bi, L + 3:2 * L + 3].tolist()))
                comp = float(np.mean([seq[p0] == res[j] for j, p0 in enumerate(idx0)]))
                sm = geom.site_metrics(pos[bi], m, idx0)
                rec = {"parent_id": pid, "length": L, "k": k, "arm": arm, "max_iter": it,
                       "seed": a.seed, "compliance": round(comp, 4),
                       "ca_step_median": round(geom.ca_step_median(pos[bi], m), 3),
                       "plddt_med": round(float(np.median(pl[bi])), 1),
                       "site": {kk: round(vv, 3) for kk, vv in sm.items()},
                       "draft_ref": {k2: {k3: round(v3, 3) for k3, v3 in v2.items()}
                                     for k2, v2 in ref.items()},
                       "pocket_gyr_at_pick": round(gyr_poc, 3),
                       "native_idx": (nat0 + 1).tolist(), "pocket_idx": (poc0 + 1).tolist(),
                       "res_requested": list(res), "draft_aa_at_sites":
                           "".join(draft_aa[p0] for p0 in idx0)}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()
        el = time.time() - t0
        print(f"  [{n}/{len(pool)}] {pid} L={L} k={k} 口袋回转{gyr_poc:.1f}Å "
              f"| 用时 {el/n:.0f}s/条", flush=True)
    w.close()
    print(f"  完成, 共 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
