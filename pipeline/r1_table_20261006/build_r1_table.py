# -*- coding: utf-8 -*-
"""给 R1 造候选表 —— 让 R1 和 R2-R4 同构。表**完全由查表构造**, 不需要草稿、不碰天然结构。

★为什么是查表而不是草稿(闸1 实测, 2026-10-06, 真无条件草稿 n=2,925 配对):
    草稿几何 dCen  0.95x  <- 比随机还差, 零信号
    模板检索       5.90x
    模板+草稿      4.57x  <- 加草稿是净负作用
  之前那个 3.24x 是拿【第2轮草稿】当代理算的, 而第2轮草稿是注入 R1 核心后重新生成的,
  几何带着答案 ⇒ 泄漏。见 [[feedback_later_round_draft_is_not_an_unconditional_proxy]]。

★全量实测(44,465 parent / 评估 21,622 / 按反应切开), 不需要草稿的信息源:
    随机                 5.55%  1.00x  前100覆盖 34.6%
    R1 现在(实测)         8.2%  1.68x  没有表
    p/L 全局先验         12.44%  2.24x  43.3%
    p/L x EC(降级)       15.26%  2.75x  53.1%
    模板检索(同EC3)       22.62%  4.07x  47.6%
    模板+先验            20.49%  3.69x  55.3%   <- 本脚本用这个
  氨基酸通道: 恒答 12.91% / EC 条件化 17.27% / EC x 位置桶 18.78% / **模板对齐残基 25.03%**
  组成通道(与位置无关): EC 条件化 F1 **0.555** vs 全局 0.440
  个数通道: 不可预测(EC 5.60 vs 常数 5.63) ⇒ **不让 R1 猜个数**, 由"目标=本档∩表"定

★两道防泄漏(这是本脚本最容易写错的地方):
  1 **模板库只用 split=='train' 的 parent。** 给 val/test 造表时若用了它们自己, 就是测试集泄漏。
  2 **留一反应(leave-one-reaction-out)**: 模板库和 p/L 先验都要减掉【本条所属那条反应】的贡献。
    - 不减的话, 训练记录的表会虚高(先验里含它自己的位置), 模型学会过度信任表,
      而 val/test 的表没这么强 ⇒ 训练/推理不一致。
    - 实测同反应兄弟值 **1.46x**(含 55.44% vs 排除 38.04%, 均为拟合侧虚高值, 只看相对差)。
    - 代价: test_overlap(占留出的 97%)真推理时兄弟是可用的, 所以这是**保守**口径;
      报数时两种口径都报(feedback_no_retrieval_generation 要求自己先报检索基线)。

★目标 = 本档位置 ∩ 候选表(与 v11 一致), 覆盖率作为**被记录的任务属性**,
  **绝不 `pool |= 答案`** —— 那在推理时不存在。
用法: build_r1_table.py <sft_r1_v11.jsonl> <r1_core_ec.json> <out.jsonl> [--budget 100]
"""
import argparse
import collections
import json
import math

import numpy as np

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
NB = 40          # p/L 分桶数
WIN = 3          # 模板锚点的软窗半径


def rxn_of(pid):
    return pid.split("_")[0]


def ec_levels(ec):
    f = ec.split(".")
    return [".".join(f[:k]) for k in (4, 3, 2, 1)]


TPL_TRUST = 5    # ★tplD 桶 >= 这个值就认为对齐不可信, tplAA 写 '-'
                 #   (blind 闸实测: 打乱的报告 1.23x < 不给报告 2.13x ⇒ 噪声列是净负作用)


def dbucket(d):
    """0->0, 1->1, 3->3, 7->4, 15->6, 30->7, 60->9"""
    return min(9, int(math.log1p(max(0, d)) * 2.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r1_jsonl"); ap.add_argument("core_ec"); ap.add_argument("out_jsonl")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--splits", default="", help="含 {train,val_overlap,val_strict}_pids.txt 的目录")
    ap.add_argument("--heldout-bands", default="",
                    help="留出集的 bands.jsonl; 它的记录会作为 split='heldout' 一起造表。"
                         "★必须走这同一份代码: 另写一个打分器就会出现两套政策"
                         "(feedback_same_class_arms_need_one_policy)")
    ap.add_argument("--no-exclude-rxn", action="store_true",
                    help="★只用于算【检索基线】: 不剔同反应兄弟。实测覆盖会到 96.5% = 答案在表里")
    a = ap.parse_args()

    def rd(fp):
        try:
            return {x.strip() for x in open(fp)} - {""}
        except OSError:
            return set()
    # ★sft_r1_v11.jsonl 里的 split 字段**全是 train**(划分是合并之后才做的), 不能信那个字段。
    SPL = {}
    for nm in ("train", "val_overlap", "val_strict"):
        for q in rd(f"{a.splits}/{nm}_pids.txt"):
            SPL[q] = nm
    assert SPL, "★--splits 必须指向含 {train,val_overlap,val_strict}_pids.txt 的目录"
    print(f"  划分文件: {collections.Counter(SPL.values())}", flush=True)
    if a.no_exclude_rxn:
        print("  ⚠ --no-exclude-rxn: 这一跑是【检索基线】, 不是我们的方案", flush=True)
    ce = json.load(open(a.core_ec))
    recs = []
    for ln in open(a.r1_jsonl, errors="ignore"):
        if not ln.lstrip().startswith("{"):
            continue
        d = json.loads(ln)
        if d.get("round_id") != "R1":
            continue
        pid = d["parent_id"]
        v = ce.get(pid)
        if not v or not v.get("core") or not v.get("ec"):
            continue
        u = json.loads(d["messages"][1]["content"])
        real = SPL.get(pid)
        if real is None:
            continue                      # 不在三份划分里 ⇒ 跳过(留出集走 --heldout-bands 那条路)
        recs.append({"parent_id": pid, "length": d["length"], "split": real,
                     "evidence_tier": d.get("evidence_tier"), "reaction": u["reaction"],
                     "ec": v["ec"], "core": v["core"], "aa": v["aa"]})
    n_held = 0
    if a.heldout_bands:
        for ln in open(a.heldout_bands, errors="ignore"):
            if not ln.lstrip().startswith("{"):
                continue
            d = json.loads(ln)
            pid = d["parent_id"]
            v = ce.get(pid)
            if not v or not v.get("core") or not v.get("ec"):
                continue
            try:
                ft = json.loads(d["function_text"])
            except Exception:
                continue
            recs.append({"parent_id": pid, "length": d["length"], "split": "heldout",
                         "evidence_tier": None,
                         "reaction": {"ec_numbers": ft.get("ec_numbers"),
                                      "reaction_smiles": ft.get("reaction_smiles"),
                                      "substrate_smiles": ft.get("substrate_smiles"),
                                      "rhea_id": ft.get("rhea_id"),
                                      "task": ft.get("function_text")},
                         "ec": v["ec"], "core": v["core"], "aa": v["aa"]})
            n_held += 1
        print(f"  留出集另加 {n_held:,} 条(split=heldout, 只造表不进教师)", flush=True)
    print(f"  读到 R1 记录 {len(recs):,} 条", flush=True)
    sp = collections.Counter(r["split"] for r in recs)
    print(f"  按真实划分 {dict(sp)}", flush=True)
    assert sp.get("train", 0) and sp.get("val_overlap", 0) and sp.get("val_strict", 0), \
        "★三份划分都必须非空"
    # ★留出集只造表, 绝不进模板库 —— 下面的库构造只收 split=="train"

    # ---------- 只用 train 建库; 同时按反应记下每条反应的贡献, 供留一使用 ----------
    pos_tot = collections.defaultdict(lambda: np.zeros(NB))
    pos_rxn = collections.defaultdict(lambda: np.zeros(NB))     # (ec_level, rxn) -> 直方图
    aab_tot = collections.defaultdict(collections.Counter)      # (ec3, bucket) -> aa 计数
    aab_rxn = collections.defaultdict(collections.Counter)      # (ec3, bucket, rxn)
    comp_tot = collections.defaultdict(collections.Counter)     # ec3 -> aa 计数(组成提示)
    comp_rxn = collections.defaultdict(collections.Counter)
    comp_n = collections.defaultdict(list)
    lib = collections.defaultdict(list)                         # ec3 -> [(fr, aas, L, rxn)]
    n_lib = 0
    for r in recs:
        if r["split"] != "train":          # ★只有 train 能进库
            continue
        n_lib += 1
        L, rk = r["length"], rxn_of(r["parent_id"])
        lvs = ec_levels(r["ec"])
        fr = []
        for pos, aa in zip(r["core"], r["aa"]):
            if not (1 <= pos <= L) or aa not in AA20:
                continue
            b = min(NB - 1, int((pos - 1) / L * NB))
            for e in lvs:
                pos_tot[e][b] += 1.0
                pos_rxn[(e, rk)][b] += 1.0
            aab_tot[(lvs[1], b)][aa] += 1
            aab_rxn[(lvs[1], b, rk)][aa] += 1
            comp_tot[lvs[1]][aa] += 1
            comp_rxn[(lvs[1], rk)][aa] += 1
            fr.append((pos - 1) / L)
        if fr:
            lib[lvs[1]].append((np.asarray(fr), list(r["aa"]), L, rk))
            comp_n[lvs[1]].append(len(fr))
    print(f"  模板库(只用 train 的 {n_lib:,} 条): EC3 类 {len(lib):,} | "
          f"模板 {sum(len(x) for x in lib.values()):,}", flush=True)
    # ★硬断言: 库成员必须全是 train。库里混进 val/留出 parent 就是测试集泄漏。
    libp = {q for q in SPL if SPL[q] != "train"}
    libmem = {r["parent_id"] for r in recs if r["split"] == "train"}
    assert not (libmem & libp), "★模板库里混进了非 train 的 parent"
    libr = {t[3] for v in lib.values() for t in v}
    nv = len({rxn_of(q) for q in libp if SPL[q] == "val_overlap"} & libr)
    print(f"  库里的反应 {len(libr):,} 个 | 其中 val_overlap 也用到的 {nv:,} 个"
          f" (按设计就该几乎全是; 留一反应在打分时逐条剔除自己那条)", flush=True)

    n = 0
    tdiag = collections.defaultdict(lambda: collections.defaultdict(
        lambda: 0) | {"td": []})
    skip = collections.Counter()
    cov_by_split = collections.defaultdict(list)
    ntgt_by_split = collections.defaultdict(list)
    has_tpl = collections.defaultdict(list)
    tier_used = collections.Counter()
    rows_len = []
    w = open(a.out_jsonl, "w")
    for r in recs:
        L, pid, rk = r["length"], r["parent_id"], rxn_of(r["parent_id"])
        lvs = ec_levels(r["ec"])
        core = [p for p in r["core"] if 1 <= p <= L]
        if not core:
            skip["核心为空"] += 1
            continue

        # ---- 留一反应的先验 ----
        lvl = None
        for e in lvs:
            h = pos_tot[e] if a.no_exclude_rxn else pos_tot[e] - pos_rxn[(e, rk)]
            if h.sum() > 5:
                lvl, hist = e, h
                break
        if lvl is None:
            lvl, hist = "全局", np.zeros(NB)
        tier_used["EC%d" % len(lvl.split(".")) if lvl != "全局" else "全局"] += 1
        hist = hist + 1.0                                   # 加一平滑
        bi = np.minimum(NB - 1, (np.arange(L) / L * NB).astype(int))
        pe = hist[bi] / hist.sum()
        zpe = (pe - pe.mean()) / (pe.std() + 1e-9)

        # ---- 留一反应的模板 ----
        cand = (list(lib.get(lvs[1], [])) if a.no_exclude_rxn
                else [t for t in lib.get(lvs[1], []) if t[3] != rk])
        has_tpl[r["split"]].append(1.0 if cand else 0.0)
        tscore = np.zeros(L)
        tpl_aa_at = {}
        if cand:
            t = min(cand, key=lambda x: abs(x[2] - L))
            tp = np.clip((t[0] * L).round().astype(int), 0, L - 1)
            for j, aa in zip(tp, t[1]):
                lo, hi = max(0, j - WIN), min(L, j + WIN + 1)
                idx = np.arange(lo, hi)
                tscore[lo:hi] = np.maximum(tscore[lo:hi], 1.0 - np.abs(idx - j) / (WIN + 1))
                tpl_aa_at.setdefault(int(j) + 1, aa if aa in AA20 else "-")
            zt = (tscore - tscore.mean()) / (tscore.std() + 1e-9)
            score = zt + zpe
        else:
            score = zpe

        order = np.argsort(-score)[:min(a.budget, L)]
        pool = sorted(int(i) + 1 for i in order)
        # ★按表内排名分桶(0=打分最高的那一成), 不是按 L —— 按 L 的话长链只会用到 0-1 桶
        prank = {int(j) + 1: i for i, j in enumerate(order)}
        nb_pool = max(1, len(order))
        anchors = sorted(tpl_aa_at)

        rows = []
        for p in pool:
            b = min(NB - 1, int((p - 1) / L * NB))
            if anchors:
                q = min(anchors, key=lambda x: abs(x - p))
                td = dbucket(abs(q - p))
                taa = tpl_aa_at[q] if td < TPL_TRUST else "-"
            else:
                taa, td = "-", 9
            bb = aab_tot.get((lvs[1], b), collections.Counter())
            if not a.no_exclude_rxn:
                bb = bb - aab_rxn.get((lvs[1], b, rk), collections.Counter())
            ecaa = bb.most_common(1)[0][0] if bb else "-"
            rows.append(" ".join([str(p), taa, str(td), str(int(b / NB * 10)), ecaa,
                                  str(min(9, int(prank[p] / nb_pool * 10)))]))
        rows_len.append(len(rows))

        seq_aa0 = {p: aa for p, aa in zip(r["core"], r["aa"])}
        target = [p for p in core if p in set(pool)]
        if not target:
            skip["目标全在表外"] += 1
            continue
        cov_by_split[r["split"]].append(len(target) / len(core))
        ntgt_by_split[r["split"]].append(len(target))
        rowmap = {int(x.split()[0]): x.split() for x in rows}
        for p in target:
            f = rowmap[p]
            tdiag[r["split"]]["n"] += 1
            tdiag[r["split"]]["td"].append(int(f[2]))
            if f[1] != "-":
                tdiag[r["split"]]["tpl_shown"] += 1
                tdiag[r["split"]]["tpl_hit"] += int(f[1] == seq_aa0[p])
            if f[4] != "-":
                tdiag[r["split"]]["ec_shown"] += 1
                tdiag[r["split"]]["ec_hit"] += int(f[4] == seq_aa0[p])

        cmp_c = comp_tot.get(lvs[1], collections.Counter()) - comp_rxn.get(
            (lvs[1], rk), collections.Counter())
        tot = sum(cmp_c.values())
        kmed = int(np.median(comp_n.get(lvs[1], [len(core)]))) if comp_n.get(lvs[1]) else len(core)
        comp_hint = " ".join(f"{aa}{max(1, int(round(c / tot * kmed)))}"
                            for aa, c in cmp_c.most_common(6) if tot) or "-"

        seq_aa = seq_aa0
        rec = {
            "schema": "agent_dplmbit.v12_r1_table.v1",
            "parent_id": pid, "round": 1, "round_id": "R1", "split": r["split"],
            "length": L, "band_name": "catalytic_core",
            "evidence_tier": r["evidence_tier"],
            "reaction": r["reaction"],
            "committed": [],
            "candidate_table_columns": "pos tplAA tplD pL ecAA prior",
            "candidate_table": rows,
            "ec_core_composition_hint": comp_hint,
            "table_source": "template_retrieval+pL_EC_prior,leave_one_reaction_out,train_only",
            "template_available": bool(cand),
            "prior_level_used": lvl,
            "target": {"addressing": "absolute_canvas", "band": "catalytic_core",
                       "pairs": [{"position_1based": p, "aa": seq_aa[p]} for p in target]},
            "n_target": len(target), "n_target_band": len(core),
            "pool_coverage_of_band": round(len(target) / len(core), 4),
            "n_candidates": len(rows),
        }
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    w.close()

    print(f"\n  写出 {n:,} 条 | 跳过 {dict(skip)}")
    print(f"  降级阶梯 {dict(tier_used)}")
    print(f"  表行数 中位 {int(np.median(rows_len))} 最大 {max(rows_len)}")
    print(f"\n  {'划分':<14}{'条数':>8}{'表对本档覆盖':>13}{'目标个数中位':>13}{'有模板比例':>11}")
    for s in sorted(cov_by_split):
        c = np.asarray(cov_by_split[s]); t = np.asarray(ntgt_by_split[s])
        print(f"  {s:<14}{len(c):>8,}{c.mean():12.1%}{int(np.median(t)):13d}"
              f"{np.mean(has_tpl[s]):10.1%}")
    print(f"\n  ===== 氨基酸通道体检(只在【目标位置】上量, 那才是有用的那些行) =====")
    print(f"  {'划分':<10}{'目标位置数':>10}{'tplAA 给出率':>13}{'tplAA 命中':>11}"
          f"{'ecAA 命中':>11}{'tplD 中位':>10}")
    for s_ in sorted(tdiag):
        t = tdiag[s_]
        ts = t["tpl_shown"] or 1
        es = t["ec_shown"] or 1
        print(f"  {s_:<10}{t['n']:>10,}{t['tpl_shown']/max(1,t['n']):12.1%}"
              f"{t['tpl_hit']/ts:10.1%}{t['ec_hit']/es:10.1%}"
              f"{int(np.median(t['td'])):9d}")
    print("  参照(全量实测, 在真实核心位置上): 模板对齐残基 25.03% / EC x 位置桶 18.78% / 恒答 12.91%")

    print("\n  ★覆盖率是被记录的任务属性, 不是造假: 表外的位置 agent 推理时本来也选不到。")
    print("  ★参照: R2-R4 的表对本档覆盖 65-67%; 全量实测模板+先验前100行覆盖 55.3%。")


if __name__ == "__main__":
    main()
