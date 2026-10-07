# -*- coding: utf-8 -*-
"""离线评测: 拿【正在训练的模型】(Qwen3.5-9B 基座 + 我们这轮 SFT 的 LoRA)真自回归生成,
算 (位置, 氨基酸) 四个能写进论文的数。

★为什么不能用训练日志里那个 71.5%:
  那是 teacher forcing 下【答案 token】的 top-1 准确率, 三层弱点 ——
  1 每步看到的前缀都是真值, 推理时要自回归走完, 错误会累积
  2 按 token 算: BPE 把位置号切成一位数字一个 token, "166" 是 '1','6','6'。
    模型吐 168 拿到 3/4=75% 的 token 分, 但这个位置整个是错的
  3 按 token 数加权, 位置多的样本权重大
  ⇒ 它只能看"答案通道有没有在学", 不能当结果。

★四个数(都按【集合】算, 与生成多少个无关, 所以不奖励"多说"):
  位置召回   |预测∩真值| / |真值|
  位置精确   |预测∩真值| / |预测|       ← 必须和召回一起报
  氨基酸@选对 位置选对的那些点上氨基酸对的比例
  整对精确   (位置,氨基酸)全对 / |真值|  ← 和论文对口的那个
  只报召回会被"多说"刷高(实测 exact 0.110->0.165 而 EC3 掉 9.3 分)。

★两条必要参照:
  随机位置召回 —— R2-R4 的候选表只有 100 行, 随机就有约 10%, 没这条读不出高低
  JSON 合法率  —— 自回归可能吐不完整 JSON, 不报就分不清"答错"和"解析失败"

★实现上的两个坑:
  1 批量生成必须【左 padding】, 右 padding 会让短样本在生成时读到 pad
  2 取样必须【按轮均匀】, 不能取前 N 条 —— 合并文件里 R1 全在最前面(今天踩过)
用法: eval_pairs.py --base B --adapter A --val V [--per-round 32] [--batch 8]
                    [--shard i --num-shards n]
"""
import argparse
import collections
import json
import random
import re

import torch

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def parse_picks(text):
    """抽 picks。模型可能吐不完整 JSON, 两条路都试, 并报是否干净解析。"""
    try:
        o = json.loads(text)
        if isinstance(o, dict) and isinstance(o.get("picks"), list):
            out = []
            for p in o["picks"]:
                if isinstance(p, dict) and "position_1based" in p and "aa" in p:
                    try:
                        out.append((int(p["position_1based"]), str(p["aa"]).strip()))
                    except (ValueError, TypeError):
                        pass
            return out, True
    except Exception:
        pass
    out = [(int(m.group(1)), m.group(2)) for m in
           re.finditer(r'"position_1based"\s*:\s*(\d+)\s*,\s*"aa"\s*:\s*"([A-Z])"', text)]
    return out, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--per-round", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    # ★1200 不够: v11 实测模型在后面几轮生成到 1600+(参考只 842), 32 条里 17 条被截断,
    #   JSON 合法率因此假塌到 3%。截断是症状, 根因是模型【不收尾】⇒ 还要看 hit_cap 比例。
    # ★v12 实测(4,637 条已组装 R1): assistant 中位 877 token / p99 1,902 / **最大 3,031**;
    #   2400 只覆盖 99.76%(11 条会被切)。生成遇 EOS 就停 ⇒ 抬高上限对短输出零代价,
    #   所以取 **3200**(提示词最大约 2,131 token, 合计 5,331, 远在上下文内)。
    ap.add_argument("--max-new", type=int, default=3200)
    ap.add_argument("--group-pids", default="",
                    help="name=pids.txt 逗号分隔; 按组分开报(v12: test_overlap / test_strict 必须分开)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="pairs_eval.jsonl")
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.base)
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"              # ★批量生成必须左 padding
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    print("  基座 + LoRA 已加载", flush=True)

    per = collections.defaultdict(list)
    for ln in open(a.val):
        d = json.loads(ln)
        per[d["round_id"]].append(d)
    groups = {}
    for spec in [x for x in a.group_pids.split(",") if x.strip()]:
        nm, fp = spec.split("=", 1)
        groups[nm] = {x.strip() for x in open(fp)} - {""}
    if groups:
        print("  分组: " + " ".join(f"{k}={len(v):,}" for k, v in groups.items()), flush=True)
    rnd = random.Random(a.seed)
    rows = []
    # ★抽样必须【按轮 x 按组】分层, 不能只按轮:
    #   v12 的留出评测集全是 R1(5,090 条), 其中 test_strict 只有 150 条(2.9%)。
    #   只按轮抽 64 条 ⇒ 按比例只有约 2 条落在 strict, 分组统计毫无意义。
    #   分层后 strict 能全取(或取到上限), overlap 另算。
    for r in sorted(per):
        if groups:
            bycg = collections.defaultdict(list)
            for d in per[r]:
                g = next((k for k, v in groups.items() if d["parent_id"] in v), "(未分组)")
                bycg[g].append(d)
            for g in sorted(bycg):
                pool = bycg[g][:]
                rnd.shuffle(pool)
                rows += pool[:a.per_round]
        else:
            pool = per[r][:]
            rnd.shuffle(pool)
            rows += pool[:a.per_round]
    rows = [x for i, x in enumerate(rows) if i % a.num_shards == a.shard]
    _d = collections.Counter()   # ★模块级已 import collections, 不再局部 import
    for d in rows:
        g = next((k for k, v in groups.items() if d["parent_id"] in v), "(未分组)") if groups else "-"
        _d[(d.get("round_id"), g)] += 1
    print(f"  本分片 {len(rows)} 条 ({a.shard}/{a.num_shards}) | 分层 {dict(_d)}", flush=True)

    # 按提示词 token 数排序成批, 同批长度相近, 少浪费 padding
    plen = {}
    prompts = {}
    for i, d in enumerate(rows):
        p = tk.apply_chat_template([d["messages"][0], d["messages"][1]],
                                   tokenize=False, add_generation_prompt=True)
        prompts[i] = p
        plen[i] = len(tk(p, add_special_tokens=False)["input_ids"])
    order = sorted(range(len(rows)), key=lambda j: plen[j])

    st = collections.defaultdict(collections.Counter)
    w = open(a.out, "w")
    done = 0
    for b0 in range(0, len(order), a.batch):
        sel = order[b0:b0 + a.batch]
        enc = tk([prompts[j] for j in sel], return_tensors="pt", padding=True,
                 add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tk.pad_token_id)
        txts = tk.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, txt in zip(sel, txts):
            d = rows[j]
            m = d["messages"]
            # ★grp 必须在这里算: 下面的分层(STRAT)、照抄诊断(COPY)、按轮(sg) 三处都要用它。
            #   原先它写在循环末尾的 w.write 之前 ⇒ UnboundLocalError, 冒烟测试当场抓到。
            grp = [k for k, v in groups.items() if d["parent_id"] in v] or ["(未分组)"]
            gold = {p["position_1based"]: p["aa"]
                    for p in json.loads(m[2]["content"])["picks"]}
            user = json.loads(m[1]["content"])
            cand = [int(x.split()[0]) for x in (user.get("candidate_table") or [])]
            picks, clean = parse_picks(txt)
            pred = {}
            for p, aa in picks:
                if aa in AA20:
                    pred.setdefault(p, aa)
            # ★量"挑到表外": 提示词明确说 Pick ONLY from these rows。违反了既是失败模式,
            #   也让和查表基线的对比不公平(查表根本不可能挑到表外)。
            _cs = set(cand)
            _oob = [p for p in pred if _cs and p not in _cs]
            inter = set(pred) & set(gold)
            right = sum(1 for p in inter if pred[p] == gold[p])
            # ★按 tplD 分层(feedback_stratify_on_external_reference: tplD 是输入特征, 合法分层轴)。
            #   实测分工: tplD 0-1 占 47.7% 的 gold 位置, 那里查表 tplAA 已对 57.5%、
            #   73.3% 被手写规则排进前 2 成 ⇒ agent 最多持平;
            #   tplD>=2 占 52.3%, tplAA 只对 19.3%/0% ⇒ **那才是 agent 的战场**。
            #   不分层就会把"表给的简单子集"记成方法的本事。
            rowmap = {}
            for _r in (user.get("candidate_table") or []):
                _t = _r.split()
                rowmap[int(_t[0])] = _t
            def _band(p):
                t = rowmap.get(p)
                if not t:
                    return "表外"
                dd = int(t[2])
                return "tplD0-1" if dd <= 1 else ("tplD2-4" if dd <= 4 else "tplD5-9")
            for _p, _aa in gold.items():
                _b = _band(_p)
                _s = st["STRAT|%s|%s" % (grp[0], _b)]
                _s["n_gold"] += 1
                _s["n"] += 1
                if _p in pred:
                    _s["pos_hit"] += 1
                    _s["pair_hit"] += int(pred[_p] == _aa)
            # ★最关键的诊断: 模型是在【照抄 tplAA】还是在【加化学】?
            #   查表-only 基线已经拿到 test_overlap 整对 23.4%, 所以"准确率高"本身不说明问题。
            #   对每个落在 gold 上的预测, 分两类看:
            #     跟随模板(pred == tplAA)  —— 对了也只是复述查表
            #     偏离模板(pred != tplAA)  —— 偏离后【还对】才是 agent 真加的东西
            #   若"偏离且对"的比例低于模板自己在那些位置上的命中率, 说明偏离是净伤害。
            for _p, _aa in gold.items():
                if _p not in pred:
                    continue
                _t = rowmap.get(_p)
                if not _t:
                    continue
                _tpl = _t[1]
                _c = st["COPY|%s|%s" % (grp[0], _band(_p))]
                _c["n"] += 1
                if _tpl == "-":
                    _c["无模板"] += 1
                    _c["无模板对"] += int(pred[_p] == _aa)
                elif pred[_p] == _tpl:
                    _c["跟随"] += 1
                    _c["跟随对"] += int(pred[_p] == _aa)
                else:
                    _c["偏离"] += 1
                    _c["偏离对"] += int(pred[_p] == _aa)
                    _c["偏离时模板本会对"] += int(_tpl == _aa)
            s = st[d["round_id"]]
            sg = st["%s|%s" % (grp[0], d["round_id"])]
            for kk, vv in (("n", 1), ("json_ok", int(clean)), ("n_gold", len(gold)),
                           ("n_pred", len(pred)), ("pos_hit", len(inter)), ("pair_hit", right),
                           ("n_oob", len(_oob))):
                sg[kk] += vv
            s["n_oob"] += len(_oob)
            s["n"] += 1; s["json_ok"] += int(clean)
            s["n_gold"] += len(gold); s["n_pred"] += len(pred)
            s["pos_hit"] += len(inter); s["pair_hit"] += right
            pool = cand if cand else list(range(1, d["length"] + 1))
            g = set(rnd.sample(pool, min(len(gold), len(pool))))
            s["rand_pos_hit"] += len(g & set(gold))
            # ★v12 多存: 原版只存计数, 事后想复算"输出残基组成"却没数据可用。
            #   另外 gold 现在是【本档 ∩ 表】, 所以要同时记 n_band 才能分清
            #   "表里能拿到的" 和 "整档" 两个分母。
            w.write(json.dumps({"parent_id": d["parent_id"], "round_id": d["round_id"],
                                "group": grp[0], "n_cand": len(cand),
                                "n_oob": len(_oob), "oob_pos": _oob[:10],
                                "gen_tokens": int(len(tk(txt, add_special_tokens=False)["input_ids"])),
                                "hit_cap": bool(len(tk(txt, add_special_tokens=False)["input_ids"]) >= a.max_new - 2),
                                "pred_pairs": [[p, aa] for p, aa in sorted(pred.items())],
                                "gold_pairs": [[p, aa] for p, aa in sorted(gold.items())],
                                "n_gold": len(gold), "n_pred": len(pred),
                                "pos_hit": len(inter), "pair_hit": right,
                                "json_ok": clean}, ensure_ascii=False) + "\n")
            w.flush()
            done += 1
        print(f"  {done}/{len(rows)}", flush=True)
    w.close()

    print()
    # ★列序必须和 line() 的输出逐列对上 —— 加了表外率就要在这里同步加, 否则读数会错位
    #   (feedback_grep_c_exit_code_shifts_monitor_fields 那个形状)
    print(f"  {'轮次':<22}{'n':>5}{'JSON':>7}{'预测/真值':>12}{'位置召回':>9}{'位置精确':>9}"
          f"{'氨基酸@选对':>12}{'整对精确':>9}{'表外率':>8}{'随机召回':>9}")
    f = lambda x, y: (x / y) if y else float("nan")

    def line(nm, s, with_rand):
        t = (f"  {nm:<22}{s['n']:>5}{f(s['json_ok'],s['n']):>6.0%}"
             f"{s['n_pred']/max(s['n'],1):>6.1f}/{s['n_gold']/max(s['n'],1):<5.1f}"
             f"{f(s['pos_hit'],s['n_gold']):>9.1%}{f(s['pos_hit'],s['n_pred']):>9.1%}"
             f"{f(s['pair_hit'],s['pos_hit']):>12.1%}{f(s['pair_hit'],s['n_gold']):>9.1%}"
             f"{f(s['n_oob'],s['n_pred']):>8.1%}")
        # ★随机基线只在不分组的那一套里统计过, 分组行不要打 nan 冒充数字
        return t + (f"{f(s['rand_pos_hit'],s['n_gold']):>9.1%}" if with_rand else "        -")

    print("  ===== 按轮(全部样本) =====")
    for r in sorted(k for k in st if "|" not in k):
        print(line(r, st[r], True))
    sk = sorted(k for k in st if k.startswith("STRAT|"))
    if sk:
        print("  ===== 按 tplD 分层(只看 gold 位置; 查表基线见 lookup_only_baseline.py) =====")
        print(f"  {'组/层':<30}{'gold数':>8}{'位置召回':>10}{'整对':>9}")
        for k in sk:
            c = st[k]
            print(f"  {k[6:]:<30}{c['n_gold']:>8}{f(c['pos_hit'],c['n_gold']):>9.1%}"
                  f"{f(c['pair_hit'],c['n_gold']):>9.1%}")
        print("  ★参照(查表-only 最强手写规则 DP, test_overlap): tplD0-1 那层查表已很强,")
        print("    tplD>=2 那层 tplAA 对 19.3%/0% ⇒ 看 agent 在后两层有没有拉开。")
    ck = sorted(k for k in st if k.startswith("COPY|"))
    if ck:
        print("  ===== 照抄还是加化学(只看预测落在 gold 上的那些位置) =====")
        print(f"  {'组/层':<30}{'n':>6}{'跟随模板':>9}{'跟随对':>8}"
              f"{'偏离':>7}{'偏离对':>8}{'偏离时模板本会对':>17}{'无模板对':>9}")
        for k in ck:
            c = st[k]
            print(f"  {k[5:]:<30}{c['n']:>6}{f(c['跟随'],c['n']):>8.0%}"
                  f"{f(c['跟随对'],c['跟随']):>8.0%}{f(c['偏离'],c['n']):>7.0%}"
                  f"{f(c['偏离对'],c['偏离']):>8.0%}"
                  f"{f(c['偏离时模板本会对'],c['偏离']):>16.0%}"
                  f"{f(c['无模板对'],c['无模板']):>9.0%}")
        print("  ★读法: '偏离对' > '偏离时模板本会对' ⇒ 偏离是净收益, agent 在加化学;")
        print("    反之 ⇒ 偏离是净伤害, 应该让它更信模板。'无模板对' 是纯化学推理那一格,")
        print("    查表在那里给不出任何东西(占 gold 位置的 28.8%)。")
    gk = sorted(k for k in st if "|" in k and not k.startswith("STRAT|")
                and not k.startswith("COPY|"))
    if gk:
        print("  ===== 按组 x 轮(v12: test_overlap 与 test_strict 必须分开看) =====")
        for r in gk:
            print(line(r, st[r], False))
        # 组内合计
        import collections as _c
        agg = _c.defaultdict(_c.Counter)
        for r in gk:
            agg[r.split("|")[0]].update(st[r])
        print("  ----- 各组合计 -----")
        for g in sorted(agg):
            print(line(g + "|ALL", agg[g], False))


if __name__ == "__main__":
    main()
