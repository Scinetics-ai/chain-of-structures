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
    ap.add_argument("--max-new", type=int, default=1200)
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
    rnd = random.Random(a.seed)
    rows = []
    for r in sorted(per):                 # ★按轮均匀取, 不取前 N 条
        pool = per[r][:]
        rnd.shuffle(pool)
        rows += pool[:a.per_round]
    rows = [x for i, x in enumerate(rows) if i % a.num_shards == a.shard]
    print(f"  本分片 {len(rows)} 条 ({a.shard}/{a.num_shards})", flush=True)

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
            gold = {p["position_1based"]: p["aa"]
                    for p in json.loads(m[2]["content"])["picks"]}
            user = json.loads(m[1]["content"])
            cand = [int(x.split()[0]) for x in (user.get("candidate_table") or [])]
            picks, clean = parse_picks(txt)
            pred = {}
            for p, aa in picks:
                if aa in AA20:
                    pred.setdefault(p, aa)
            inter = set(pred) & set(gold)
            right = sum(1 for p in inter if pred[p] == gold[p])
            s = st[d["round_id"]]
            s["n"] += 1; s["json_ok"] += int(clean)
            s["n_gold"] += len(gold); s["n_pred"] += len(pred)
            s["pos_hit"] += len(inter); s["pair_hit"] += right
            pool = cand if cand else list(range(1, d["length"] + 1))
            g = set(rnd.sample(pool, min(len(gold), len(pool))))
            s["rand_pos_hit"] += len(g & set(gold))
            w.write(json.dumps({"parent_id": d["parent_id"], "round_id": d["round_id"],
                                "n_gold": len(gold), "n_pred": len(pred),
                                "pos_hit": len(inter), "pair_hit": right,
                                "json_ok": clean}, ensure_ascii=False) + "\n")
            w.flush()
            done += 1
        print(f"  {done}/{len(rows)}", flush=True)
    w.close()

    print()
    print(f"  {'轮次':<6}{'n':>5}{'JSON':>7}{'预测/真值':>12}{'位置召回':>9}{'位置精确':>9}"
          f"{'氨基酸@选对':>12}{'整对精确':>9}{'随机召回':>9}")
    for r in sorted(st):
        s = st[r]
        f = lambda x, y: (x / y) if y else float("nan")
        print(f"  {r:<6}{s['n']:>5}{f(s['json_ok'],s['n']):>6.0%}"
              f"{s['n_pred']/max(s['n'],1):>6.1f}/{s['n_gold']/max(s['n'],1):<5.1f}"
              f"{f(s['pos_hit'],s['n_gold']):>9.1%}{f(s['pos_hit'],s['n_pred']):>9.1%}"
              f"{f(s['pair_hit'],s['pos_hit']):>12.1%}"
              f"{f(s['pair_hit'],s['n_gold']):>9.1%}{f(s['rand_pos_hit'],s['n_gold']):>9.1%}")


if __name__ == "__main__":
    main()
