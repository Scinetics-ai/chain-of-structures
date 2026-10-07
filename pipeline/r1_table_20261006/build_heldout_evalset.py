# -*- coding: utf-8 -*-
"""把留出集的观测做成 SFT 格式的评测集(不花教师)。

★为什么不用教师: 评测只需要【金标 picks】来对分, 不需要推理文本。
  eval_pairs 只从 messages[2] 里取 picks。

★系统提示词必须和训练时【逐字节一致】, 否则评的是另一个任务。
  所以不重新推导 —— 直接从一条真实的【已组装训练记录】里把 system 原样取出来,
  再硬断言它不含教师那套措辞(给了你答案 / why_by_position)。
  自己重写一遍 system 是 feedback_same_class_arms_need_one_policy 那类坑:
  两处措辞分两次写就会不一致, 不报错, 只让评测数字莫名偏低。

★user 消息的构造也照抄组装器: obs 去掉 committed_plan, 然后 sort_keys 序列化。
用法: build_heldout_evalset.py <obs(含heldout).jsonl> <一条已组装的训练记录.jsonl> <out.jsonl>
"""
import json
import sys

obs_f, ref_f, out_f = sys.argv[1:4]

SYS = None
for ln in open(ref_f, errors="ignore"):
    if ln.lstrip().startswith("{"):
        d = json.loads(ln)
        if d.get("round_id") == "R1":
            SYS = d["messages"][0]["content"]
            break
assert SYS, "★参考文件里没有 R1 记录"
assert "why_by_position" not in SYS, "★取到的 system 还是教师那版"
assert "committed plan" not in SYS.lower(), "★取到的 system 还带'给了你答案'"
assert "Pick ONLY from these rows" in SYS, "★system 里没有'只能从表里挑', 版本不对"
print(f"  system 原样取自已组装记录, {len(SYS):,} 字符, 三条断言过")

n = 0
by = {}
w = open(out_f, "w")
for ln in open(obs_f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    r = json.loads(ln)
    if r["split"] != "heldout":
        continue
    stu = {k: v for k, v in r.items()
           if k in ("round_id", "round_of", "length", "this_round_commits", "reaction",
                    "committed", "candidate_table_columns", "candidate_table",
                    "ec_core_composition_hint")}
    stu["round_id"] = "R1"
    stu["round_of"] = 4
    stu["this_round_commits"] = "catalytic_core"
    picks = [{"position_1based": p["position_1based"], "aa": p["aa"], "why": ""}
             for p in r["target"]["pairs"]]
    asst = json.dumps({"reasoning": "", "picks": picks, "uncertainty": ""},
                      separators=(",", ":"), ensure_ascii=False)
    assert asst.startswith('{"reasoning"')
    rec = {"schema": "agent_dplmbit.v12_eval.v1", "parent_id": r["parent_id"],
           "round_id": "R1", "band": "catalytic_core", "split": "heldout",
           "length": r["length"], "n_picks": len(picks),
           "n_candidates": r["n_candidates"],
           "pool_coverage_of_band": r["pool_coverage_of_band"],
           "n_target_band": r["n_target_band"],
           "messages": [{"role": "system", "content": SYS},
                        {"role": "user", "content": json.dumps(
                            stu, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
                        {"role": "assistant", "content": asst}]}
    assert "committed_plan" not in rec["messages"][1]["content"], "★学生看见了答案"
    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    by[len(picks)] = by.get(len(picks), 0) + 1
    n += 1
w.close()
import statistics as st
ks = sorted(by)
tot = sum(by.values())
print(f"  写出 {n:,} 条 | picks 中位 "
      f"{st.median([k for k in ks for _ in range(by[k])])} | 最小 {ks[0]} 最大 {ks[-1]}")
print(f"  ★user 里无 committed_plan; assistant 以 reasoning 开头(两条都已硬断言)")
