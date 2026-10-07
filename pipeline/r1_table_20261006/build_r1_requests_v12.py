# -*- coding: utf-8 -*-
"""v12 的 R1 教师请求: R1 从【闭卷】改成【开卷】, 给一张 100 行候选表。

★为什么改(2026-10-06 实测):
  R1 没有表 ⇒ 位置只有 **1.68x 随机**(8.2% vs 配对随机 4.88%), 选对的位置上
  **氨基酸只有 4.9%**; 而 R2 有表、同模型同训练, 氨基酸 **54.7%** —— 差 11 倍。
  val 答案 CE: R1 0.8764 vs R2 0.4635, 差 0.41 nat, 正好是"从 331 挑"降到"从 100 挑"的量级。
★表从哪来(全查表, 不需要草稿、不碰天然结构):
  模板检索(同 EC3 最近长度, 位置 4.07x / 氨基酸 25.03%) + p/L x EC 先验(2.75x)。
  闸1 实测真无条件草稿的几何是 **0.95x**(比随机差)且加上去是净负作用, 所以草稿那条路砍了。
★三处必须改的地方:
  1 删掉「Be honest that absolute position is weakly determined」——
    现在有表了, 不该再教它推脱(v11 里 97.8% 的 R1 文本真写了这句)。
  2 加「只能从表里挑」, 和 R2-R4 一致。
  3 ★**明确禁止在 why 里复述列值。** 实测 v11 的 why 字段被填成列值转录:
    `"why":"Near core, L for packing, burial 4, dCore 12.3, plddt 52, draftwin QRR_WGG, seqdistInj 5"`
    后果是 R4 生成到 1600 token(参考只 842)、**JSON 合法率塌到 3%**, 17/17 个失败全是截断。
    这是 v10 那个泄漏病(48.4% 照抄列名)在 v11 的复发。
用法: build_r1_requests_v12.py <obs_r1_v12.jsonl> <out.jsonl> [--limit N]
"""
import argparse
import hashlib
import json

CORE_DESC = "the catalytic residues themselves - the ones that make or break bonds"

SYS = """You are the planning module of an enzyme-design agent. The generator is a masked discrete diffusion model over protein sequence; you tell it which canvas positions to write and which amino acid to write there.

This is ROUND 1 of 4. No structure exists yet - you have the target reaction, the requested chain length, and a shortlist of candidate positions assembled from enzymes of the same EC class. This round commits the CATALYTIC CORE: """ + CORE_DESC + """.

`candidate_table` has one row per candidate position, columns `pos tplAA tplD pL ecAA prior`:
- `pos` is the canvas position (1-based). Pick ONLY from these rows.
- `tplAA` is the residue that the closest same-EC-class enzyme carries at the aligned position, or `-` when the alignment is too loose to be worth anything.
- `tplD` is how far that alignment anchor is, 0 = on it, 9 = far. Trust `tplAA` only when `tplD` is small.
- `pL` is which tenth of the chain the position falls in, 0 = N-terminus.
- `ecAA` is the residue most often catalytic in that tenth of the chain for this EC class.
- `prior` is how strongly the position is favoured overall, 0 = most favoured.
`ec_core_composition_hint` is the residue multiset a catalytic core of this EC class usually has.

You are given the plan this round committed. Write the reasoning that derives it from the reaction chemistry and the table.

Rules:
- `why_by_position` has exactly one entry per position in the plan; the keys are fixed for you. For each, give the reason that position carries the amino acid the plan assigns to it.
- Each `why_by_position` entry must argue chemistry: which group is attacked, what is stabilised, what donates or accepts a proton. Under 120 characters.
- NEVER transcribe column values. Writing `burial 4`, `tplD 2`, `prior 0`, `pL 5`, `plddt 52` or any numeric copy of a column is forbidden. Use the columns to decide, then state the chemical reason.
- Never write that anything "was given", "was provided", or that you are following an instruction. Write the derivation as your own deduction.
- Do NOT name any enzyme and do NOT state what the EC number's family "is" beyond what the supplied EC digits already imply.
- Chemistry: use ONLY the supplied equation and SMILES. Do NOT invent mechanisms or literature.
- Do not claim the plan is uniquely determined; say where another candidate row could serve.
- reasoning: 4-8 complete sentences. uncertainty: 1-2 sentences naming what would change the plan."""


def schema_for(pos):
    return {"type": "json_schema", "json_schema": {
        "name": "agent_round_plan_v12_r1", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["reasoning", "why_by_position", "uncertainty"],
                   "properties": {
                       "reasoning": {"type": "string", "minLength": 1, "maxLength": 2400},
                       "why_by_position": {
                           "type": "object", "additionalProperties": False, "required": pos,
                           "properties": {x: {"type": "string", "minLength": 1,
                                              "maxLength": 120} for x in pos}},
                       "uncertainty": {"type": "string", "minLength": 1, "maxLength": 700}}}}}


MODEL = "Qwen/Qwen3.8-27B"
SEED = 20261006


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obs_jsonl"); ap.add_argument("out_jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    n = 0
    lens = []
    bysplit = {}
    with open(a.out_jsonl, "w") as w:
        for ln in open(a.obs_jsonl):
            if not ln.strip().startswith("{"):
                continue
            r = json.loads(ln)
            assert r["round"] == 1 and r["band_name"] == "catalytic_core"
            pairs = r["target"]["pairs"]
            pos = [str(p["position_1based"]) for p in pairs]
            assert len(set(pos)) == len(pos), f"{r['parent_id']} 位置重复"
            tbl_pos = {x.split()[0] for x in r["candidate_table"]}
            assert set(pos) <= tbl_pos, f"★{r['parent_id']} 目标不在表里 —— 造表那步漏了过滤"
            obs = {"round_id": "R1", "round_of": 4, "length": r["length"],
                   "this_round_commits": "catalytic_core",
                   "reaction": r["reaction"],
                   "committed": [],
                   "candidate_table_columns": r["candidate_table_columns"],
                   "candidate_table": r["candidate_table"],
                   "ec_core_composition_hint": r["ec_core_composition_hint"],
                   # ★committed_plan 只给教师; assemble 时会从学生的 user 消息里删掉
                   "committed_plan": [{"position_1based": p["position_1based"], "aa": p["aa"]}
                                      for p in pairs]}
            # ★payload 的形状必须和 dispatch_v10.py 的契约逐字一致:
            #   顶层要 request_id / parent_id / round_id / split, body 放在 request_payload。
            #   我第一版写成 custom_id + 顶层 messages, 冒烟直接 KeyError: 'request_id'。
            payload = {"model": MODEL, "n": 1, "temperature": 0, "top_p": 1, "seed": SEED,
                       "max_tokens": 4096, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "response_format": schema_for(pos),
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": canon(obs)}]}
            rec = {"schema": "agent_dplmbit.v12_r1_table.v1",
                   "parent_id": r["parent_id"], "round_id": "R1", "band": "catalytic_core",
                   "split": r["split"], "length": r["length"],
                   "evidence_tier": r.get("evidence_tier"),
                   "n_plan": len(pairs), "n_candidates": r["n_candidates"],
                   "pool_coverage_of_band": r["pool_coverage_of_band"],
                   "template_available": r["template_available"],
                   "request_id": hashlib.sha256(
                       canon({"s": "v12r1", "p": r["parent_id"], "pl": payload}).encode()).hexdigest(),
                   "request_payload": payload,
                   "teacher_saw_reference_answer": True,
                   "prompt_chars": len(SYS) + len(payload["messages"][1]["content"])}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            lens.append(rec["prompt_chars"])
            bysplit[r["split"]] = bysplit.get(r["split"], 0) + 1
            n += 1
            if a.limit and n >= a.limit:
                break
    import statistics
    print(f"  写出 {n:,} 条 | 划分 {dict(sorted(bysplit.items()))}")
    print(f"  提示词字符数 中位 {int(statistics.median(lens)):,} / p95 "
          f"{int(statistics.quantiles(lens, n=20)[18]):,} / 最大 {max(lens):,}")
    print(f"  (参照: v11 的 R2-R4 提示词约 4,000-5,000 字符; 上限 7,965 token)")
    print(f"  ★系统提示词 {len(SYS):,} 字符; 已删「位置弱定」那句, 已加「只能从表里挑」"
          f"和「禁止复述列值」")


if __name__ == "__main__":
    main()
