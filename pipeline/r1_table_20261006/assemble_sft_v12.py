# -*- coding: utf-8 -*-
"""v11 组装 SFT。教师拿着答案写了推导, 这里把答案从**学生输入**里删掉。

★★v12 = v11 的组装器 + 三处改动, 因为 R1 这一轮现在**有候选表**了:
  1 **「picks 必须在表里」从 R2-R4 扩到 R1** —— 以前 R1 没表所以跳过这条。
  2 **「why 里有表外编号」从 R2-R4 扩到 R1** —— 同理。
  3 **★新闸: 禁止复述列值。** v11 实测这条病复发了: why 字段被填成列值转录
    `"why":"Near core, L for packing, burial 4, dCore 12.3, plddt 52, draftwin QRR_WGG, seqdistInj 5"`
    后果是 R4 生成到 1600 token(参考只 842)、**JSON 合法率塌到 3%**, 17/17 个失败全是截断。
    提示词里加了禁令还不够 —— v10 的教训就是光靠提示词拦不住(48.4% 照抄列名), 必须机械查。
★v11 比 v10 多【四道机械闸】。理由是 v8 的审计教训(project_v8_teacher_audit_20261003):
  那次生产批 21.1% 违规、最大一项是编造酶名 12.9%, 而这些**全是机械可查的, 却没进闸**。
  这次先用 600 条小批实测了各项违规率, 再把每一项都写成闸:
    闸6 EC 类矛盾 3.5%      —— 说"hydrolysis"而 EC 第一位是 4(lyase) 之类
    闸7 配体距离   1.2%      —— 我们没给配体, 它自己编了一个距离
    闸8 跨位点指认错残基 1.0% —— 写 "H262" 而 262 处承诺的是别的残基
    闸9 表外位置   —— why 里显式写 "position N" 而 N 不是任何合法位置
                      (原版抓裸数字, 实测 14/600 全是假阳: Mg2+ 的 2、被 140 字截断的 D482)
  temperature=0 ⇒ 原样重跑必然得到同样的错(project_v8_teacher_audit_20261003),
  所以闸只能是【丢弃】, 不是重试。合计预计丢 5-6%。
★另修一处 v10 的提示词/目标不一致: 教师那版说 "`why_by_position` ... 键是给你定好的",
  但学生的目标格式是 `picks` 数组、位置要自己选。学生那版必须改写成讲 `picks`,
  否则是在教它一个它根本产不出的格式。

★两处绝不能错(都是实测踩过的):
 1. **学生看不到 committed_plan** —— 教师那份是为了写推导才给的。不删, 训出来的模型
    只会在推理时等别人把答案递过来。
 2. **assistant 的 JSON 绝不能 sort_keys** —— 字母序会排成 picks 在 reasoning 前,
    等于先给答案再补解释, CoT 形同虚设(v9 实测 3764/3764 条全被排反)。
    必须 reasoning -> picks -> uncertainty, 并加硬断言。

五道闸(逐条, 不过的落 reject_ledger):
  JSON 可解析且 finish_reason==stop / picks 与 plan 完全一致 / 不出现"答案是给的" /
  不编造酶名 / 第2-4轮的 picks 必须全在候选表里(第1轮没有候选表, 不查这条)
"""
import argparse, json, re, statistics
from collections import Counter, defaultdict

# ★收窄过一次(2026-10-05): 原版把 1,657 条判成泄露, 但触发词 97% 是
#   'plan commits'(674) / 'committed plan'(523) / 'in the plan'(459) —— 这三个**是我自己给的词**:
#   system 提示词写的就是 "This is ROUND R, which **commits** the BAND", 观测字段叫 `committed`。
#   教师只是在用我的词汇描述任务, 并不暴露"答案是被给的"。真正的泄露只有约 25 条。
#   现在只拦**明确说出"这答案是别人给我的"**那几种说法。
LEAK = re.compile(r"\b(was given|were given|was provided|were provided|as instructed|"
                  r"per the instruction|the plan provided|provided plan|given plan|"
                  r"you (?:were |are )?given|i (?:was |am )?given|"
                  r"not in (?:the )?(?:candidate )?table|table omission|"
                  r"omitted from (?:the )?table)\b", re.I)
ASE = re.compile(r"\b([A-Za-z]{4,}(?:ase|ases))\b")
# ★白名单放宽过一次(2026-10-05): 原版把 2,482 条判成"编造酶名", 但看了实际词之后发现两类都是误判:
#   1. **nucleobase(669 条)** —— 核碱基, 正当化学术语, 只是碰巧 nucleob+ase 结尾
#   2. **EC 分类的通名**(phosphatase/transferase/hydrolase/isomerase…约 800 条) ——
#      EC 号就在输入里, 说"这是个转移酶"是**读出来的不是编的**。
#   真正该拦的是**编造具体酶名**(烟测抓到过"把 EC 2.5.1.7 说成 phosphatidylcholine phosphatase"),
#   那种的区别词(phosphatidylcholine)不以 -ase 结尾, 本来就不靠这条正则拦。
NOT_ENZYME = {
    # 普通英文词, 碰巧 -ase 结尾
    "release", "releases", "increase", "increases", "decrease", "decreases", "database",
    "databases", "phase", "phases", "please", "disease", "diseases", "purchase", "base",
    "bases", "case", "cases", "showcase", "chase", "erase",
    # 化学术语
    "nucleobase", "nucleobases", "diphosphatase", "pyrophosphatase",
    # EC 分类通名(可从输入里的 EC 号直接读出)
    "hydrolase", "hydrolases", "transferase", "transferases", "oxidoreductase", "oxidoreductases",
    "isomerase", "isomerases", "ligase", "ligases", "lyase", "lyases", "racemase", "racemases",
    "dehydrogenase", "dehydrogenases", "phosphatase", "phosphatases", "phosphotransferase",
    "phosphotransferases", "kinase", "kinases", "synthase", "synthases", "synthetase",
    "synthetases", "reductase", "reductases", "oxidase", "oxidases", "esterase", "esterases",
    "phosphoesterase", "phosphoesterases", "glycosidase", "glycosidases", "peptidase",
    "peptidases", "protease", "proteases", "decarboxylase", "decarboxylases", "carboxylase",
    "carboxylases", "mutase", "mutases", "epimerase", "epimerases", "aminotransferase",
    "aminotransferases", "acyltransferase", "acyltransferases", "methyltransferase",
    "methyltransferases", "glycosyltransferase", "glycosyltransferases", "nuclease", "nucleases",
    "polymerase", "polymerases", "phospholipase", "phospholipases", "glutaminase", "glutaminases",
    "amidase", "amidases", "deaminase", "deaminases", "hydratase", "hydratases", "dehydratase",
    "dehydratases", "aldolase", "aldolases", "cyclase", "cyclases", "transaminase", "transaminases",
    "atpase", "gtpase", "permease", "permeases", "lipase", "lipases", "amylase", "amylases",
    # ★2026-10-06 小批实测补: 这 10 个把 10/600 判成"编造酶名", 但它们全是 EC 通名
    #   (formyltransferase / nucleotidase …), 从 EC 号就能读出来。真正该拦的"编造具体酶名"
    #   区别词(如 phosphatidylcholine)本来不以 -ase 结尾, 不靠这条正则拦。
    #   其中确实有一条是真错 —— 把 EC 4.1.2.43 说成 formyltransferase ——
    #   但那是【类别说反了】, 归下面的 EC 对账闸管, 不归这条。
    "formyltransferase", "formyltransferases", "prenyltransferase", "prenyltransferases",
    "acetyltransferase", "acetyltransferases", "nucleotidase", "nucleotidases",
    "phosphorylase", "phosphorylases", "thiolase", "thiolases", "metallophosphatase",
    "metallophosphatases", "sulfotransferase", "sulfotransferases", "monooxygenase",
    "monooxygenases", "dioxygenase", "dioxygenases", "peroxidase", "peroxidases",
    "tautomerase", "tautomerases", "thioesterase", "thioesterases", "endonuclease",
    "endonucleases", "exonuclease", "exonucleases", "pyrophosphorylase", "pyrophosphorylases",
}


# ★闸6: EC 第一位 ⇒ 绝不该出现的反应类词。只收【确定矛盾】的, 不收"没对上"的 ——
#   "没对上EC类词" 实测 19%, 大多是写得抽象而不是写错, 拦了纯损失数据。
EC_CONTRA = {
    1: r"\bhydrolysis\b|\bhydrolase\b|\bhydrolytic\b",
    2: r"\bhydrolase\b",
    4: r"\bhydrolysis\b|\bhydrolase\b|\bhydrolytic\b",
    5: r"\bhydrolysis\b|\bhydrolase\b|\bdecarboxylat",
}
EC_CONTRA = {k: re.compile(v, re.I) for k, v in EC_CONTRA.items()}
# ★EC 大类的通名表。闸只在【断言这个反应是什么】时才判 —— 顺带提到别的类
#   ("unlike a hydrolase")不算错, 所以必须配 ASSERT 模式一起用。
EC_CLASS_WORDS = {
    1: ("oxidoreductase", "dehydrogenase", "reductase", "oxidase", "peroxidase",
        "hydroxylase", "monooxygenase", "dioxygenase"),
    2: ("transferase", "kinase", "phosphorylase", "acetyltransferase", "methyltransferase",
        "glycosyltransferase", "aminotransferase", "transaminase", "formyltransferase",
        "prenyltransferase", "sulfotransferase", "acyltransferase", "phosphotransferase"),
    3: ("hydrolase", "peptidase", "protease", "esterase", "lipase", "nuclease", "phosphatase",
        "nucleotidase", "amidase", "glycosidase", "deaminase", "thioesterase",
        "metallophosphatase", "endonuclease", "exonuclease"),
    4: ("lyase", "aldolase", "decarboxylase", "dehydratase", "hydratase", "cyclase"),
    5: ("isomerase", "mutase", "epimerase", "racemase", "tautomerase"),
    6: ("ligase", "synthetase", "carboxylase"),
    7: ("translocase", "atpase", "pump"),
}
WORD2EC = {w: k for k, ws in EC_CLASS_WORDS.items() for w in ws}
# "The reaction is a formyltransferase" / "this is an aldolase" / "acts as a hydrolase" / "an X that"
ASSERT_CLASS = re.compile(
    r"(?:reaction is|this is|it is|acts as|functions as|classified as"
    r"|catalysed by|catalyzed by)[ ]+(?:a|an|the)?[ ]*(?:[a-z-]+[ -])?"
    r"([a-z]{4,}(?:ase))(?![a-z])", re.I)
# ★闸7: 我们没给配体距离(v11 的整个重做就是为了删掉那一列), 它自己编一个就是幻觉。
#   注意不能只拦 "ligand/substrate" 这个词 —— 实测 89.8% 的文本里有"substrate",
#   那是正当的化学叙述("stabilize the zwitterionic substrate")。只拦【距离】。
LIG_DIST = re.compile(r"(ligand|substrate)[^.]{0,40}"
                      r"(distance|\u00c5|angstrom|\b\d+(?:\.\d+)?\s*A\b)", re.I)
PTR = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})\b")
# ★闸9 收窄过一次(2026-10-06 小批实测): 原版抓裸数字, 14/600 全是假阳 ——
#   "Mg2+" 的 2、"2'-OH" 的 2、plddt 写成 52 而行里是 51、以及被 140 字上限
#   截断的 "D482"/"C48…"。而 v11 的位置是 schema 的 required 锁死的, 根本不可能
#   输出表外位置 ⇒ 这道闸真正要管的只剩【散文里提到一个不存在的位置】。
#   所以只查显式的 "position 123" / "pos 123" / "residue 123" 形式;
#   "H262" 那种字母+数字的指认归闸8 查残基对不对。
NUM = re.compile(r"\b(?:position|pos\.?|residue|res\.?)\s*#?\s*(\d{1,4})\b", re.I)
# ★闸10(v12 新增): 禁止把候选表的列值抄进 why。
#   v12 的列是 pos tplAA tplD pL ecAA prior; 旧列名也一并拦(模型会迁移旧习惯)。
#   只拦【列名 + 紧跟数值/残基】这种转录形态, 不拦正常提到化学的句子。
RECITE = re.compile(
    r"\b(?:tplAA|tplD|ecAA|prior|pL|burial|plddt|dCore|dCen|draftwin|seqdistInj)\b"
    r"(?:\s*(?:is|=|:)?\s*[-]?\d|\s+[ACDEFGHIKLMNPQRSTVWY]\b)", re.I)
# 连列名本身都不该出现在 why 里(教师被明确禁止了), 单独数一次用于审计
COLNAME = re.compile(r"\b(?:tplAA|tplD|ecAA|draftwin|seqdistInj|dCore|dCen|plddt)\b", re.I)


def row_own_numbers(tok):
    """该行自己的数值(含四舍五入形式)。★第一版审计器没排除它们, 把 plddt 的 51
    当成"表外位置", 报出 59.78% 的假阳 —— 先修仪表再读数。"""
    own = set()
    for x in tok[1:]:
        for num in re.findall(r"\d+", x):
            own.add(int(num))
        try:
            own.add(int(round(float(x))))
        except ValueError:
            pass
    return own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests_jsonl"); ap.add_argument("results_jsonl")
    ap.add_argument("out_sft"); ap.add_argument("out_reject")
    # ★证据档位: 第一批 20,507 个 parent 有 ECO 实验证据的催化位点标注;
    #   第二批 29,399 个没有(位点是推断的), 实测可预测性 0.2664 vs 0.3158(低 15.6%)。
    #   合并后训练要能分开加权或过滤, 所以每条都带这个字段, 不靠文件名区分。
    ap.add_argument("--evidence-tier", choices=["eco", "predicted"], default="eco",
                    help="没有 --tier-map 时用的统一档位")
    # ★必须按 parent 取档位, 不能按文件: v11 的分片是把两批【交错】后再切的
    #   (为了让每台机器都拿到两批、三轮的混合), 所以每个分片里 eco 和 predicted 都有。
    #   这个字段不能错 —— 实测 eco 的可预测性 0.3158 vs predicted 0.2664, 差 15.6%,
    #   训练要按它加权或过滤。
    ap.add_argument("--tier-map", default="", help="parent_id -> 'eco'/'predicted' 的 json")
    a = ap.parse_args()
    tier_map = json.load(open(a.tier_map)) if a.tier_map else {}
    if tier_map:
        print(f"档位表 {len(tier_map):,} 个 parent", flush=True)
    miss_tier = Counter()
    req = {}
    for ln in open(a.requests_jsonl):
        if ln.strip().startswith("{"):
            q = json.loads(ln); req[q["request_id"]] = q
    print(f"请求 {len(req):,} 条", flush=True)

    st = Counter(); per = Counter(); seen = set(); sec = []
    nw = open(a.out_sft, "w"); rj = open(a.out_reject, "w")
    for ln in open(a.results_jsonl, errors="ignore"):
        if not ln.strip().startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except Exception:
            st["结果行坏"] += 1; continue
        rid = r.get("request_id")
        if rid in seen:
            st["重复结果"] += 1; continue
        seen.add(rid)
        q = req.get(rid)
        if not q:
            st["找不到请求"] += 1; continue

        def rej(w):
            st[w] += 1
            rj.write(json.dumps({"request_id": rid, "parent_id": r.get("parent_id"),
                                 "round_id": r.get("round_id"), "why": w}, ensure_ascii=False) + "\n")

        if not r.get("ok"): rej("请求失败"); continue
        if r.get("finish") != "stop": rej("未正常结束"); continue
        try:
            c = json.loads(r["content"])
        except Exception:
            rej("JSON解析失败"); continue
        obs = json.loads(q["request_payload"]["messages"][1]["content"])
        plan = {x["position_1based"]: x["aa"] for x in obs["committed_plan"]}
        # ★两种响应格式: v1 让教师回显 picks(位置和氨基酸都由它写, 856 条因此改错了氨基酸);
        #   v2_fixed 用"按位置做键"的 schema —— 位置由 schema 的 required 锁死、氨基酸不让它写,
        #   组装时从真答案取 ⇒ 回显错误在构造上不可能发生。
        if "why_by_position" in c:
            whys = c["why_by_position"]
            if {int(k) for k in whys} != set(plan): rej("按位置的键与plan不一致"); continue
            c = {"reasoning": c["reasoning"], "uncertainty": c["uncertainty"],
                 "picks": [{"position_1based": p, "aa": plan[p], "why": whys[str(p)]}
                           for p in sorted(plan)]}
            got = dict(plan)
        else:
            got = {x["position_1based"]: x["aa"] for x in c["picks"]}
        if plan != got: rej("回显与plan不一致"); continue
        rd = int(q["round_id"][1:])
        # ★v12: R1 也有表了, 所以这条不再只查 R2-R4
        if obs.get("candidate_table"):
            cand = {int(x.split(" ", 1)[0]) for x in obs["candidate_table"]}
            if not set(got) <= cand: rej("picks含表外位置"); continue
        txt = c["reasoning"] + " " + c["uncertainty"] + " " + " ".join(x["why"] for x in c["picks"])
        if LEAK.search(txt): rej("泄露答案是给的"); continue

        # ---- 闸6 EC 类矛盾 ----
        ecs = (obs.get("reaction") or {}).get("ec_numbers") or []
        ec1 = int(ecs[0].split(".")[0]) if ecs and ecs[0][:1].isdigit() else 0
        if ec1 in EC_CONTRA and EC_CONTRA[ec1].search(txt):
            rej("EC类矛盾(反应词)"); continue
        # ★断言的酶类与 EC 第一位对不上 ⇒ 类别说反了(实测抓到 EC 4.1.2.43 被说成 formyltransferase)
        if ec1 and any(WORD2EC.get(mm.group(1).lower(), ec1) != ec1
                       for mm in ASSERT_CLASS.finditer(txt)):
            rej("EC类矛盾(断言酶类)"); continue
        # ---- 闸7 自己编配体距离 ----
        if LIG_DIST.search(txt): rej("编造配体距离"); continue
        # ---- 闸8 跨位点指认错残基 ----
        known = {x["position_1based"]: x["aa"] for x in (obs.get("committed") or [])}
        known.update(plan)
        if any(known.get(int(m.group(2)), m.group(1)) != m.group(1) for m in PTR.finditer(txt)):
            rej("跨位点指认错残基"); continue
        # ---- 闸9 表外编号(v12: R1 也有表, 所以都查) ----
        if obs.get("candidate_table"):
            rows = {}
            for row in obs["candidate_table"]:
                tk = row.split()
                rows[int(tk[0])] = tk
            legit = set(rows) | set(plan) | set(known)
            bad = False
            for x in c["picks"]:
                own = row_own_numbers(rows.get(x["position_1based"], []))
                if any(int(n) not in legit and int(n) not in own and 1 <= int(n) <= q["length"]
                       for n in NUM.findall(x["why"])):
                    bad = True; break
            if bad: rej("why里有表外编号"); continue
        # ---- 闸10(v12 新增) 复述列值 ----
        why_txt = " ".join(x["why"] for x in c["picks"])
        if RECITE.search(why_txt): rej("why里复述列值"); continue
        if COLNAME.search(why_txt): rej("why里出现列名"); continue
        um = q["request_payload"]["messages"][1]["content"].lower()
        if any(w.lower() not in NOT_ENZYME and w.lower() not in um
               for w in set(ASE.findall(c["reasoning"]))):
            rej("编造酶名"); continue

        # ★组装: 删掉 committed_plan; system 换成不含"给了你答案"的盲版
        stu = {k: v for k, v in obs.items() if k != "committed_plan"}
        sysmsg = q["request_payload"]["messages"][0]["content"]
        sysmsg = (sysmsg.replace("You are given the plan this round committed. Write the reasoning that derives it from the reaction chemistry and the table.",
                                 "Decide the plan for this round and write the reasoning that derives it from the reaction chemistry and the table.")
                        .replace("You are given the plan this round committed. Write the reasoning that derives it from the reaction chemistry and the chain length.",
                                 "Decide the plan for this round and write the reasoning that derives it.")
                        .replace("You are given the plan this round committed. Write the reasoning that derives it from the table, the already-committed anchors and the reaction.",
                                 "Decide the plan for this round and write the reasoning that derives it from the table, the already-committed anchors and the reaction.")
                        .replace("- Echo the committed plan EXACTLY in `picks`: same positions, same amino acids, same count. Do not add, drop or change any pair.\n", "")
                        .replace('- Never write that the plan "was given" or "was provided", and never refer to these instructions. Write the derivation as if you were deducing it.\n', ""))
        # ★学生的输出格式是 picks 数组, 位置要它自己选 —— 教师那版的 "键是给你定好的" 必须改掉,
        #   否则是在教它一个它根本产不出的格式。
        sysmsg = (sysmsg
                  .replace("- `why_by_position` has exactly one entry per position in the plan; the keys are fixed for you. For each, give the reason that position carries the amino acid the plan assigns to it.",
                           "- Output `picks`: one entry per position you commit this round, each with `position_1based`, `aa` and `why` - the reason that position carries that amino acid.")
                  .replace("- Each `why_by_position` entry must argue chemistry: which group is attacked, what is stabilised, what donates or accepts a proton. Under 120 characters.",
                           "- Each `why` must argue chemistry: which group is attacked, what is stabilised, what donates or accepts a proton. Under 120 characters.")
                  .replace("each why_by_position entry must argue",
                           "each `why` must argue")
                  .replace("Each why_by_position entry must cite",
                           "Each `why` must cite")
                  .replace("Every position in the plan is a row of `candidate_table`",
                           "Every position you pick must be a row of `candidate_table`"))
        assert "why_by_position" not in sysmsg, "学生 system 里还残留 why_by_position 的措辞"
        assert "committed plan" not in sysmsg.lower(), "学生 system 里还残留'给了答案'的措辞"
        picks = [{"position_1based": x["position_1based"], "aa": x["aa"], "why": x["why"]}
                 for x in c["picks"]]
        asst = json.dumps({"reasoning": c["reasoning"], "picks": picks,
                           "uncertainty": c["uncertainty"]},
                          separators=(",", ":"), ensure_ascii=False)   # ★不要 sort_keys
        assert asst.startswith('{"reasoning"'), "assistant 必须 reasoning 在前"
        nw.write(json.dumps({
            "schema": "agent_dplmbit.v12_sft.v1",
            "parent_id": r["parent_id"], "round_id": q["round_id"], "band": q["band"],
            "split": q.get("split", "train"), "length": q["length"],
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": json.dumps(stu, sort_keys=True,
                                                                separators=(",", ":"), ensure_ascii=False)},
                         {"role": "assistant", "content": asst}],
            "n_picks": len(picks), "n_candidates": q.get("n_candidates", 0),
            "teacher_was_target_informed": True, "student_sees_plan": False,
            "evidence_tier": tier_map.get(r["parent_id"], a.evidence_tier),
        }, ensure_ascii=False) + "\n")
        if tier_map and r["parent_id"] not in tier_map:
            miss_tier[r["parent_id"]] += 1
        st["采用"] += 1; per[q["round_id"]] += 1; sec.append(r.get("secs") or 0)
    nw.close(); rj.close()
    tot = sum(st.values())
    print(f"\n处理 {tot:,} | 采用 {st['采用']:,} ({st['采用']/max(tot,1):.2%})")
    print("各轮采用:", dict(sorted(per.items())))
    print("拒绝原因:", {k: v for k, v in st.most_common() if k != "采用"})
    if sec: print(f"教师耗时 中位 {statistics.median(sec):.1f}s")
    if tier_map:
        # ★join 命中率必须断言(feedback_join_key_must_assert_hit_rate), 否则会静默回落到默认档位
        hit = 1 - len(miss_tier) / max(st["采用"], 1)
        print(f"档位表命中 {1-len(miss_tier)/max(st['采用'],1):.4%} "
              f"(未命中的 parent {len(miss_tier)} 个)")
        assert len(miss_tier) == 0, f"★有 {len(miss_tier)} 个 parent 不在档位表里, 会被打成默认档位"
    print("\n★学生输入不含 committed_plan; assistant 以 reasoning 开头; 学生 system 无 why_by_position(三条都已硬断言)")


if __name__ == "__main__":
    main()
