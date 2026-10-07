# -*- coding: utf-8 -*-
"""v12 新 R1 的数据审查。采用率高 ≠ 数据有用。

★要查的四件事(每一件都有过先例):
  1 **近重复文本**: 教师 temperature=0, 同 EC3 的提示词又很像 ⇒ 可能产出成批几乎一样的推理。
    那样"37,290 条"的有效信息远少于条数(project_dataset_duplication_audit 的形状)。
    按 reasoning 的 5-gram 指纹聚类, 报最大簇和去重后的有效条数。
  2 **输出的氨基酸组成** vs 天然催化组成: 偏了就说明教师在复述频率表而不是读反应
    (project_s0_learns_exactly_the_catalytic_prior 的形状)。
  3 **残留的列值复述**: 组装闸只拦了明确形态, 这里用更宽的网再扫一遍。
  4 **reasoning 是否真提到这条反应的化学**: 统计提到底物/产物关键词、以及纯模板句的比例。
用法: audit_v12_r1.py <sft.jsonl> <core_ec.json>
"""
import collections
import hashlib
import json
import re
import sys

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
f_sft, f_ce = sys.argv[1], sys.argv[2]
ce = json.load(open(f_ce))

recs = []
for ln in open(f_sft, errors="ignore"):
    if ln.lstrip().startswith("{"):
        recs.append(json.loads(ln))
print(f"  审查 {len(recs):,} 条")

# ---- 1 近重复 ----
def shingles(t, k=5):
    w = re.findall(r"[a-z]+", t.lower())
    return {hashlib.md5(" ".join(w[i:i + k]).encode()).hexdigest()[:10]
            for i in range(max(0, len(w) - k + 1))}


sig = {}
for r in recs:
    a = json.loads(r["messages"][2]["content"])
    sh = shingles(a["reasoning"])
    # 用最小 16 个 shingle 当签名(MinHash 的简化版)
    key = tuple(sorted(sh)[:16])
    sig.setdefault(key, []).append(r["parent_id"])
cl = sorted((len(v) for v in sig.values()), reverse=True)
uniq = len(sig)
print(f"\n  1) 近重复(reasoning 的 5-gram MinHash 签名)")
print(f"     不同签名 {uniq:,} / {len(recs):,} = {uniq/len(recs):.1%} ⇒ 有效条数约 {uniq:,}")
print(f"     最大簇 {cl[0]} 条 | 前5簇 {cl[:5]} | 簇>=10 的个数 {sum(1 for x in cl if x>=10)}")

# ---- 2 氨基酸组成 ----
out = collections.Counter()
nat = collections.Counter()
for r in recs:
    a = json.loads(r["messages"][2]["content"])
    for p in a["picks"]:
        if p["aa"] in AA20:
            out[p["aa"]] += 1
    # ★口径更正: picks 的氨基酸是从【被选中的那些位置】的真答案取的, 所以要和
    #   那些位置的天然残基比, 不能和该 parent【全部】催化残基比 ——
    #   后者会把"表只覆盖了一部分位置"这个选择效应算成组装误差(第一版就把 TVD 0.0304
    #   误判成"组装有错")。
    v = ce.get(r["parent_id"])
    if v:
        m = {p: a for p, a in zip(v["core"], v["aa"])}
        for p in a["picks"]:
            x = m.get(p["position_1based"])
            if x in AA20:
                nat[x] += 1
to, tn = sum(out.values()), sum(nat.values())
tvd = 0.5 * sum(abs(out[k] / to - nat[k] / tn) for k in (set(out) | set(nat)))
print(f"\n  2) 输出氨基酸组成 vs 该 parent 的天然催化组成")
print(f"     输出 top6 {' '.join('%s%.1f%%'%(k,100*v/to) for k,v in out.most_common(6))}")
print(f"     天然 top6 {' '.join('%s%.1f%%'%(k,100*v/tn) for k,v in nat.most_common(6))}")
print(f"     TVD = {tvd:.4f}  (和【被选中位置】的天然残基比, 必须是 0 —— 不是 0 就是组装把氨基酸弄错了)")

# ---- 3 列值复述(更宽的网) ----
WIDE = re.compile(r"\b(tplAA|tplD|ecAA|pL|prior|burial|plddt|dCore|dCen|draftwin|seqdistInj|"
                  r"row|column|table|shortlist|candidate)\b", re.I)
hit = collections.Counter()
for r in recs:
    a = json.loads(r["messages"][2]["content"])
    blob = a["reasoning"] + " " + a["uncertainty"] + " " + " ".join(x["why"] for x in a["picks"])
    for m in set(x.lower() for x in WIDE.findall(blob)):
        hit[m] += 1
print(f"\n  3) 残留的表/列词(宽网; 组装闸只拦明确形态)")
print(f"     {dict(hit.most_common(8))}  占比 "
      f"{sum(1 for r in recs if WIDE.search(json.loads(r['messages'][2]['content'])['reasoning']))/len(recs):.1%} 的 reasoning 里出现")

# ---- 4 是否真读了反应 ----
CHEM = re.compile(r"\b(nucleophil|electrophil|proton|carboxylate|phosphate|carbonyl|thiol|imine|"
                  r"hydroxyl|tetrahedral|transition state|oxyanion|hydride|decarboxylat|"
                  r"hydrolys|transfer|redox|oxidis|oxidiz|reduc)\w*", re.I)
nchem = sum(1 for r in recs
            if len(set(x.lower() for x in CHEM.findall(
                json.loads(r["messages"][2]["content"])["reasoning"]))) >= 3)
print(f"\n  4) reasoning 里出现 >=3 个不同化学概念的比例: {nchem/len(recs):.1%}")
lens = [len(json.loads(r["messages"][2]["content"])["reasoning"]) for r in recs]
import statistics as st
print(f"     reasoning 字符 中位 {int(st.median(lens))} / p10 {sorted(lens)[len(lens)//10]} / "
      f"p90 {sorted(lens)[9*len(lens)//10]}")
