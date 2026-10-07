# -*- coding: utf-8 -*-
"""v12 合并 + 重切: 新 R1(带表) + 旧 R2-R4(不动), 按 v12 三份划分重新分配。

★为什么 split 必须重判, 不能信旧字段:
  v11 的记录里 split 只有 train/val, 而 v12 把验证集改成了 val_overlap + val_strict
  (测试集 97% 是反应重叠的, v11 那个反应不交的 val 比 test 难得多 ⇒ 判错训练决策)。
  旧字段一律丢掉, 一切按 splits_v12/{train,val_overlap,val_strict}_pids.txt 重判。

★八道硬断言(前七条沿用 v11, 第八条是 v12 新增):
  1 assistant 必须以 `{"reasoning"` 开头 —— 否则是答案排在解释前面
    (feedback_sort_keys_puts_the_answer_before_the_cot: 字母序会让 picks < reasoning)
  2 学生 user 消息里不许有 committed_plan
  3 学生 system 里不许有 why_by_position(那是教师的格式, 学生产不出)
  4 (parent, round) 不许重复
  5 与留出集零交集
  6 轮次只能是 1-4
  7 evidence_tier 必须有
  8 ★每条的 split 必须等于它 parent 在划分文件里的那一份, 且三份互不相交
用法: merge_split_v12.py <新R1.jsonl> <v11合并.jsonl> <splits目录> <留出pids> <出目录>
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

r1_f, v11_f, spl_d, held_f, outd = sys.argv[1:6]
Path(outd).mkdir(parents=True, exist_ok=True)


def rd(fp):
    return {x.strip() for x in open(fp)} - {""}


held = rd(held_f)
SPL = {}
for nm in ("train", "val_overlap", "val_strict"):
    for q in rd(f"{spl_d}/{nm}_pids.txt"):
        assert q not in SPL, f"★parent {q} 出现在两份划分里"
        SPL[q] = nm
print(f"  划分 {collections.Counter(SPL.values())} | 留出 {len(held):,}")
assert not (set(SPL) & held), "★划分与留出集相交"

# ★补划分: splits_v12 是从 sft_r1_v11.jsonl 的 parent 建的, 那文件只含【有 R1 的 parent】。
#   有些 parent 只有 R2-R4(它们的 R1 当年被 v11 的闸拒了) ⇒ 不在任何划分里。
#   硬断言抓到 5,650 条。按【反应】补: 反应属于 val_strict 的必须进 val_strict,
#   否则会破坏 val_strict 与 train 的反应不交。
VS_RXN = {q.split("_")[0] for q, v in SPL.items() if v == "val_strict"}
VO_RXN = {q.split("_")[0] for q, v in SPL.items() if v == "val_overlap"}


def backfill(pid):
    rx = pid.split("_")[0]
    # val_overlap 是按 parent 抽的, 它的反应同时也在 train 里 ⇒ 补进来的 parent 归 train
    return "val_strict" if rx in VS_RXN else "train"


bad = collections.Counter()
filled = collections.Counter()
cnt = collections.Counter()
by_split = collections.Counter()
seen = set()
tier = collections.Counter()
by_parent = collections.defaultdict(set)
n = 0
outs = {nm: open(f"{outd}/sft_v12_{nm}.jsonl", "w") for nm in
        ("train", "val_overlap", "val_strict")}


def take(d, src):
    global n
    pid = d["parent_id"]
    rid = d["round_id"]
    k = int(rid[1:])
    if pid in held:
        bad["留出集的记录"] += 1
        return
    sp = SPL.get(pid)
    if sp is None:
        sp = backfill(pid)
        SPL[pid] = sp
        filled[sp] += 1
    if not (1 <= k <= 4):
        bad["轮次非法"] += 1
        return
    if (pid, k) in seen:
        bad["(parent,轮次)重复"] += 1
        return
    m = d["messages"]
    if not m[2]["content"].startswith('{"reasoning"'):
        bad["assistant未以reasoning开头"] += 1
        return
    if "committed_plan" in m[1]["content"]:
        bad["user里有committed_plan"] += 1
        return
    if "why_by_position" in m[0]["content"]:
        bad["system里有why_by_position"] += 1
        return
    if not d.get("evidence_tier"):
        bad["缺evidence_tier"] += 1
        return
    seen.add((pid, k))
    d["split"] = sp                       # ★重判, 丢掉旧字段
    d["source"] = src
    outs[sp].write(json.dumps(d, ensure_ascii=False) + "\n")
    cnt[rid] += 1
    by_split[sp] += 1
    tier[d["evidence_tier"]] += 1
    by_parent[pid].add(k)
    n += 1


for ln in open(r1_f, errors="ignore"):
    if ln.lstrip().startswith("{"):
        d = json.loads(ln)
        assert d["round_id"] == "R1", "★新 R1 文件里混进了别的轮次"
        take(d, "v12_r1_table")
print(f"  新 R1 收下 {cnt['R1']:,}")

nr = 0
for ln in open(v11_f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    d = json.loads(ln)
    if d["round_id"] == "R1":
        continue                          # ★旧 R1 整批丢掉, 由新的替换
    nr += 1
    take(d, "v11_r2_r4")
print(f"  旧文件里的 R2-R4 {nr:,} 条")

for f in outs.values():
    f.close()
print(f"\n  合计 {n:,} 条 | parent {len(by_parent):,}")
print(f"  各轮 {dict(sorted(cnt.items()))}")
print(f"  各份 {dict(by_split)}")
print(f"  档位 {dict(tier)}")
assert not bad, f"★硬断言不过: {dict(bad)}"
if filled:
    print(f"  补划分(只有 R2-R4 的 parent, 按反应归档): {dict(filled)}")
# ★补完之后必须复核反应不交, 否则补划分本身就是泄漏
tr_rxn = {p.split("_")[0] for p, v in SPL.items() if v == "train"}
st_rxn = {p.split("_")[0] for p, v in SPL.items() if v == "val_strict"}
assert not (tr_rxn & st_rxn), f"★补划分破坏了反应不交: {len(tr_rxn & st_rxn)} 条反应两边都有"
print(f"  ★复核: train 反应 {len(tr_rxn):,} / val_strict 反应 {len(st_rxn):,} / 交集 0")
print("  ★八道硬断言全过")
full = sum(1 for v in by_parent.values() if v == {1, 2, 3, 4})
print(f"  四轮齐全的 parent {full:,} ({full/max(1,len(by_parent)):.1%})")
r1p = {p for p, v in by_parent.items() if 1 in v}
rnp = {p for p, v in by_parent.items() if v & {2, 3, 4}}
print(f"  有 R1 的 parent {len(r1p):,} | 有 R2-R4 的 {len(rnp):,} | "
      f"只有 R1 的 {len(r1p - rnp):,} | 只有 R2-R4 的 {len(rnp - r1p):,}")
for nm in outs:
    p = f"{outd}/sft_v12_{nm}.jsonl"
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    print(f"  {nm:<12} {by_split[nm]:>8,} 条  md5 {h}")
