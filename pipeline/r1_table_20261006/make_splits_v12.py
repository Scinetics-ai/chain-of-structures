# -*- coding: utf-8 -*-
"""v12 划分: 验证集改成和【测试集同分布】, 并保留一个诚实读数。

★为什么改(2026-10-06 查出来的):
  测试集 5,426 parent 里 **5,269(97%) 是 test_overlap**(反应与训练集重叠), 只有 157 是全新反应。
  而 v11 的 val 是**反应完全不交**的 ⇒ val 比 test 难得多, 拿它做训练决策会判错。
  实测后果: 同一套造表规则, 表对本档覆盖 train 61.6% vs val 51.1%,
  模板残基命中 train 48.9% vs val 29.9% —— 不是规则有问题, 是**两个分布不一样**。

★新划分(三份):
  val_overlap  = 从 parent>=4 的反应里各抽 1 个 parent, **那条反应仍留在 train**
                 ⇒ 和 test_overlap 同一个 régime, **这是选模型/早停用的那一份**
  val_strict   = 沿用 v11 那个反应不交的 val ⇒ 全新化学的诚实读数, **不用于选模型**
  train        = 其余
★造表时**仍然剔掉「同一条反应」的兄弟**(leave-one-reaction-out), 三份一视同仁。
  不剔的话表的覆盖率是 **96.5%** —— 答案就在表里, agent 退化成纯检索
  ([[feedback_retrieval_baseline_is_now_a_published_threat]]: 检索基线已经能打赢模型)。
  那个 96.5% 要单独作为**检索基线**报出来, 不能混进我们的数。
用法: make_splits_v12.py <sft_r1_v11.jsonl> <val_pids(v11,反应不交)> <heldout_pids> <出目录> [每反应最少留 3]
"""
import collections
import json
import random
import sys
from pathlib import Path

r1f, val_old_f, held_f, outd = sys.argv[1:5]
KEEP = int(sys.argv[5]) if len(sys.argv) > 5 else 3
Path(outd).mkdir(parents=True, exist_ok=True)

held = {x.strip() for x in open(held_f) if x.strip()}
val_strict = {x.strip() for x in open(val_old_f) if x.strip()}
pool = collections.defaultdict(list)
for ln in open(r1f, errors="ignore"):
    if not ln.lstrip().startswith("{"):
        continue
    p = json.loads(ln)["parent_id"]
    if p in held or p in val_strict:
        continue
    pool[p.split("_")[0]].append(p)
for k in pool:
    pool[k] = sorted(set(pool[k]))
print(f"  训练池(已去掉留出集与 val_strict): parent {sum(len(v) for v in pool.values()):,}"
      f" / 反应 {len(pool):,}")

# ★抽法必须和留出集/test_overlap 一样: **均匀随机抽 parent**, 不按反应大小筛。
#   第一版按"parent>=4 的反应各抽 1 个"抽, 结果 val_overlap 比 train 还容易
#   (覆盖 70.4% vs 61.5%, 模板命中 66.0% vs 48.4%) —— 那个条件和表的质量正相关,
#   等于挑了容易的 ([[feedback_pilot_sample_is_biased]])。
rng = random.Random(20261006)
allp = sorted(p for v in pool.values() for p in v)
rng.shuffle(allp)
N_OV = int(sys.argv[6]) if len(sys.argv) > 6 else 1500
left = {k: len(v) for k, v in pool.items()}
val_ov = set()
for q in allp:
    if len(val_ov) >= N_OV:
        break
    k = q.split("_")[0]
    if left[k] - 1 >= 1:                  # 那条反应至少还要给 train 留 1 个
        val_ov.add(q); left[k] -= 1
train = {p for v in pool.values() for p in v} - val_ov

rx_tr = {p.split("_")[0] for p in train}
rx_ov = {p.split("_")[0] for p in val_ov}
rx_st = {p.split("_")[0] for p in val_strict}
print(f"\n  train        {len(train):,} parent / {len(rx_tr):,} 反应")
print(f"  val_overlap  {len(val_ov):,} parent / {len(rx_ov):,} 反应"
      f"  ← 反应全在 train 里? {rx_ov <= rx_tr}")
print(f"  val_strict   {len(val_strict):,} parent / {len(rx_st):,} 反应"
      f"  ← 与 train 反应交集 {len(rx_st & rx_tr)}")
assert rx_ov <= rx_tr, "★val_overlap 的反应必须全在 train 里"
assert not (rx_st & rx_tr), "★val_strict 必须与 train 反应不交"
assert not (train & val_ov) and not (train & val_strict) and not (val_ov & val_strict)
assert not ((train | val_ov | val_strict) & held), "★不许和留出集相交"

for nm, s in (("train", train), ("val_overlap", val_ov), ("val_strict", val_strict)):
    open(f"{outd}/{nm}_pids.txt", "w").write("\n".join(sorted(s)) + "\n")
print(f"\n  写出 {outd}/{{train,val_overlap,val_strict}}_pids.txt")
print("  ★val_overlap 是和 test_overlap(占测试集 97%)同 régime 的那一份, 用它选模型;")
print("    val_strict 只做全新化学的诚实读数, 两份分开报, 永远不要合在一起。")
