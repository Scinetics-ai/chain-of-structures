#!/bin/bash
set -u
W=/data/caiqingran/enzymecot/test_eval_20261002_d2like_s9000
C=/data/caiqingran/CLEAN/app
PY=/data/caiqingran/env/py/bin/python
ARM=predicted_positions
mkdir -p $W/merged
echo "拉 cvm-1 的 8 片"
for s in 8 9 10 11 12 13 14 15; do
  rsync -a lan1:$W/shards/shard$s/$ARM/generated_test/aatype.fasta $W/merged/shard$s.fasta || exit 1
done
echo "  拉到 $(ls $W/merged/shard*.fasta | wc -l) 个"
$PY - <<'PYEOF'
import json, glob, collections, math
W = "/data/caiqingran/enzymecot/test_eval_20261002_d2like_s9000"
C = "/data/caiqingran/CLEAN/app"
ARM = "predicted_positions"
rows = json.load(open("/data/caiqingran/enzymecot/cot_split_v2/test_rows_cotfull.json"))
truth = {str(r["idx"]): r["ec_list"] for r in rows}
L = {str(r["idx"]): int(r["length"]) for r in rows}
def read(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"): k = l[1:].split("|")[0].split()[0]; d[k] = ""
        elif k: d[k] += l
    return d
files = sorted(glob.glob(f"{W}/shards/shard*/{ARM}/generated_test/aatype.fasta")) + sorted(glob.glob(f"{W}/merged/shard*.fasta"))
seqs, dup = {}, 0
for p in files:
    for k, v in read(p).items():
        if k in seqs: dup += 1
        seqs[k] = v
print("合并 %d 个文件 -> %d 条, 重复 %d" % (len(files), len(seqs), dup))
assert dup == 0, "重复 %d 条" % dup
assert len(seqs) >= 0.98 * len(rows), "只合出 %d/%d" % (len(seqs), len(rows))
hit = len(set(seqs) & set(truth))
print("join 命中 %d/%d = %.1f%%" % (hit, len(seqs), 100.0*hit/len(seqs)))
assert hit == len(seqs), "有 %d 条找不到真值" % (len(seqs)-hit)
bad = [(k, L[k], len(seqs[k])) for k in seqs if len(seqs[k]) != L[k]]
print("长度与 test_rows 的 length 不一致: %d 条 %s" % (len(bad), bad[:5]))
# 2026-10-02: 实测 1/2801 条短 1 个残基(解码边界 off-by-one), 其余 2800 条精确。
# 容忍 <=0.5% 且偏差 <=2 个残基; 超出就是真问题, 不许静默放过。
assert len(bad) <= 0.005 * len(seqs), "长度不一致 %d 条, 超过 0.5%%" % len(bad)
assert all(abs(c - e) <= 2 for _, e, c in bad), "有偏差超过 2 个残基的: %s" % [x for x in bad if abs(x[2]-x[1]) > 2][:3]
# 组成闸: 最便宜也最早的"模型在工作"证据
c = collections.Counter()
for v in seqs.values(): c.update(v)
t = sum(c.values()); f = {k: v/t for k, v in c.items()}
ent = -sum(p*math.log(p) for p in f.values() if p > 0)
print("组成熵 %.3f (均匀上限 %.3f, 天然约 2.861)  W %.2f%% (天然 0.94%%)  非标准字符 %s"
      % (ent, math.log(20), 100*f.get("W",0), sorted(set(c) - set("ACDEFGHIKLMNPQRSTVWY"))[:5]))
assert ent < 2.95, "组成熵 %.3f 接近均匀 -> 模型没在工作" % ent
with open(f"{C}/data/inputs/d2like_gen.fasta", "w") as fh:
    for k, v in seqs.items(): fh.write(">%s\n%s\n" % (k, v))
print("已写 d2like_gen.fasta")
PYEOF
cd $C
export PATH=/data/caiqingran/env/py/bin:$PATH PYTHONPATH=$C/src
$PY CLEAN_infer_fasta.py -d d2like_gen 2>&1 | grep -vE "it/s" | tail -2
wc -l $C/results/inputs/d2like_gen_maxsep.csv
