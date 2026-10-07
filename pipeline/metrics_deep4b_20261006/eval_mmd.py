# -*- coding: utf-8 -*-
"""按 CodeFP 官方口径算 MMD_lin 和 MMD_rbf(附录 A.1)。

★口径逐条照抄论文, 不自己发明:
  表示   normalized Spectrum Mapping (k-mer 频率) —— 论文原文, 不是 ESM 嵌入也不是结构描述子
  MMD_lin  |mu_P - mu_G|^2       两组均值嵌入的欧氏距离(论文 Eq.7)
  MMD_rbf  RBF 核 exp(-gamma*||x-y||^2), 带宽 gamma 用【中位数启发式】
  越低越好 = 生成序列的统计性质越接近天然
★论文没写 k 等于几 ⇒ 同时报 k=1,2,3, 把这件事摊开而不是偷偷选一个。
★两个会让数失真的坑, 都处理掉:
  1 MMD 对样本量敏感 ⇒ 两组必须【同样多】。这里按 id 取交集后配对, 不用不同分母。
  2 中位数启发式必须在【合并后的样本】上算, 对两组一视同仁; 只在一组上算会偏。
用法: eval_mmd.py --gen <gen.fasta...> --nat <nat.fasta> [--subset ids.txt] [--k 1,2,3]
"""
import argparse
import itertools
import json

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(p):
    out, k, buf = {}, None, []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            if k:
                out[k] = "".join(buf)
            k, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if k:
        out[k] = "".join(buf)
    return out


def spectrum(seq, k, index):
    """归一化 k-mer 频率。未知字符的 k-mer 直接丢(不塞进任何一维)。"""
    v = np.zeros(len(index), dtype=np.float64)
    n = 0
    for i in range(len(seq) - k + 1):
        j = index.get(seq[i:i + k])
        if j is not None:
            v[j] += 1.0
            n += 1
    return v / n if n else v


def mmd_lin(X, Y):
    d = X.mean(0) - Y.mean(0)
    return float(d @ d)                     # |mu_P - mu_G|^2, 论文 Eq.7


def mmd_rbf(X, Y):
    Z = np.vstack([X, Y])
    # ★中位数启发式: 在【合并样本】上取两两距离平方的中位数, 对两组一视同仁
    m = min(len(Z), 800)
    idx = np.random.default_rng(0).choice(len(Z), m, replace=False)
    S = Z[idx]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices(m, 1)])
    gamma = 1.0 / med if med > 0 else 1.0

    def K(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)

    n, m2 = len(X), len(Y)
    kxx = (K(X, X).sum() - n) / (n * (n - 1))          # 无偏: 去掉对角
    kyy = (K(Y, Y).sum() - m2) / (m2 * (m2 - 1))
    kxy = K(X, Y).mean()
    return float(kxx + kyy - 2 * kxy), float(gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--nat", required=True)
    ap.add_argument("--subset", default="")
    ap.add_argument("--k", default="1,2,3")
    a = ap.parse_args()

    gen = {}
    for p in a.gen:
        gen.update(read_fasta(p))
    nat = read_fasta(a.nat)
    ids = sorted(set(gen) & set(nat))
    if a.subset:
        keep = {x.strip() for x in open(a.subset) if x.strip()}
        ids = [i for i in ids if i in keep]
    print(f"  生成 {len(gen):,} | 天然 {len(nat):,} | 配对后 {len(ids):,}")
    # ★join 命中率必须断言(feedback_join_key_must_assert_hit_rate)
    assert len(ids) > 0.9 * min(len(gen), len(nat)) or a.subset, \
        f"★配对只剩 {len(ids)}, id 格式可能不一致"
    assert len(ids) >= 100, "★样本太少, MMD 不稳"

    for k in [int(x) for x in a.k.split(",")]:
        index = {"".join(t): i for i, t in enumerate(itertools.product(AA, repeat=k))}
        G = np.stack([spectrum(gen[i], k, index) for i in ids])
        P = np.stack([spectrum(nat[i], k, index) for i in ids])
        lin = mmd_lin(P, G)
        rbf, gam = mmd_rbf(P, G)
        # 同量级参照: 把天然随机对半分, 这是"同分布"时 MMD 应有的量级
        rng = np.random.default_rng(1)
        pi = rng.permutation(len(P))
        h = len(P) // 2
        lin0 = mmd_lin(P[pi[:h]], P[pi[h:2 * h]])
        rbf0, _ = mmd_rbf(P[pi[:h]], P[pi[h:2 * h]])
        print(f"  k={k} ({len(index):,} 维)  MMD_lin {lin:.3e}  MMD_rbf {rbf:.3e}   "
              f"| 天然自比(同分布基准) lin {lin0:.3e} rbf {rbf0:.3e}  | gamma {gam:.3g}")
    print("  读法: 越低越好。与【天然自比】那一列比才有意义 —— 它是同分布时的量级,")
    print("        我们的数离它多远, 才是真正的分布偏离。")


if __name__ == "__main__":
    main()
