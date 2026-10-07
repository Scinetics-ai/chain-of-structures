# -*- coding: utf-8 -*-
"""从 final_struct_feature 解码几何, 并给出位点判据。生产机一张卡都不碰。

★CA/CB 的 atom37 下标必须从 residue_constants 取, 不能假定(feedback_esmfold_scrmsd_pipeline:
  别用 positions[*,:,1] 取 CA)。
★合法性硬闸: 相邻 CA 距离中位必须 3.6-4.0 Å(feedback_gate_generation_on_geometry_not_exit_code:
  EXIT=0 也能塌成球)。
★位点判据沿用项目已有口径, 这样能直接和这些参照比:
  天然结构本身 最近一对 CA 6.87Å / <8Å 80.6%
  钉天然结构bit@天然位置 6.08Å / 67.7%
  只钉身份@天然位置(共生成) 12.56Å / 33.3%
  只钉身份@随机位置(共生成) 14.96Å / 12.5%
"""
import numpy as np
import torch

_IDX = {}


def atom_idx(name, m):
    if name not in _IDX:
        from byprot.utils.protein import residue_constants as rc
        _IDX[name] = list(rc.atom_types).index(name)
    return _IDX[name]


def decode(m, feat, L, res_mask=None):
    """feat: final_struct_feature。★实测它是 (B, L, 13) —— 【不含 bos/eos】,
    我一开始按含 bos/eos 切成 1:L+1, 结果长度差一位、在 post_quant 里报维度不匹配。
    这里两种形状都接受, 并断言, 不再凭印象。"""
    T = feat.shape[1]
    if T == L:
        q = feat.float()
    elif T == L + 2:
        q = feat[:, 1:L + 1].float()
    else:
        raise AssertionError(f"★final_struct_feature 第二维 {T} 既不是 L={L} 也不是 L+2")
    B = q.shape[0]
    # ★res_mask 必须是 float32: structok_lfq 内部做 `padding_mask = 1 - mask`,
    #   新版 torch 对 bool 张量的减法直接报错。生产脚本用的也是 float32。
    msk = torch.ones(B, L, dtype=torch.float32, device=q.device) if res_mask is None else res_mask
    with torch.no_grad():
        dec = m.struct_tokenizer.detokenize(q, res_mask=msk)
    pos = dec["atom37_positions"].float().cpu().numpy()
    pl = dec["plddt"].float().cpu().numpy()
    assert pos.shape[1] >= L, f"★解码出的残基数 {pos.shape[1]} < L={L}"
    return pos[:, :L], pl[:, :L]


def ca_step_median(pos, m):
    c = pos[:, atom_idx("CA", m)]
    d = np.linalg.norm(c[1:] - c[:-1], axis=-1)
    return float(np.median(d))


def burial(pos, m, cut=10.0):
    c = pos[:, atom_idx("CA", m)]
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    return (d < cut).sum(1) - 1


def site_metrics(pos, m, idx0, sep_nl=4):
    """idx0: 0-based 位置数组。返回位点几何。

    ★`min_CA`(所有对) 对含序列相邻残基的集合【没有区分力】: 相邻 CA 恒为约 3.8Å,
      所以实测它在天然核心和我挑的口袋上都是 3.7-3.8, 读不出任何东西。
      项目里那些参照值(天然 6.87 / 只钉身份 12.56 / 随机 14.96)测的是
      **3 个序列上分散的催化位点**, 本质上就是非局部对。
      ⇒ 真正可比的是 `min_CA_nl`: 只在 |i-j| >= sep_nl 的对上取最近距离。
    """
    out = {}
    idx0 = np.asarray(idx0)
    ii, jj = np.triu_indices(len(idx0), 1)
    nl = np.abs(idx0[ii] - idx0[jj]) >= sep_nl
    for nm in ("CA", "CB"):
        c = pos[idx0, atom_idx(nm, m)]
        d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
        pair = d[ii, jj]
        out[f"min_{nm}"] = float(pair.min()) if len(pair) else float("nan")
        out[f"min_{nm}_nl"] = float(pair[nl].min()) if nl.any() else float("nan")
        out[f"med_{nm}_nl"] = float(np.median(pair[nl])) if nl.any() else float("nan")
    out["n_pairs_nl"] = int(nl.sum())
    # ★k 稳健的"收敛度": 最近一对会随 k 变大而机械变小(在 3 对里取最小 vs 在 30 对里取最小),
    #   所以绝对值不能和项目里 k=3 测出来的 6.87/12.56 对比。中位与"<8Å 占比"对 k 不敏感。
    c = pos[idx0, atom_idx("CA", m)]
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    pr = d[ii, jj][nl] if nl.any() else np.array([np.nan])
    out["frac_nl_lt8"] = float(np.mean(pr < 8.0)) if nl.any() else float("nan")
    out["frac_nl_lt12"] = float(np.mean(pr < 12.0)) if nl.any() else float("nan")
    c = pos[idx0, atom_idx("CA", m)]
    out["gyr"] = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))
    return out


def pick_pocket(pos, m, k, burial_floor_pct=50.0, sep=4, frac_units=0.5):
    """只用草稿几何挑 k 个位置, 当作"口袋"。不碰任何天然信息。

    ★第一版是错的: 取"离某个中心最近的 k 个", 结果挑出来的是【一段连续序列】——
      序列上相邻的残基在三维里自然最近, 实测回转半径只有 1.6-3.0 Å,
      那是链上的一小段, 不是口袋。
    ★改用天然核心的【间距分布】当结构先验(来自数据集整体, 不是单条样本的答案):
      相邻位置间距 中位 3, 其中 42.8% 恰好是 1(紧邻短 motif), 42.3% >=10(换结构单元);
      每个核心跨 6 个结构单元(间距>=4 算换) ⇒ k≈12 时约 6 个单元、每个约 2 个残基。
    做法: 先贪心挑 u 个【两两序列间隔 >= sep】的锚点使它们在三维上最聚拢,
          再把剩下的名额贴在锚点旁边组成短片段。
    """
    c = pos[:, atom_idx("CA", m)]
    L = c.shape[0]
    bur = burial(pos, m)
    floor = np.percentile(bur, burial_floor_pct)
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    u = max(2, min(k, int(round(k * frac_units))))

    def gyr_of(ix):
        p = c[list(ix)]
        return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))

    best, bset = np.inf, None
    for s0 in range(L):
        if bur[s0] < floor:
            continue
        anch = [s0]
        while len(anch) < u:
            cand = [j for j in range(L)
                    if bur[j] >= floor and all(abs(j - q) >= sep for q in anch)]
            if not cand:
                break
            j = min(cand, key=lambda t: gyr_of(anch + [t]))
            anch.append(j)
        if len(anch) < u:
            continue
        g = gyr_of(anch)
        if g < best:
            best, bset = g, list(anch)
    if bset is None:                       # 埋藏下限太严 ⇒ 放开它重来一次
        return pick_pocket(pos, m, k, burial_floor_pct=0.0, sep=sep, frac_units=frac_units)

    # 把剩余名额贴在锚点旁, 组成短片段(对上"42.8% 间距=1"那一峰)
    sel = list(bset)
    off = 1
    while len(sel) < k:
        added = False
        for q in bset:
            for cand in (q + off, q - off):
                if 0 <= cand < L and cand not in sel:
                    sel.append(cand)
                    added = True
                    if len(sel) >= k:
                        break
            if len(sel) >= k:
                break
        off += 1
        if not added and off > L:
            break
    sel = sorted(sel[:k])
    return np.array(sel), gyr_of(sel)
