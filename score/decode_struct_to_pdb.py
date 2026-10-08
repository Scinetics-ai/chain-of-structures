# -*- coding: utf-8 -*-
"""把 4 段注入臂生成的结构 token 解码成 PDB, 供 ESMFold 自洽评测用。

★为什么必须补这一步: `eval_direct_esmfold_consistency_local.py` 的 sc_rmsd / sc_tm 是
  【生成序列经 ESMFold 折出的结构】vs【生成的结构】之间的一致性, 所以要 generated_test/pdb/。
  EVAL_KEEP_deep4b 里只有 aatype.fasta 和 struct_token.fasta, 没有 pdb。

★用解码器自带的 string_to_tensor + output_to_pdb, 不自己写 PDB 写出器 ——
  自己写要处理 atom37 顺序/占位/残基编号, 出错面大。记录里那条坑就是手搓索引搞出来的:
  `positions[*,:,1]` 取 CA 让 scRMSD 从真实 1-2Å 变成 12-29Å, 把 80% 可设计性读成 20%。

★两道硬闸(不过就整条样本作废, 绝不带着坏几何往下走):
  1 相邻 CA 距离中位必须在 3.6-4.0 Å (feedback_gate_generation_on_geometry_not_exit_code:
    EXIT=0 也能塌成球)
  2 解码出的残基数必须等于序列长度 (token 和 aa 一一对应, 对不上说明文件错位)
★token 是裸码本索引(实测取值 0-7954, 码本 8192), 不带 struct_vocab_offset —— 已核。
用法: decode_struct_to_pdb.py <gen目录> <输出目录> [--shard i --num-shards n]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def read_fasta(p):
    out, k, buf = {}, None, []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            if k:
                out[k] = "".join(buf)
            k, buf = line[1:], []
        elif line:
            buf.append(line)
    if k:
        out[k] = "".join(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stok", default="/data/workspace/caiqingran/protein_design/DISCO_repo/runs/dplm2_struct_tokenizer_hydra_wrapper")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    from byprot.models.utils import get_struct_tokenizer
    stok = get_struct_tokenizer(a.stok).eval().cuda()
    print(f"  struct_tokenizer 已加载: {type(stok).__name__}", flush=True)

    aa = read_fasta(Path(a.gen_dir) / "aatype.fasta")
    st = read_fasta(Path(a.gen_dir) / "struct_token.fasta")
    ids = sorted(set(aa) & set(st))
    hit = len(ids) / max(len(aa), 1)
    print(f"  aa {len(aa):,} | struct {len(st):,} | 两边都有 {len(ids):,} = {hit:.1%}", flush=True)
    assert hit > 0.99, "★两个 fasta 的 id 对不上, 不能继续"
    ids = [x for i, x in enumerate(ids) if i % a.num_shards == a.shard]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    n_ok = n_bad = n_skip = 0
    steps = []
    for i, idx in enumerate(ids, 1):
        f = out / f"{idx}.pdb"
        if f.exists() and f.stat().st_size > 0:
            n_skip += 1
            continue
        s, t = aa[idx], st[idx]
        ntok = len([x for x in t.split(",") if x.strip()])
        if ntok != len(s):
            print(f"  ★{idx}: token {ntok} != 序列 {len(s)}, 跳过", flush=True)
            n_bad += 1
            continue
        try:
            aat, tok = stok.string_to_tensor(s, t)
            with torch.no_grad():
                dec = stok.detokenize(tok.cuda(),
                                      res_mask=torch.ones(1, ntok, dtype=torch.float32, device="cuda"))
            pos = dec["atom37_positions"][0].float().cpu().numpy()
            ca = pos[:, 1]                     # 这里的 1 是 atom37 的 CA, 已在本地核过
            step = float(np.median(np.linalg.norm(ca[1:] - ca[:-1], axis=-1)))
            if not (3.6 <= step <= 4.0):
                print(f"  ★{idx}: 相邻CA中位 {step:.2f}Å 不在 3.6-4.0, 作废", flush=True)
                n_bad += 1
                continue
            steps.append(step)
            # ★output_to_pdb 要一个 header 列表当文件名 —— 直接给 idx,
            #   文件名就是 id, 不用事后猜哪个文件是刚写的(少一个出错点)。
            d1 = {k: v[:1] for k, v in dec.items()}
            d1["header"] = [idx]
            stok.output_to_pdb(d1, str(out))
            assert f.exists() and f.stat().st_size > 0, f"★{idx}.pdb 没写出来"
            n_ok += 1
        except Exception as e:
            print(f"  ★{idx}: 解码失败 {type(e).__name__}: {e}", flush=True)
            n_bad += 1
        if i % 50 == 0:
            print(f"  {i}/{len(ids)} ok={n_ok} 作废={n_bad} 跳过={n_skip}", flush=True)
    print(f"\n  分片 {a.shard}/{a.num_shards}: ok {n_ok} | 作废 {n_bad} | 已存在跳过 {n_skip}", flush=True)
    if steps:
        print(f"  相邻CA距离中位的中位 {np.median(steps):.2f}Å (全部在 3.6-4.0 闸内)", flush=True)


if __name__ == "__main__":
    main()
