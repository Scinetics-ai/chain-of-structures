# -*- coding: utf-8 -*-
"""冒烟: 本地能不能加载 + 无条件生成一条 + 解码出合法几何。
★用几何把关不是退出码(feedback_gate_generation_on_geometry_not_exit_code: EXIT=0 也能塌成球):
  判据是相邻 CA 距离中位必须落在 3.6-4.0 Å。
★CA 的 atom37 下标必须从 residue_constants 取, 不能假定是 1(feedback_esmfold_scrmsd_pipeline)。
"""
import numpy as np
import torch

from dplm_local import load

m = load()
print(f"  token id: aa_mask={m.aa_mask_id} struct_mask={m.struct_mask_id} "
      f"aa_bos={m.aa_bos_id} struct_bos={m.struct_bos_id}", flush=True)

L, B = 200, 2
dev = "cuda"
# 坑4: 两半等长
struct = torch.full((B, L + 2), m.struct_mask_id, dtype=torch.long, device=dev)
struct[:, 0] = m.struct_bos_id; struct[:, -1] = m.struct_eos_id
aa = torch.full((B, L + 2), m.aa_mask_id, dtype=torch.long, device=dev)
aa[:, 0] = m.aa_bos_id; aa[:, -1] = m.aa_eos_id
x = torch.cat([struct, aa], dim=1)
print(f"  输入 {tuple(x.shape)}  (两半各 {L+2})", flush=True)

with torch.no_grad():
    out = m.generate(input_tokens=x, max_iter=100, temperature=1.0, unmasking_strategy="stochastic1.0",
                     sampling_strategy="annealing@2.2:1.0")
print(f"  generate 返回 {type(out).__name__}; 键 {list(out.keys()) if isinstance(out, dict) else '-'}", flush=True)
tok = out["output_tokens"] if isinstance(out, dict) else out
feat = out["final_struct_feature"] if isinstance(out, dict) else None
print(f"  output_tokens {tuple(tok.shape)}  final_struct_feature "
      f"{tuple(feat.shape) if feat is not None else None}", flush=True)

from byprot.utils.protein import residue_constants as rc
ca = list(rc.atom_types).index("CA")
print(f"  atom37 里 CA 的下标 = {ca}", flush=True)

import geom as G
pos, pl = G.decode(m, feat, L)
for i in range(B):
    c = pos[i, :L, ca]
    step = np.linalg.norm(c[1:] - c[:-1], axis=-1)
    med = float(np.median(step))
    print(f"  样本{i}: 相邻CA中位 {med:.2f}Å  plddt中位 {np.median(pl[i,:L]):.0f}", flush=True)
    assert 3.6 <= med <= 4.0, f"★相邻 CA 距离中位 {med:.2f}Å 不在 3.6-4.0 ⇒ 生成塌了, 停止"
print("  ✅ 本地 H100 上 DPLM-bit 跑通, 几何合法", flush=True)
