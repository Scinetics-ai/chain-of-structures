#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from biotite.structure import get_residue_starts
from biotite.structure.io.pdb import PDBFile
from peft.peft_model import PeftModel
from tqdm.auto import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DISCO_ROOT = Path("/data/workspace/caiqingran/protein_design/DISCO_repo")
if DISCO_ROOT.exists():
    sys.path.insert(0, str(DISCO_ROOT))

# _FLA_PREIMPORT: byprot pulls deepspeed, which wraps torch.compile so that `model` becomes
# positional while functools.wraps keeps the original signature visible. fla evaluates
# `torch.compile(fullgraph=True)` at module scope, so it has to be imported while torch.compile is
# still the real one, or Qwen3.5 cannot be loaded later. Costs about a second; harmless when the
# text model is something else.
try:
    import fla.modules as _fla_modules  # noqa: F401
except Exception as _fla_err:  # fla absent, or a genuinely broken install
    _fla_modules = None

from byprot.models.dplm2 import DPLM2Bit
from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.models.utils import sample_from_categorical, stochastic_sample_from_categorical, top_k_top_p_filtering
from generate_dplm2 import save_results


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


def allreduce_grads(params, world_size: int):
    # Every rank must reduce the same parameters in the same order. A module skipped on one rank
    # only - any `if <data condition>: return` inside a submodule - breaks that and NCCL hangs for
    # ten minutes before reporting a tensor size that has nothing to do with the cause. One scalar
    # collective per step converts that into an immediate, named failure.
    if world_size > 1:
        _n = torch.tensor(float(sum(1 for p in params if p.grad is not None)),
                          device=next((p.device for p in params), "cpu"))
        _mx = _n.clone(); _mn = _n.clone()
        dist.all_reduce(_mx, op=dist.ReduceOp.MAX)
        dist.all_reduce(_mn, op=dist.ReduceOp.MIN)
        if float(_mx) != float(_mn):
            raise RuntimeError(
                f"gradient desync: ranks disagree on how many params have grads "
                f"(min {float(_mn):.0f}, max {float(_mx):.0f}, this rank {float(_n):.0f}). "
                f"A submodule ran on some ranks and not others.")

    if world_size <= 1:
        return
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def broadcast_params(params, world_size: int):
    if world_size <= 1:
        return
    for p in params:
        dist.broadcast(p.data, src=0)


def reduce_float(value: float, device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t.div_(world_size)
    return float(t.item())


def text_hash(text: str, dim: int = 4096) -> torch.Tensor:
    text = (text or "").lower()
    toks = text.replace("|", " ").replace(",", " ").replace(";", " ").split()
    feats = torch.zeros(dim, dtype=torch.float32)
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        feats[h % dim] += 1.0
        if len(tok) >= 3:
            for i in range(len(tok) - 2):
                gram = tok[i : i + 3]
                h = int(hashlib.md5(("c:" + gram).encode("utf-8")).hexdigest()[:8], 16)
                feats[h % dim] += 0.25
    return feats / feats.norm().clamp_min(1.0)


class TextRetrievalConditioner(nn.Module):
    def __init__(self, text_dim: int, hidden: int):
        super().__init__()
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 2),
        )
        self.ret_aa = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.ret_st = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.text_scale = nn.Parameter(torch.tensor(0.05))
        self.ret_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, text_feats, motif_aa_emb, motif_st_emb):
        aa_t, st_t = self.text(text_feats).chunk(2, dim=-1)
        aa = aa_t * self.text_scale + self.ret_aa(motif_aa_emb) * self.ret_scale
        st = st_t * self.text_scale + self.ret_st(motif_st_emb) * self.ret_scale
        return aa, st


class CodeFPCrossConditioner(nn.Module):
    """CodeFP-style functional/motif cross-attention conditioner.

    The target sequence+structure tokens are queries.  The EC-comment token and
    retrieved local motif tokens are key/value context.
    """

    def __init__(self, text_dim: int, hidden: int, num_heads: int, num_text_tokens: int = 1):
        super().__init__()
        self.num_text_tokens = int(num_text_tokens)
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * self.num_text_tokens),
        )
        self.text_ln = nn.LayerNorm(hidden)
        self.text_pos = nn.Parameter(torch.zeros(self.num_text_tokens, hidden))
        self.motif_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.query_ln = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.aa_type = nn.Parameter(torch.zeros(hidden))
        self.st_type = nn.Parameter(torch.zeros(hidden))
        self.text_type = nn.Parameter(torch.zeros(hidden))
        self.cross_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, text_feats, motif_aa_emb, motif_st_emb):
        text_tok = self.text(text_feats).view(text_feats.shape[0], self.num_text_tokens, -1)
        text_tok = self.text_ln(text_tok + self.text_pos[None, :, :]).mean(dim=1)
        aa = self.motif_proj(motif_aa_emb) + text_tok
        st = self.motif_proj(motif_st_emb) + text_tok
        return aa * 0.05, st * 0.05

    def apply_cross(self, emb, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, emb_layer, input_mask=None):
        text_tok = self.text(text_feats).view(emb.shape[0], self.num_text_tokens, emb.shape[-1])
        text_tok = self.text_ln(text_tok + self.text_pos[None, :, :] + self.text_type[None, None, :])
        st_tok = emb_layer(motif_st_ids) + self.st_type[None, None, :]
        aa_tok = emb_layer(motif_aa_ids) + self.aa_type[None, None, :]
        motif_tok = self.motif_proj(torch.cat([st_tok, aa_tok], dim=1))
        context = torch.cat([text_tok, motif_tok], dim=1)
        text_mask = torch.zeros((emb.shape[0], self.num_text_tokens), dtype=torch.bool, device=emb.device)
        context_mask = torch.cat([text_mask, ~motif_st_mask, ~motif_aa_mask], dim=1)
        q = self.query_ln(emb)
        out, _ = self.cross_attn(q, context, context, key_padding_mask=context_mask, need_weights=False)
        return emb + self.cross_scale * out


class MMDiTTextConditioner(nn.Module):
    """Lightweight MMDiT-style joint text/protein token conditioner.

    Text tokens and protein tokens are concatenated and updated by joint
    self-attention before the frozen/unfrozen DPLM2 trunk sees the protein
    embeddings.  Only the protein stream is returned.
    """

    def __init__(self, text_dim: int, hidden: int, num_heads: int, num_text_tokens: int = 4, num_layers: int = 1):
        super().__init__()
        self.num_text_tokens = int(num_text_tokens)
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * self.num_text_tokens),
        )
        self.text_pos = nn.Parameter(torch.zeros(self.num_text_tokens, hidden))
        self.blocks = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln_attn": nn.LayerNorm(hidden),
                        "attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                        "ln_ff": nn.LayerNorm(hidden),
                        "ff": nn.Sequential(
                            nn.Linear(hidden, hidden * 4),
                            nn.GELU(),
                            nn.Linear(hidden * 4, hidden),
                        ),
                    }
                )
            )
        self.joint_scale = nn.Parameter(torch.tensor(0.2))

    def apply_cross(self, emb, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, emb_layer, input_mask=None):
        bsz, _, hidden = emb.shape
        text_tok = self.text(text_feats).view(bsz, self.num_text_tokens, hidden)
        text_tok = text_tok + self.text_pos[None, :, :]
        x = torch.cat([text_tok, emb], dim=1)
        key_padding_mask = None
        if input_mask is not None:
            text_mask = torch.zeros((bsz, self.num_text_tokens), dtype=torch.bool, device=emb.device)
            key_padding_mask = torch.cat([text_mask, ~input_mask], dim=1)
        for block in self.blocks:
            h = block["ln_attn"](x)
            attn_out, _ = block["attn"](h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
            x = x + block["ff"](block["ln_ff"](x))
        protein = x[:, self.num_text_tokens :, :]
        return emb + self.joint_scale * (protein - emb)


class MMDiTPerTokenConditioner(nn.Module):
    """Joint attention between the protein and the text's own token sequence.

    Differs from MMDiTTextConditioner only in what enters the text stream: a projected sequence of
    hidden states with a padding mask, instead of one pooled vector fanned out into a fixed four
    tokens. Everything downstream - the joint attention, the gated residual into the protein
    embeddings - is unchanged, so the two can be compared directly.
    """

    def __init__(self, text_dim: int, hidden: int, num_heads: int, max_text_tokens: int = 576,
                 num_layers: int = 1):
        super().__init__()
        self.max_text_tokens = int(max_text_tokens)
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.text_norm = nn.LayerNorm(hidden)
        self.text_pos = nn.Parameter(torch.zeros(self.max_text_tokens, hidden))
        nn.init.normal_(self.text_pos, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln_attn": nn.LayerNorm(hidden),
                        "attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                        "ln_ff": nn.LayerNorm(hidden),
                        "ff": nn.Sequential(
                            nn.Linear(hidden, hidden * 4),
                            nn.GELU(),
                            nn.Linear(hidden * 4, hidden),
                        ),
                    }
                )
            )
        self.joint_scale = nn.Parameter(torch.tensor(0.2))

    def apply_cross(self, emb, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask,
                    emb_layer, input_mask=None, text_mask=None):
        """text_feats: (B, N, text_dim); text_mask: (B, N) with True on real tokens."""
        bsz, _, hidden = emb.shape
        if text_feats.dim() == 2:                      # tolerate a pooled vector, as one token
            text_feats = text_feats[:, None, :]
        n = min(text_feats.shape[1], self.max_text_tokens)
        text_feats = text_feats[:, -n:, :]             # keep the end: the residues are named late
        if text_mask is None:
            text_mask = torch.ones(bsz, n, dtype=torch.bool, device=emb.device)
        else:
            text_mask = text_mask[:, -n:]

        text_tok = self.text_norm(self.text(text_feats.to(emb.dtype)))
        text_tok = text_tok + self.text_pos[None, :n, :]
        # a padded slot must not leak a value into attention even though it is masked as a key
        text_tok = text_tok * text_mask[..., None].to(text_tok.dtype)

        x = torch.cat([text_tok, emb], dim=1)
        pad = torch.cat(
            [~text_mask,
             ~input_mask if input_mask is not None
             else torch.zeros(bsz, emb.shape[1], dtype=torch.bool, device=emb.device)],
            dim=1,
        )
        # a row that is entirely padding makes softmax undefined; there is always at least one real
        # text token, so only guard the degenerate case
        pad[pad.all(dim=1), 0] = False
        for block in self.blocks:
            h = block["ln_attn"](x)
            attn_out, _ = block["attn"](h, h, h, key_padding_mask=pad, need_weights=False)
            x = x + attn_out
            x = x + block["ff"](block["ln_ff"](x))
        protein = x[:, n:, :]
        return emb + self.joint_scale * (protein - emb)


class CrossModalNCE(nn.Module):
    """Two projections and a temperature, so a protein state and a text state can be compared.

    The EC head asked which of 52 classes; this asks which of the 12 documents in the gathered batch,
    which class-level information cannot answer. Both projections are small - the cost is one 12x12
    similarity matrix per round.
    """

    def __init__(self, prot_dim: int, text_dim: int, proj: int = 256, init_scale: float = 2.6593):
        super().__init__()
        self.p = nn.Sequential(nn.LayerNorm(2 * prot_dim), nn.Linear(2 * prot_dim, 4 * proj),
                               nn.GELU(), nn.Linear(4 * proj, proj))
        self.t = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 4 * proj),
                               nn.GELU(), nn.Linear(4 * proj, proj))
        # ln(1/0.07), CLIP's initial temperature; clamped below so it cannot run away
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, hp, ht):
        zp = F.normalize(self.p(hp), dim=-1)
        zt = F.normalize(self.t(ht), dim=-1)
        return zp, zt, self.logit_scale.clamp(max=4.6052).exp()


def load_text_bank(path, device, rows=None):
    """Frozen stage-1 embeddings, indexed so length-matched negatives are a slice."""
    d = torch.load(path, map_location="cpu")
    z = d["z"].float().to(device)
    ln = d["len"].to(device)
    order = torch.argsort(ln)
    bank = {"pos": {k: i for i, k in enumerate(d["keys"])},
            "z": z, "len": ln, "order": order, "slen": ln[order]}
    if rows is not None:
        # class id per bank row, so same-class negatives are a lookup rather than a scan
        _ec_of = {}
        for r in rows:
            v = r.get("ec_list") or r.get("target_ec") or r.get("ec") or ""
            e = v if isinstance(v, str) else (v[0] if v else "")
            if e:
                _ec_of[r["idx"]] = e
        by_ec = {}
        for k, i in bank["pos"].items():
            e = _ec_of.get(k)
            if e:
                by_ec.setdefault(e, []).append(i)
        bank["by_ec"] = {e: torch.tensor(v, device=ln.device) for e, v in by_ec.items()}
        bank["ec_of_row"] = {i: e for e, v in by_ec.items() for i in v}
        _sizes = sorted((len(v) for v in by_ec.values()), reverse=True)
        print(f"[NCE] EC index: {len(by_ec)} classes over {sum(_sizes)} rows; "
              f"largest {_sizes[0] if _sizes else 0}, median {_sizes[len(_sizes)//2] if _sizes else 0}",
              flush=True)
    return bank


def _nce_candidates(anchor_row, k, tol, generator=None):
    """k negatives for this anchor: same EC class when asked and the class is big enough, else
    length-matched, which is what this always did."""
    b = NCE_BANK
    _hard = None
    if NCE_SAME_EC and "by_ec" in b:
        _e = b["ec_of_row"].get(int(anchor_row))
        _pool = b["by_ec"].get(_e) if _e else None
        if _pool is not None and _pool.numel() >= NCE_EC_MIN:
            # every distinct member of the class, minus the anchor; classes are small (median 3)
            # so sampling with replacement here would mostly repeat one or two rows
            _p = _pool[_pool != anchor_row]
            if _p.numel() > k:
                _p = _p[torch.randperm(_p.numel(), device=_p.device)[:k]]
            _hard = _p
            if _hard.numel() >= k:
                return _hard[:k]
    v = b["len"][anchor_row]
    lo = int(torch.searchsorted(b["slen"], v - tol, right=False))
    hi = int(torch.searchsorted(b["slen"], v + tol, right=True))
    if hi - lo < k + 1:                       # widen to the nearest ones rather than repeat
        c = (b["len"] - v).abs().argsort()[:k + 1]
        return c[c != anchor_row][:k]
    pick = torch.randint(lo, hi, (k + 4,), device=b["order"].device)
    cand = b["order"][pick]
    cand = cand[cand != anchor_row]
    if cand.numel() < k:                      # astronomically unlikely, but never silently short
        cand = torch.cat([cand, cand])[:k]
    if _hard is not None and _hard.numel():
        # hard negatives first, then top up with length-matched ones to reach k
        _mask = ~torch.isin(cand, _hard)
        cand = torch.cat([_hard, cand[_mask]])
    return cand[:k]


def _nce_gather(x, differentiable: bool):
    """Concatenate this tensor across ranks. The local slice keeps its graph when asked."""
    ws, rk = NCE_WS[0], NCE_RANK[0]
    if ws <= 1:
        return x
    xs = [torch.zeros_like(x) for _ in range(ws)]
    dist.all_gather(xs, x.detach().contiguous())
    if differentiable:
        xs[rk] = x
    return torch.cat(xs, 0)


def cross_modal_nce(hidden, x, tf, tmask, st_len, pad_id, idx_list):
    """Can this protein state pick its own design request out of same-length alternatives?

    The candidate set is the record's own stage-1 embedding plus NCE_K others of the same length,
    all read from a frozen bank, and the answer's position is random. Chance is 1/(1+K) with no
    residual cue: not length, because every candidate matches; not recency, because the bank never
    moves; not position, because the slot is drawn uniformly.
    """
    valid = x.ne(pad_id).to(hidden.dtype)
    hs, ha = hidden.chunk(2, dim=1)
    vs, va = valid.chunk(2, dim=1)
    ps = (hs * vs[..., None]).sum(1) / vs.sum(1, keepdim=True).clamp_min(1.0)
    pa = (ha * va[..., None]).sum(1) / va.sum(1, keepdim=True).clamp_min(1.0)
    hp = torch.cat([ps, pa], dim=-1)

    rows = [NCE_BANK["pos"].get(str(i), -1) for i in idx_list]
    keep = [b for b, r in enumerate(rows) if r >= 0]
    if not keep:
        # Every parameter must take part on every rank: allreduce_grads reduces only params whose
        # grad is not None, so skipping the heads here and not there is a desync. Push one row
        # through and multiply the loss by zero - same trick as the injector's empty mask.
        z = sum((p_ * 0.0).sum() for p_ in NCE_HEADS.parameters()) + hp.sum() * 0.0
        return z, float("nan"), 1 + NCE_K, 1

    dev = hidden.device
    cand_rows, labels = [], []
    for b in keep:
        neg = _nce_candidates(rows[b], NCE_K, NCE_TOL)
        slot = int(torch.randint(0, NCE_K + 1, (1,)).item())
        r = torch.cat([neg[:slot],
                       torch.tensor([rows[b]], device=neg.device, dtype=neg.dtype),
                       neg[slot:]])
        cand_rows.append(r)
        labels.append(slot)
    cand_rows = torch.stack(cand_rows).to(dev)                     # (B, 1+K)
    lab = torch.tensor(labels, device=dev, dtype=torch.long)

    ht = NCE_BANK["z"][cand_rows.reshape(-1)]                      # (B*(1+K), D), frozen
    zp = F.normalize(NCE_HEADS.p(hp[keep].float()), dim=-1)
    zt = F.normalize(NCE_HEADS.t(ht), dim=-1).view(len(keep), NCE_K + 1, -1)
    scale = NCE_HEADS.logit_scale.clamp(max=4.6052).exp()
    logits = scale * torch.einsum("bd,bkd->bk", zp, zt)
    loss = F.cross_entropy(logits, lab)
    acc = float((logits.argmax(1) == lab).float().mean().detach())
    return loss, acc, int(logits.shape[1]), 1


class MMDiTLogitConditioner(MMDiTTextConditioner):
    """MMDiT input conditioning plus a direct text-conditioned logit residual.

    The previous EC auxiliary head proved that EC information can be decoded
    from pooled hidden states, but it did not force the denoising head to use
    that information.  This module makes the condition affect the actual token
    CE by adding a small residual to every sequence/structure logit.
    """

    def __init__(
        self,
        text_dim: int,
        hidden: int,
        num_heads: int,
        vocab_size: int,
        num_text_tokens: int = 4,
        num_layers: int = 1,
        rank: int = 256,
        init_scale: float = 0.02,
    ):
        super().__init__(text_dim, hidden, num_heads, num_text_tokens, num_layers)
        self.logit_text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.logit_ln = nn.LayerNorm(hidden)
        self.logit_down = nn.Linear(hidden, int(rank))
        self.logit_up = nn.Linear(int(rank), vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def apply_logits(self, logits, hidden_states, text_feats, st_len):
        text_vec = self.logit_text(text_feats)[:, None, :]
        h = self.logit_ln(hidden_states + text_vec)
        residual = self.logit_up(F.gelu(self.logit_down(h)))
        return logits + self.logit_scale * residual


class DeepTextCrossConditioner(nn.Module):
    """Per-layer EC-comment cross-attention adapters.

    This is the no-retrieval/no-LSFS analogue of CodeFP's deep conditioning:
    protein sequence+structure hidden states query text tokens at selected
    DPLM2 layers.  The base model can remain frozen, so the text adapter learns
    to speak in DPLM2's native hidden space instead of perturbing only input
    embeddings.
    """

    is_deep_text_cross = True

    def __init__(self, text_dim: int, hidden: int, num_heads: int, num_layers: int, num_text_tokens: int = 4, selected_layers: int = 8):
        super().__init__()
        self.num_text_tokens = int(num_text_tokens)
        self.num_layers = int(num_layers)
        if selected_layers <= 0 or selected_layers >= num_layers:
            layer_ids = list(range(num_layers))
        else:
            layer_ids = list(range(num_layers - int(selected_layers), num_layers))
        self.layer_ids = set(layer_ids)
        self.layer_to_adapter = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * self.num_text_tokens),
        )
        self.text_ln = nn.LayerNorm(hidden)
        self.text_pos = nn.Parameter(torch.zeros(self.num_text_tokens, hidden))
        self.adapters = nn.ModuleList()
        self.scales = nn.ParameterList()
        for _ in layer_ids:
            self.adapters.append(
                nn.ModuleDict(
                    {
                        "q_ln": nn.LayerNorm(hidden),
                        "ctx_ln": nn.LayerNorm(hidden),
                        "attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                    }
                )
            )
            self.scales.append(nn.Parameter(torch.tensor(0.1)))

    def text_tokens(self, text_feats):
        bsz = text_feats.shape[0]
        tok = self.text(text_feats).view(bsz, self.num_text_tokens, -1)
        return self.text_ln(tok + self.text_pos[None, :, :])

    def apply_layer(self, hidden_states, layer_idx: int, text_feats, input_mask=None):
        adapter_idx = self.layer_to_adapter.get(int(layer_idx))
        if adapter_idx is None:
            return hidden_states
        block = self.adapters[adapter_idx]
        context = block["ctx_ln"](self.text_tokens(text_feats))
        query = block["q_ln"](hidden_states)
        out, _ = block["attn"](query, context, context, need_weights=False)
        return hidden_states + self.scales[adapter_idx] * out


class DeepPerTokenCrossConditioner(nn.Module):
    """Per-layer cross-attention onto the real text token sequence.

    DeepTextCrossConditioner expands one pooled vector into `num_text_tokens` learned slots, so it
    cannot take the (B, N, text_dim) sequence the per-token path produces. This keeps every text
    token - 243 to 357 per stage, measured over the corpus - and lets the protein hidden states
    query them again at each of the last `selected_layers` DPLM2 layers, instead of once before
    layer 0 after which the text is no longer in the graph at all.

    The window is right-aligned, matching qwen_stage_states, which keeps the tail of the span
    because the residues are named late; the positional table is therefore indexed from the end too,
    so a token that is 5-from-last always gets the same position however long the span was.
    """

    is_deep_text_cross = True

    def __init__(self, text_dim: int, hidden: int, num_heads: int, num_layers: int,
                 selected_layers: int = 8, max_text_tokens: int = 576,
                 n_ec: int = 0, vocab_size: int = 0):
        super().__init__()
        self.max_text_tokens = int(max_text_tokens)
        self.n_ec = int(n_ec)
        if self.n_ec > 0 and vocab_size > 0:
            # text -> class, supervised. The mapping is easy - a bag-of-words retriever gets 92.9% -
            # so the point of the supervision is not difficulty but guaranteeing the signal exists.
            self.ec_text = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.n_ec),
            )
            self.ec_emb = nn.Embedding(self.n_ec, hidden)
            nn.init.normal_(self.ec_emb.weight, std=0.02)
            self.logit_ln = nn.LayerNorm(hidden)
            self.logit_down = nn.Linear(hidden, DEEP_LOGIT_RANK)
            self.logit_up = nn.Linear(DEEP_LOGIT_RANK, int(vocab_size), bias=False)
            nn.init.zeros_(self.logit_up.weight)      # starts as an exact no-op
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        if selected_layers <= 0 or selected_layers >= num_layers:
            layer_ids = list(range(num_layers))
        else:
            layer_ids = list(range(num_layers - int(selected_layers), num_layers))
        self.layer_ids = set(layer_ids)
        self.layer_to_adapter = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.text_ln = nn.LayerNorm(hidden)
        self.text_pos = nn.Parameter(torch.zeros(self.max_text_tokens, hidden))
        self.adapters = nn.ModuleList()
        self.scales = nn.ParameterList()
        for _ in layer_ids:
            self.adapters.append(
                nn.ModuleDict(
                    {
                        "q_ln": nn.LayerNorm(hidden),
                        "ctx_ln": nn.LayerNorm(hidden),
                        "attn": nn.MultiheadAttention(hidden, num_heads, batch_first=True),
                    }
                )
            )
            # ControlNet's zero-convolution: the adapter's output projection starts at zero, so at
            # init it contributes exactly nothing and the run begins bit-identical to the un-adapted
            # model. d(loss)/d(out_proj) is still non-zero, so the projection grows first and pulls
            # the rest of the adapter in behind it. A small scalar gate was tried instead and failed:
            # it both injected noise from step one and divided the adapter's own gradient by itself,
            # so the gates collapsed 0.100 -> 0.015 and stayed there.
            nn.init.zeros_(self.adapters[-1]["attn"].out_proj.weight)
            if self.adapters[-1]["attn"].out_proj.bias is not None:
                nn.init.zeros_(self.adapters[-1]["attn"].out_proj.bias)
            # kept as the per-layer read-out of how much text each layer uses, no longer a gate that
            # has to grow before the adapter can learn
            self.scales.append(nn.Parameter(torch.tensor(1.0),
                                            requires_grad=not DEEP_FIXED_SCALE))

    def text_tokens(self, text_feats):
        n = min(text_feats.shape[1], self.max_text_tokens)
        tok = self.text(text_feats[:, -n:, :])
        return self.text_ln(tok + self.text_pos[None, -n:, :])

    def ec_logits(self, text_feats, text_mask=None):
        """Class logits from the text alone, so the CE supervises the text pathway and nothing else."""
        if getattr(self, "n_ec", 0) <= 0:
            return None
        if text_feats.dim() == 2:
            pooled = text_feats
        elif text_mask is not None:
            m = text_mask[..., None].to(text_feats.dtype)
            pooled = (text_feats * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            pooled = text_feats.mean(1)
        return self.ec_text(pooled)

    def apply_logits(self, logits, hidden_states, text_feats, st_len, text_mask=None):
        """Bias every amino-acid logit by the class the text names.

        The class embedding is a soft mixture over the predicted distribution rather than an argmax,
        so the path stays differentiable and an uncertain prediction contributes proportionally
        rather than committing to one class.
        """
        if getattr(self, "n_ec", 0) <= 0 or DEEP_EC_LOGIT_OFF:
            return logits
        el = self.ec_logits(text_feats, text_mask)
        p = torch.softmax(el.float(), dim=-1).to(hidden_states.dtype)
        cls = (p @ self.ec_emb.weight)[:, None, :]
        h = self.logit_ln(hidden_states + cls)
        return logits + self.logit_scale * self.logit_up(F.gelu(self.logit_down(h)))

    def apply_layer(self, hidden_states, layer_idx: int, text_feats, input_mask=None, text_mask=None):
        adapter_idx = self.layer_to_adapter.get(int(layer_idx))
        if adapter_idx is None:
            return hidden_states
        if text_feats.dim() == 2:                      # tolerate a pooled vector, as one token
            text_feats = text_feats[:, None, :]
        block = self.adapters[adapter_idx]
        ctx = block["ctx_ln"](self.text_tokens(text_feats.to(hidden_states.dtype)))
        n = ctx.shape[1]
        kpm = None
        if text_mask is not None:
            kpm = ~text_mask[:, -n:]
            if bool(kpm.all(dim=1).any()):
                # an all-padding row makes softmax undefined; qwen_stage_states already guarantees
                # one real token, so this only covers the degenerate case
                kpm = kpm.clone()
                kpm[kpm.all(dim=1), -1] = False
        out, _ = block["attn"](block["q_ln"](hidden_states), ctx, ctx,
                               key_padding_mask=kpm, need_weights=False)
        return hidden_states + self.scales[adapter_idx] * out


class MotifLabelHead(nn.Module):
    def __init__(self, hidden: int, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_labels),
        )

    def forward(self, x):
        return self.net(x)


class ECLabelHead(nn.Module):
    def __init__(self, hidden: int, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_labels),
        )

    def forward(self, x):
        return self.net(x)


def load_json(path):
    return json.load(open(path))


def parse_ecs(row):
    value = row.get("ec_list") or row.get("target_ec") or row.get("ec") or ""
    return [x.strip() for x in str(value).replace(",", ";").split(";") if x.strip()]


def ec_prefix(ec: str, depth: int) -> str:
    return ".".join(str(ec).split(".")[:depth])


def ec_label_for_row(row: dict, depth: int, label_to_id: dict[str, int] | None = None) -> str | None:
    for ec in parse_ecs(row):
        parts = str(ec).split(".")
        if len(parts) < depth:
            continue
        label = f"EC{depth}:{ec_prefix(ec, depth)}"
        if label_to_id is None or label in label_to_id:
            return label
    return None


def build_ec_label_vocab(rows: list[dict], depth: int) -> list[str]:
    labels = set()
    for row in rows:
        label = ec_label_for_row(row, depth)
        if label is not None:
            labels.add(label)
    return sorted(labels)


def attach_ec_aux_labels(rows: list[dict], label_to_id: dict[str, int] | None, depth: int) -> list[dict]:
    if not label_to_id:
        return rows
    out = []
    for row in rows:
        rr = dict(row)
        label = ec_label_for_row(rr, depth, label_to_id)
        rr["ec_aux_label"] = label
        rr["ec_aux_label_id"] = int(label_to_id[label]) if label is not None else -1
        out.append(rr)
    return out


def select_rows(rows, ids=None, limit=0, min_len=1, max_len=512):
    if ids is not None:
        row_by_id = {str(r["idx"]): r for r in rows}
        rows = [row_by_id[str(i)] for i in ids if str(i) in row_by_id]
    out = []
    for r in rows:
        if not r.get("native_pdb") or not r.get("comment"):
            continue
        length = int(r.get("length") or r.get("ca_count") or 0)
        if length < min_len or length > max_len:
            continue
        rr = dict(r)
        rr["idx"] = str(rr["idx"])
        out.append(rr)
        if limit and len(out) >= limit:
            break
    return out


def read_manifest(path: str) -> dict[str, dict]:
    manifest = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("label"):
                    manifest[row["label"]] = row
    return manifest


def pick_prototype(manifest: dict[str, dict], row: dict, max_depth: int = 3):
    ecs = parse_ecs(row)
    for depth in range(4, max_depth - 1, -1):
        for ec in ecs:
            label = f"EC{depth}:{ec_prefix(ec, depth)}"
            if label in manifest:
                proto = dict(manifest[label])
                proto["_picked_label"] = label
                proto["_picked_depth"] = depth
                return proto
    return None


def slice_residue_window(atom_array, start0: int, end0: int):
    starts = np.asarray(get_residue_starts(atom_array), dtype=np.int64)
    start0 = int(start0)
    end0 = min(int(end0), len(starts))
    if start0 < 0 or start0 >= len(starts) or end0 <= start0:
        raise IndexError(f"bad motif window start={start0} end={end0} residues={len(starts)}")
    atom_start = int(starts[start0])
    atom_end = int(starts[end0]) if end0 < len(starts) else len(atom_array)
    window = atom_array[atom_start:atom_end].copy()
    window.chain_id[:] = "A"
    return window


def write_pdb(atom_array, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(str(path))


def load_atom_array_from_pdb(path: str):
    pdb = PDBFile.read(str(path))
    return pdb.get_structure(model=1)


def load_from_pdb_for_dplm(pdb_path, process_chain):
    from byprot.datamodules.pdb_dataset import utils as du

    raw_chain_feats, metadata = du.process_pdb_file(str(pdb_path))
    chain_feats = process_chain(raw_chain_feats)
    chain_feats["pdb_name"] = metadata["pdb_name"]
    return chain_feats


def tokenize_pdbs(model, rows, cache_path: Path):
    by_idx = {}
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        by_idx = {str(r["idx"]): r for r in cached}
        has_rows = all(str(r["idx"]) in by_idx for r in rows)
        if has_rows:
            return [by_idx[str(r["idx"])] for r in rows]

    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn
    from byprot.utils import recursive_to

    device = next(model.parameters()).device
    stok = model.struct_tokenizer
    tokenized = []
    for i, r in enumerate(rows, 1):
        if str(r["idx"]) in by_idx:
            tokenized.append(by_idx[str(r["idx"])])
            continue
        feats = load_from_pdb_for_dplm(r["native_pdb"], process_chain=stok.process_chain)
        batch = recursive_to(collate_fn([feats]), device)
        struct_ids = stok.tokenize(batch["all_atom_positions"], batch["res_mask"], batch["seq_length"])
        struct_seq = stok.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
        aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
        tokenized.append({**r, "sequence": r.get("sequence") or aa_seq, "aa_seq_from_pdb": aa_seq, "struct_seq": struct_seq})
        if i % 50 == 0:
            print(f"tokenized native {i}/{len(rows)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized, cache_path)
    return tokenized


def tokenize_pdbs_distributed(model, rows, cache_path: Path, rank: int, world_size: int):
    if world_size <= 1 or cache_path.exists():
        return tokenize_pdbs(model, rows, cache_path)

    shard_path = cache_path.with_name(f"{cache_path.stem}.rank{rank:02d}{cache_path.suffix}")
    shard_rows = [r for i, r in enumerate(rows) if i % world_size == rank]
    tokenize_pdbs(model, shard_rows, shard_path)
    barrier(world_size)

    if rank == 0:
        merged = {}
        for shard_rank in range(world_size):
            part_path = cache_path.with_name(f"{cache_path.stem}.rank{shard_rank:02d}{cache_path.suffix}")
            for row in torch.load(part_path, map_location="cpu"):
                merged[str(row["idx"])] = row
        tokenized = [merged[str(r["idx"])] for r in rows]
        torch.save(tokenized, cache_path)
        print(f"merged tokenized cache {len(tokenized)} -> {cache_path}", flush=True)
    barrier(world_size)
    return tokenize_pdbs(model, rows, cache_path)


def tokenize_motif_pdb(model, pdb_path: Path):
    from byprot.datamodules.pdb_dataset import utils as du
    from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn
    from byprot.utils import recursive_to

    device = next(model.parameters()).device
    stok = model.struct_tokenizer
    feats = load_from_pdb_for_dplm(pdb_path, process_chain=stok.process_chain)
    batch = recursive_to(collate_fn([feats]), device)
    struct_ids = stok.tokenize(batch["all_atom_positions"], batch["res_mask"], batch["seq_length"])
    struct_seq = stok.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
    aa_seq = du.aatype_to_seq(batch["aatype"].cpu().tolist()[0])
    return aa_seq, struct_seq


def encode_motif_ids(model, aa_seq: str, struct_seq: str):
    tok = model.tokenizer
    aa_ids = tok.encode(aa_seq, add_special_tokens=False)
    st_ids = tok.encode("".join(struct_seq.split(",")), add_special_tokens=False)
    if not aa_ids or not st_ids:
        raise ValueError("empty motif token ids")
    return aa_ids, st_ids


def load_metadata_index(path: str):
    import pandas as pd

    meta = pd.read_parquet(path)
    by_idx = {str(r.idx): r for r in meta.itertuples(index=False)}
    by_uid = {str(r.uid): r for r in meta.itertuples(index=False)}
    return by_idx, by_uid


def read_anchor_map(path: str):
    anchor_path = Path(path)
    if not anchor_path.exists():
        return {}
    out = {}
    with anchor_path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[str(row.get("idx", ""))] = row
    return out


def anchor_positions(anchor: dict) -> list[int]:
    vals = []
    for p in anchor.get("positions", []) or []:
        try:
            vals.append(int(p))
        except Exception:
            pass
    if vals:
        return sorted(set(vals))
    for mr in anchor.get("mapped_residues", []) or []:
        try:
            vals.append(int(mr.get("resnum")))
        except Exception:
            pass
    return sorted(set(vals))


def target_anchor_windows(anchor_row: dict | None, length: int, radius: int, max_groups: int) -> list[list[int]]:
    if not anchor_row:
        return []
    scored = []
    for ai, anchor in enumerate(anchor_row.get("anchors", []) or []):
        pos = [p for p in anchor_positions(anchor) if 1 <= p <= length]
        if not pos:
            continue
        feature = str(anchor.get("feature_type", ""))
        weight = float(anchor.get("weight", 1.0) or 1.0)
        priority = (2 if feature == "ACT_SITE" else 1 if feature == "BINDING" else 0, weight, len(pos))
        scored.append((priority, ai, pos))
    scored.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2], x[1]))
    windows = []
    for _, _, pos in scored[:max_groups]:
        start0 = max(0, min(pos) - 1 - radius)
        end0 = min(length, max(pos) + radius)
        if end0 - start0 > 5:
            windows.append([int(start0 + 1), int(end0 + 1)])
    return windows


def build_retrieval_rows(
    model,
    rows,
    manifest,
    metadata_by_idx,
    metadata_by_uid,
    cache_path: Path,
    pdb_dir: Path,
    forbidden_ids: set[str],
    label_to_id: dict[str, int] | None = None,
    anchor_map: dict[str, dict] | None = None,
    anchor_radius: int = 12,
    max_anchor_groups: int = 4,
):
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        by_idx = {str(r["idx"]): r for r in cached}
        if all(str(r["idx"]) in by_idx for r in rows):
            return [by_idx[str(r["idx"])] for r in rows]

    proto_cache = {}
    out_rows = []
    missing = []
    leak = []
    for i, row in enumerate(rows, 1):
        proto = pick_prototype(manifest, row)
        if proto is None:
            missing.append(row["idx"])
            rr = dict(row)
            rr.update({"retrieval_label": None, "motif_aa_ids": [], "motif_st_ids": []})
            rr.update({"lsfs_windows": [], "lsfs_label_id": -1})
            out_rows.append(rr)
            continue

        label = proto["_picked_label"]
        med = proto["medoid"]
        med_idx = str(med["idx"])
        if med_idx in forbidden_ids:
            leak.append((row["idx"], label, med_idx))
        key = f"{label}|{med_idx}|{int(med['start0'])}|{int(med['end0'])}"
        if key not in proto_cache:
            med_meta = metadata_by_idx.get(med_idx) or metadata_by_uid.get(str(med.get("uid", "")))
            if med_meta is None:
                raise RuntimeError(f"missing medoid metadata for idx={med_idx} uid={med.get('uid','')}")
            atom_array = load_atom_array_from_pdb(str(med_meta.native_pdb))
            window = slice_residue_window(atom_array, int(med["start0"]), int(med["end0"]))
            safe = key.replace(":", "_").replace("|", "__").replace("/", "_")
            pdb_path = pdb_dir / f"{safe}.pdb"
            write_pdb(window, pdb_path)
            aa_seq, struct_seq = tokenize_motif_pdb(model, pdb_path)
            aa_ids, st_ids = encode_motif_ids(model, aa_seq, struct_seq)
            proto_cache[key] = {
                "label": label,
                "medoid_idx": med_idx,
                "medoid_uid": str(med.get("uid", "")),
                "start0": int(med["start0"]),
                "end0": int(med["end0"]),
                "support_n": int(proto.get("support_n", 0)),
                "motif_len": len(aa_seq),
                "motif_aa_seq": aa_seq,
                "motif_struct_seq": struct_seq,
                "motif_aa_ids": aa_ids,
                "motif_st_ids": st_ids,
                "motif_pdb": str(pdb_path),
            }
        rr = dict(row)
        rr.update({f"retrieval_{k}": v for k, v in proto_cache[key].items() if k not in {"motif_aa_ids", "motif_st_ids"}})
        rr["motif_aa_ids"] = proto_cache[key]["motif_aa_ids"]
        rr["motif_st_ids"] = proto_cache[key]["motif_st_ids"]
        windows = []
        if anchor_map is not None:
            windows = target_anchor_windows(anchor_map.get(str(row["idx"])), len(row["aa_seq_from_pdb"]), anchor_radius, max_anchor_groups)
        rr["lsfs_windows"] = windows
        rr["lsfs_label_id"] = int(label_to_id.get(label, -1)) if (label_to_id is not None and windows) else -1
        out_rows.append(rr)
        if i % 50 == 0:
            print(f"built retrieval {i}/{len(rows)}", flush=True)

    if leak:
        raise RuntimeError(f"retrieval medoid leakage into forbidden ids: {leak[:5]} total={len(leak)}")
    print(f"retrieval coverage {len(rows)-len(missing)}/{len(rows)} missing={len(missing)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_rows, cache_path)
    return out_rows


def build_no_retrieval_rows(rows, cache_path: Path, label_to_id: dict[str, int] | None = None, ec_hidden_depth: int = 3):
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        by_idx = {str(r["idx"]): r for r in cached}
        has_rows = all(str(r["idx"]) in by_idx for r in rows)
        has_labels = label_to_id is None or all(
            int(by_idx[str(r["idx"])].get("lsfs_label_id", -1)) >= 0 for r in rows if str(r["idx"]) in by_idx
        )
        if has_rows and has_labels:
            return [by_idx[str(r["idx"])] for r in rows]
    out_rows = []
    for row in rows:
        rr = dict(row)
        label = ec_label_for_row(rr, ec_hidden_depth, label_to_id)
        label_id = int(label_to_id[label]) if label is not None and label_to_id is not None else -1
        aa_len = len(rr.get("aa_seq_from_pdb") or rr.get("sequence") or "")
        # Full-residue hidden supervision: no retrieved motif/prototype is used.
        windows = [[1, aa_len + 1]] if label_id >= 0 and aa_len > 2 else []
        rr.update(
            {
                "retrieval_label": None,
                "retrieval_medoid_idx": None,
                "retrieval_start0": None,
                "retrieval_end0": None,
                "retrieval_support_n": 0,
                "retrieval_motif_len": 0,
                "motif_aa_ids": [],
                "motif_st_ids": [],
                "lsfs_windows": windows,
                "lsfs_label": label,
                "lsfs_label_id": label_id,
            }
        )
        out_rows.append(rr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_rows, cache_path)
    print(f"no retrieval rows {len(out_rows)}", flush=True)
    return out_rows


def encode_targets(model, rows, use_retrieval_prefix: bool = False):
    tok = model.tokenizer
    aa_rows = []
    st_rows = []
    aa_prefix_lens = []
    st_prefix_lens = []
    for r in rows:
        aa = r["aa_seq_from_pdb"]
        st = r["struct_seq"].split(",")
        aa_target = tok.encode(tok.aa_cls_token + aa + tok.aa_eos_token, add_special_tokens=False)
        st_target = tok.encode(tok.struct_cls_token + "".join(st) + tok.struct_eos_token, add_special_tokens=False)
        aa_prefix = list(r.get("motif_aa_ids") or []) if use_retrieval_prefix else []
        st_prefix = list(r.get("motif_st_ids") or []) if use_retrieval_prefix else []
        aa_prefix_lens.append(len(aa_prefix))
        st_prefix_lens.append(len(st_prefix))
        aa_rows.append(aa_prefix + aa_target)
        st_rows.append(st_prefix + st_target)

    max_st = max(len(x) for x in st_rows)
    max_aa = max(len(x) for x in aa_rows)
    st_ids = torch.full((len(rows), max_st), model.pad_id, dtype=torch.long)
    aa_ids = torch.full((len(rows), max_aa), model.pad_id, dtype=torch.long)
    for i, vals in enumerate(st_rows):
        st_ids[i, : len(vals)] = torch.tensor(vals, dtype=torch.long)
    for i, vals in enumerate(aa_rows):
        aa_ids[i, : len(vals)] = torch.tensor(vals, dtype=torch.long)
    target = torch.cat([st_ids, aa_ids], dim=1)
    input_ids = target.clone()

    special = {model.pad_id, model.aa_bos_id, model.aa_eos_id, model.struct_bos_id, model.struct_eos_id}
    st_len = st_ids.shape[1]
    loss_mask = torch.ones_like(target, dtype=torch.bool)
    for s in special:
        loss_mask &= target.ne(s)
    if use_retrieval_prefix:
        for i, pref in enumerate(st_prefix_lens):
            if pref:
                loss_mask[i, :pref] = False
        for i, pref in enumerate(aa_prefix_lens):
            if pref:
                loss_mask[i, st_len : st_len + pref] = False
    input_ids[:, :st_len] = input_ids[:, :st_len].masked_fill(loss_mask[:, :st_len], model.struct_mask_id)
    input_ids[:, st_len:] = input_ids[:, st_len:].masked_fill(loss_mask[:, st_len:], model.aa_mask_id)
    return (
        input_ids,
        target,
        loss_mask,
        st_len,
        torch.tensor(st_prefix_lens, dtype=torch.long),
        torch.tensor(aa_prefix_lens, dtype=torch.long),
    )


def pad_id_lists(rows, key, pad_id):
    max_len = max([len(r.get(key) or []) for r in rows] + [1])
    ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for i, r in enumerate(rows):
        values = r.get(key) or []
        if values:
            ids[i, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[i, : len(values)] = True
    return ids, mask


def pad_windows(rows):
    max_windows = max([len(r.get("lsfs_windows") or []) for r in rows] + [1])
    windows = torch.full((len(rows), max_windows, 2), -1, dtype=torch.long)
    mask = torch.zeros((len(rows), max_windows), dtype=torch.bool)
    labels = torch.full((len(rows),), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        vals = r.get("lsfs_windows") or []
        if vals and int(r.get("lsfs_label_id", -1)) >= 0:
            windows[i, : len(vals)] = torch.tensor(vals, dtype=torch.long)
            mask[i, : len(vals)] = True
            labels[i] = int(r.get("lsfs_label_id", -1))
    return windows, mask, labels


def pad_ec_aux_labels(rows):
    labels = torch.full((len(rows),), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        labels[i] = int(r.get("ec_aux_label_id", -1))
    return labels


def motif_mean_embeddings(model, aa_ids, aa_mask, st_ids, st_mask):
    emb_layer = model.net.esm.embeddings.word_embeddings
    aa_emb = emb_layer(aa_ids)
    st_emb = emb_layer(st_ids)
    aa_den = aa_mask.sum(dim=1, keepdim=True).clamp_min(1).to(aa_emb.dtype)
    st_den = st_mask.sum(dim=1, keepdim=True).clamp_min(1).to(st_emb.dtype)
    aa_mean = (aa_emb * aa_mask[..., None].to(aa_emb.dtype)).sum(dim=1) / aa_den
    st_mean = (st_emb * st_mask[..., None].to(st_emb.dtype)).sum(dim=1) / st_den
    return aa_mean, st_mean


def standard_aa_ids(model):
    ids = []
    tok_to_id = getattr(model.tokenizer, "_token_to_id", {})
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        if aa in tok_to_id:
            ids.append(int(tok_to_id[aa]))
    if not ids:
        raise RuntimeError("could not resolve standard amino-acid token ids")
    return ids


def apply_modality_vocab_mask(model, logits, st_len):
    vocab = logits.shape[-1]
    device = logits.device
    aa_allowed = torch.zeros(vocab, dtype=torch.bool, device=device)
    aa_allowed[torch.tensor(standard_aa_ids(model), device=device)] = True
    struct_allowed = torch.ones(vocab, dtype=torch.bool, device=device)
    struct_allowed[:33] = False
    for sid in model.special_token_list:
        if 0 <= int(sid) < vocab:
            aa_allowed[int(sid)] = False
            struct_allowed[int(sid)] = False
    logits[:, :st_len, ~struct_allowed] = -torch.inf
    logits[:, st_len:, ~aa_allowed] = -torch.inf
    return logits


# --- Bitwise Self-Correction (Infinity 2412.04431) -------------------------------------------
# Set from --bsc_prob. 0.0 => behaviour identical to the pre-BSC baseline.
BSC_PROB = 0.0


# ---------------- PHASE1: four independently-ablatable knobs (all default to OFF) --------------
P1_CORE_BOOST = 0.0        # 1a  extra multiplicative weight on catalytic-core positions
P1_NOISE_W = "constant"    # 1b  constant | linear | minsnr | logitnorm
P1_STAGE_MASK_P = 0.0      # 1c  probability a batch uses stage-ordered masking
P1_LAST_BITACC = (float('nan'), -1)
P1_CORE_POS = {}           # {idx: set(0-based positions)}  filled once at startup
P1_POCKET_POS = {}         # {idx: set(0-based first-shell positions)}
P1_BATCH_IDX = []          # record idx of the rows in the batch currently being processed
P1_WIRED = True


def p1_timestep_weight(t, num_timesteps, mode):
    """Per-sample loss weight as a function of the noise level."""
    tf = t.float()
    if mode == "linear":
        return (num_timesteps - (tf - 1)) / num_timesteps
    if mode == "minsnr":
        # masked diffusion has no continuous SNR; the mask ratio t/T plays its role, so the
        # SNR-like quantity is (1-r)/r. Min-SNR clamps the weight of the easy (low-noise) end.
        r = (tf / num_timesteps).clamp(1e-3, 1 - 1e-3)
        snr = (1.0 - r) / r
        return torch.minimum(snr, torch.full_like(snr, 5.0)) / 5.0
    if mode == "logitnorm":
        # SD3-style: concentrate weight on the middle of the trajectory
        r = (tf / num_timesteps).clamp(1e-3, 1 - 1e-3)
        x = torch.log(r / (1 - r))
        return torch.exp(-0.5 * x * x)
    return torch.ones_like(tf)


def p1_core_weight(idx_list, shape, device):
    """(B, L) multiplier that up-weights the catalytic-core positions of each row."""
    w = torch.ones(shape, device=device)
    if P1_CORE_BOOST <= 0.0 or not P1_CORE_POS:
        return w
    L = shape[1]
    for b, idx in enumerate(idx_list):
        for p in P1_CORE_POS.get(idx, ()):  # 0-based
            if 0 <= p < L:
                w[b, p] = 1.0 + P1_CORE_BOOST
    return w


def p1_stage_order_mask(target, model, idx_list, core_pos, pocket_pos):
    """Reveal in the data's own order (core first, scaffold last) instead of uniformly at random.

    Returns a boolean mask of positions to CORRUPT. The staged order is what inference follows, so
    training on uniform masks leaves a train/inference mismatch.
    """
    B, L = target.shape
    maskable = model.get_non_special_symbol_mask(target)
    rank = torch.rand(B, L, device=target.device)          # scaffold: random order
    for b, idx in enumerate(idx_list):
        for p in core_pos.get(idx, ()):
            if 0 <= p < L: rank[b, p] += 2.0               # core revealed first -> corrupted last
        for p in pocket_pos.get(idx, ()):
            if 0 <= p < L: rank[b, p] += 1.0
    frac = torch.rand(B, 1, device=target.device)          # how much of the chain to corrupt
    thr = torch.quantile(rank, frac.squeeze(1), dim=1, keepdim=True) if hasattr(torch, "quantile") else rank.median(dim=1, keepdim=True).values
    return (rank <= thr) & maskable


def is_bit_model(model) -> bool:
    return hasattr(model.net, "lm_head_struct") and hasattr(model.net, "quant2emb")


def bit_struct_targets(model, struct_target):
    bits = model.struct_tokenizer.quantize.get_codebook_entry(
        struct_target - model.struct_vocab_offset
    )
    return (bits > 0).long()


def bit_input_embeddings(model, input_ids, input_mask):
    input_struct_ids, input_aatype_ids = input_ids.chunk(2, dim=1)
    _, input_aatype_mask = input_mask.chunk(2, dim=1)
    input_aatype_embeds = model.net.esm.embeddings(
        input_aatype_ids,
        attention_mask=input_aatype_mask,
    )
    quant = model.struct_tokenizer.quantize.get_codebook_entry(
        input_struct_ids - model.struct_vocab_offset
    )
    if BSC_PROB > 0.0 and model.training:
        # flip a random subset of the bits of the ALREADY-DECODED struct tokens. A bit is the sign of
        # a quantized dimension, so flipping the bit == negating that dimension. Targets stay clean.
        p = torch.rand(quant.shape[0], 1, 1, device=quant.device) * BSC_PROB   # per-sample strength ~U[0,p]
        keep = (input_struct_ids != model.struct_mask_id) & (input_struct_ids != model.struct_eos_id)
        keep = keep.clone(); keep[:, 0] = False                                # never touch BOS/EOS/MASK
        flip = (torch.rand_like(quant) < p) & keep.unsqueeze(-1)
        quant = torch.where(flip, -quant, quant)
    input_struct_embeds = model.net.quant2emb(quant).float()
    input_struct_embeds[:, 0] = model.net.struct_bos_emb
    input_struct_embeds[input_struct_ids == model.struct_eos_id] = model.net.struct_eos_emb
    input_struct_embeds[input_struct_ids == model.struct_mask_id] = model.net.struct_mask_emb
    return torch.cat([input_struct_embeds, input_aatype_embeds], dim=1)


def split_conditioned_outputs(model, logits_or_dict, st_len):
    if isinstance(logits_or_dict, dict):
        return logits_or_dict["struct"], logits_or_dict["aatype"]
    return logits_or_dict[:, :st_len], logits_or_dict[:, st_len:]


def conditioned_forward(
    model,
    cond,
    input_ids,
    text_feats,
    st_len,
    motif_aa_ids,
    motif_aa_mask,
    motif_st_ids,
    motif_st_mask,
    single_modality_index=None,
    text_mask=None,   # JOINT_WIRE
    ground=None, ground_ids=None,   # GROUND_WIRE
):
    input_mask = input_ids.ne(model.pad_id)
    type_ids = model.get_modality_type(input_ids)
    L = input_ids.shape[1]
    num_heads = model.net.config.num_attention_heads
    attn = model.net.esm.get_extended_attention_mask(input_mask, input_ids.shape)
    attn = attn.repeat(1, num_heads, L, 1)
    if single_modality_index is not None:
        struct_attn, aa_attn = attn.chunk(2, dim=-2)
        struct_attn[single_modality_index, :, :, L // 2 :] = -torch.inf
        aa_attn[single_modality_index, :, :, : L // 2] = -torch.inf
        attn = torch.cat([struct_attn, aa_attn], dim=-2)
    globals()["_LAST_JOINT_LEN"] = int(input_ids.shape[1])
    bit = is_bit_model(model)
    emb = bit_input_embeddings(model, input_ids, input_mask) if bit else model.net.esm.embeddings(input_ids, attention_mask=input_mask)
    if ground is not None and ground_ids is not None:   # GROUND_WIRE
        emb = ground(emb, ground_ids, st_len)
    if getattr(cond, "is_deep_text_cross", False):
        pass
    elif hasattr(cond, "apply_cross"):
        emb = cond.apply_cross(
            emb,
            text_feats,
            motif_aa_ids,
            motif_aa_mask,
            motif_st_ids,
            motif_st_mask,
            model.net.esm.embeddings.word_embeddings,
            input_mask=input_mask,
            **({"text_mask": text_mask} if text_mask is not None else {}),
            
        )
    else:
        motif_aa_emb, motif_st_emb = motif_mean_embeddings(model, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask)
        aa_c, st_c = cond(text_feats, motif_aa_emb, motif_st_emb)
        add = torch.zeros_like(emb)
        add[:, :st_len, :] = st_c[:, None, :]
        add[:, st_len:, :] = aa_c[:, None, :]
        emb = emb + add
    if getattr(cond, "is_deep_text_cross", False):
        hidden_states = emb
        # byprot gates activation checkpointing inside model.net.forward, which this loop bypasses,
        # so DPLM_GRAD_CKPT would silently do nothing here and a long step would OOM.
        _deep_ckpt = bool(
            DPLM_CKPT_MIN_TOKENS
            and torch.is_grad_enabled()
            and hidden_states.requires_grad
            and hidden_states.shape[1] >= DPLM_CKPT_MIN_TOKENS
        )
        for layer_idx, layer_module in enumerate(model.net.esm.encoder.layer):

            def _deep_block(_h, _li=layer_idx, _lm=layer_module):
                _h = _lm(_h, attn, None, None, None, None, False, type_ids)[0]
                return cond.apply_layer(_h, _li, text_feats,
                                        input_mask=input_mask, text_mask=text_mask)

            if _deep_ckpt:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    _deep_block, hidden_states, use_reentrant=False)
            else:
                hidden_states = _deep_block(hidden_states)
        if model.net.esm.encoder.emb_layer_norm_after:
            hidden_states = model.net.esm.encoder.emb_layer_norm_after(hidden_states)
        hidden = hidden_states
        if bit:
            logits = {
                "struct": model.net.lm_head_struct(hidden_states.chunk(2, dim=1)[0]).reshape(input_ids.shape[0], st_len, -1, 2),
                "aatype": model.net.lm_head(hidden_states.chunk(2, dim=1)[1]),
            }
        else:
            logits = model.net.lm_head(hidden_states)
    else:
        out = model.net(input_ids=input_ids, inputs_embeds=emb, attention_mask=attn, type_ids=type_ids)
        logits = {
            "struct": out["struct_logits"].reshape(input_ids.shape[0], st_len, -1, 2),
            "aatype": out["aatype_logits"],
        } if bit else out["logits"]
        hidden = out.get("last_hidden_state", None)
    if hasattr(cond, "apply_logits") and hidden is not None:
        if bit:
            logits["aatype"] = cond.apply_logits(
                torch.cat(
                    [
                        torch.zeros(
                            logits["aatype"].shape[0],
                            st_len,
                            logits["aatype"].shape[-1],
                            device=logits["aatype"].device,
                            dtype=logits["aatype"].dtype,
                        ),
                        logits["aatype"],
                    ],
                    dim=1,
                ),
                hidden,
                text_feats,
                st_len,
            )[:, st_len:]
        else:
            logits = cond.apply_logits(logits, hidden, text_feats, st_len)
    if bit:
        logits["aatype"][:, :, :4] = -torch.inf
        logits["aatype"][:, :, 24:] = -torch.inf
    else:
        logits = apply_modality_vocab_mask(model, logits, st_len)
    return logits, hidden


def conditioned_logits(model, cond, input_ids, text_feats, st_len, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask):
    return conditioned_forward(model, cond, input_ids, text_feats, st_len, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask)[0]


def _zero_like_loss(logits):
    """A graph-connected zero that survives the -inf entries DPLM-2 puts in its logits.

    `logits.sum() * 0.0` was the obvious way to write this and it is wrong: the forbidden vocabulary
    slots hold -inf, so the sum is -inf and -inf * 0.0 is NaN. With a stage whose aa mask happens to
    be empty that turned 4 of the first 13 logged steps into loss=nan. It did not reach the weights
    here (step 300 came back finite, which a NaN parameter cannot do), but in any path that backwards
    this value it would, and it makes the loss curve unreadable either way.
    """
    return logits.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _ce_with_mask_and_weight(logits, target, loss_mask, weight):
    if not bool(loss_mask.any()):
        return _zero_like_loss(logits)
    loss = F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="none")
    if weight is not None:
        loss = loss * weight[loss_mask].to(loss.dtype)
    return loss.sum() / loss_mask.to(loss.dtype).sum().clamp_min(1.0)


def _bit_struct_ce_with_mask_and_weight(model, struct_logits, struct_target, loss_mask, weight):
    if not bool(loss_mask.any()):
        return _zero_like_loss(struct_logits)
    target_bits = bit_struct_targets(model, struct_target)
    loss = F.cross_entropy(
        struct_logits[loss_mask].reshape(-1, 2),
        target_bits[loss_mask].reshape(-1),
        reduction="none",
    )
    if weight is not None:
        loss = loss * weight[loss_mask].repeat_interleave(target_bits.shape[-1]).to(loss.dtype)
    return loss.sum() / loss_mask.to(loss.dtype).sum().mul(target_bits.shape[-1]).clamp_min(1.0)


def _joint_losses(model, logits, target, st_len, struct_mask, aa_mask, struct_weight=None, aa_weight=None):
    struct_target = target[:, :st_len]
    aa_target = target[:, st_len:]
    # PHASE1a: up-weight the catalytic core. Unweighted, the core (median 11 of 338 residues)
    # contributes ~3% of the loss - the part the dataset exists for is the part barely trained.
    if P1_CORE_BOOST > 0.0 and P1_BATCH_IDX:
        _cw = p1_core_weight(P1_BATCH_IDX, struct_target.shape, struct_target.device)
        struct_weight = _cw if struct_weight is None else struct_weight * _cw
        _aw = p1_core_weight(P1_BATCH_IDX, aa_target.shape, aa_target.device)
        aa_weight = _aw if aa_weight is None else aa_weight * _aw
    struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
    if is_bit_model(model):
        struct_loss = _bit_struct_ce_with_mask_and_weight(model, struct_logits, struct_target, struct_mask, struct_weight)
    else:
        struct_loss = _ce_with_mask_and_weight(struct_logits, struct_target, struct_mask, struct_weight)
    aa_loss = _ce_with_mask_and_weight(aa_logits, aa_target, aa_mask, aa_weight)
    return struct_loss, aa_loss


def _combine_modality_losses(struct_loss, aa_loss, struct_gamma=1.0, aa_gamma=1.0):
    denom = float(struct_gamma) + float(aa_gamma)
    if denom <= 0:
        denom = 1.0
    return (float(struct_gamma) * struct_loss + float(aa_gamma) * aa_loss) / denom


def fullmask_forward(
    model,
    cond,
    input_ids,
    target,
    loss_mask,
    st_len,
    text_feats,
    motif_aa_ids,
    motif_aa_mask,
    motif_st_ids,
    motif_st_mask,
    struct_gamma=1.0,
    aa_gamma=1.0,
):
    """Train the exact prompt-to-full-generation state used by one-shot generation."""
    logits, hidden = conditioned_forward(
        model,
        cond,
        input_ids,
        text_feats,
        st_len,
        motif_aa_ids,
        motif_aa_mask,
        motif_st_ids,
        motif_st_mask,
    )
    struct_loss, aa_loss = _joint_losses(
        model,
        logits,
        target,
        st_len,
        loss_mask[:, :st_len],
        loss_mask[:, st_len:],
    )
    loss = _combine_modality_losses(struct_loss, aa_loss, struct_gamma, aa_gamma)
    return loss, struct_loss.detach(), aa_loss.detach(), hidden, {"logits": logits}


def joint_high_noise_diffusion_forward(
    model,
    cond,
    target,
    st_len,
    text_feats,
    motif_aa_ids,
    motif_aa_mask,
    motif_st_ids,
    motif_st_mask,
    timestep_min_frac=0.75,
    weighting="constant",
    struct_gamma=1.0,
    aa_gamma=1.0,
):
    """Joint sequence+structure denoising biased to high-noise states where text is needed."""
    struct_target = target[:, :st_len]
    aatype_target = target[:, st_len:]
    bsz = struct_target.shape[0]
    num_timesteps = model.cfg.num_diffusion_timesteps
    lo = max(1, min(num_timesteps, int(round(float(timestep_min_frac) * num_timesteps))))
    t = torch.randint(lo, num_timesteps + 1, (bsz,), device=target.device)

    struct_type_id = model.get_modality_type(struct_target)
    aa_type_id = model.get_modality_type(aatype_target)
    _maskable = model.get_non_special_symbol_mask(struct_target)
    # PHASE1c: training masks are uniform while inference reveals structure in the staged
    # core -> pocket -> scaffold order; that mismatch is a documented failure mode.
    if P1_STAGE_MASK_P > 0.0 and P1_BATCH_IDX and torch.rand(()).item() < P1_STAGE_MASK_P:
        _maskable = _maskable & p1_stage_order_mask(
            struct_target, model, P1_BATCH_IDX, P1_CORE_POS, P1_POCKET_POS)
    struct_x_t, struct_mask = model.q_sample(
        struct_target,
        t,
        struct_type_id,
        maskable_mask=_maskable,
    )
    aa_x_t, aa_mask = model.q_sample(
        aatype_target,
        t,
        aa_type_id,
        maskable_mask=model.get_non_special_symbol_mask(aatype_target),
    )
    x_t = torch.cat([struct_x_t, aa_x_t], dim=1)
    logits, hidden = conditioned_forward(
        model,
        cond,
        x_t,
        text_feats,
        st_len,
        motif_aa_ids,
        motif_aa_mask,
        motif_st_ids,
        motif_st_mask,
    )

    w = p1_timestep_weight(t, num_timesteps, P1_NOISE_W if P1_NOISE_W != "constant" else weighting)
    struct_weight = w[:, None].expand_as(struct_target)
    aa_weight = w[:, None].expand_as(aatype_target)
    struct_loss, aa_loss = _joint_losses(model, logits, target, st_len, struct_mask, aa_mask, struct_weight, aa_weight)
    loss = _combine_modality_losses(struct_loss, aa_loss, struct_gamma, aa_gamma)
    return loss, struct_loss.detach(), aa_loss.detach(), hidden, {
        "struct_mask": struct_mask,
        "aa_mask": aa_mask,
        "x_t": x_t,
        "logits": logits,
        "t": t,
    }



P1_WIRED2 = True


def p1_restage_mask(x_t, target, mask, model, idx_list, core_pos, pocket_pos):
    """Keep each row's mask RATIO but move which positions are masked, so the core is masked last.

    construct_x_t picks positions uniformly. Inference reveals core -> pocket -> scaffold, so
    training on uniform positions leaves a mismatch. We re-draw the positions with the same count
    per row, ranking scaffold < pocket < core, and rebuild x_t accordingly.
    """
    B, L = mask.shape
    rank = torch.rand(B, L, device=mask.device)
    for b, idx in enumerate(idx_list[:B]):
        for p in core_pos.get(idx, ()):
            if 0 <= p < L: rank[b, p] -= 2.0        # lowest rank -> masked last
        for p in pocket_pos.get(idx, ()):
            if 0 <= p < L: rank[b, p] -= 1.0
    rank = torch.where(mask | (x_t != target), rank, torch.full_like(rank, float("inf")))
    keep_n = mask.sum(dim=1)                        # same number of masked positions per row
    order = rank.argsort(dim=1, descending=True)    # scaffold first
    newmask = torch.zeros_like(mask)
    ar = torch.arange(L, device=mask.device)[None, :].expand(B, L)
    sel = ar < keep_n[:, None]
    newmask.scatter_(1, order, sel)
    newmask = newmask & (mask | (x_t != target) | (x_t == x_t))   # stay within maskable region
    mask_id = model.struct_mask_id if hasattr(model, "struct_mask_id") else None
    new_x = torch.where(newmask, torch.full_like(x_t, mask_id) if mask_id is not None else x_t, target)
    return new_x, newmask



# ================================ MULTIROUND ==================================================
# Staged training over the 5-step CoT. See module docstring of multiround.py for the rationale.
MR_STAGE_POS = {}      # {idx: {"core": [...], "pocket": [...]}} 0-based
MR_STAGE_EMB = None    # nn.Embedding(6, text_dim), index 0 unused
MR_TEXT_EMB = {}       # {"idx#sK": tensor}  per-stage text embeddings
MR_CTX_CORRUPT = 0.15  # Diffusion-Forcing style corruption rate of the given context
# MR_SCHED_SAMPLE: probability that a revealed context position carries round k-1's own
# prediction instead of the truth. Orthogonal to MR_CTX_CORRUPT, which replaces the position
# with MASK: MASK says 'not told', a wrong token says 'told wrongly', and only the second is
# what inference actually presents. Scheduled-sampled positions stay revealed and stay out of
# the loss, matching inference, where committed positions are frozen and never re-predicted.
MR_SCHED_SAMPLE = float(os.environ.get("MR_SCHED_SAMPLE", "0") or 0)
# MR_FREERUN_EVAL: report a second set of validation numbers where round k is conditioned on
# round k-1's own output rather than the truth - the regime generation runs in. Costs nothing
# extra: the argmaxes are already computed for the accuracy counters.
MR_FREERUN_EVAL = os.environ.get("MR_FREERUN_EVAL", "0") not in ("0", "", "false", "False")
# MR_SINGLE_ROUND: one round that writes the whole protein from an all-masked start, instead of
# four nested ones. mr_build_round takes its context from sets[stage - 1]; leaving stage 1 out of
# the map makes that lookup empty, so nothing is revealed and the round is a full generation.
MR_SINGLE_ROUND = os.environ.get("MR_SINGLE_ROUND", "0") not in ("0", "", "false", "False")
MR_STAGES = (2,) if MR_SINGLE_ROUND else (2, 3, 4, 5)
# MR_FULL_TEXT: condition on the entire document rather than the current stage's span.
MR_FULL_TEXT = os.environ.get("MR_FULL_TEXT", "0") not in ("0", "", "false", "False")
QWEN_HIDDEN_MU_FULL = None   # token-count-weighted mean over all five spans; built at load
MR_STAGE_W = {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
MR_WIRED = True
# MR_ABLATE5 reads claims.json from here to find a same-EC partner, inside a try/except that turns
# every failure into ec "?" -- the ablation then still runs and still reports a number, but it is
# swapping with a random protein instead of a same-class one. Overridable, and asserted below.
MR_CLAIMS_ROOT = os.environ.get("MR_CLAIMS_ROOT",
                                "/data/workspace/caiqingran/protein_design/DISCO_repo/runs_v2")
MR_ROUNDS_PER_STEP = 0   # 0 = all rounds every step
# MR_TRIM_PAD: drop the trailing pad columns of the drawn batch before the rounds run.
# encode_targets pads the pack to its longest row, so without this every step costs
# 2*global_max+4 tokens regardless of the protein actually drawn.
MR_TRIM_PAD = os.environ.get("MR_TRIM_PAD", "0") not in ("0", "", "false", "False")
# EPOCH_SAMPLER: visit every record exactly once per epoch instead of drawing
# i.i.d. with replacement, which leaves 13.5% of the corpus unseen over two epochs.
EPOCH_SAMPLER = os.environ.get("EPOCH_SAMPLER", "0") not in ("0", "", "false", "False")
# DPLM_CKPT_MIN_TOKENS: checkpoint the protein trunk only when the step is actually long.
# Only the last 8 layers train, but cond injects into the embeddings and is trainable, so
# every layer's output requires grad and the whole stack gets recomputed - a full extra
# forward for memory a short step never needed. Checkpointing is an identity transform, so
# toggling it per step cannot change the gradients. 0 = leave what the constructor set.
DPLM_CKPT_MIN_TOKENS = int(os.environ.get("DPLM_CKPT_MIN_TOKENS", "0") or 0)
# MR_EVAL_SKIP_TEST: run the staged evaluation on val only during training.
# Both halves cost the same and only val feeds best_val; the test numbers are
# logged and never read, so producing them at every eval doubles the stall.
MR_EVAL_SKIP_TEST = os.environ.get("MR_EVAL_SKIP_TEST", "0") not in ("0", "", "false", "False")
# MR_TEXT_DROPOUT: --text_dropout never reached multiround_forward, so this objective
# has trained without a single unconditional step. 0.1 matches Imagen's rate.
MR_TEXT_DROPOUT = float(os.environ.get("MR_TEXT_DROPOUT", "0") or 0)
# CKPT_EVERY: also keep a checkpoint every N steps, not just the best one, so a run
# stopped by hand still has a trajectory to resume from or compare across.
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "0") or 0)
# MEM_PREFLIGHT_IDX: run the first N steps on these row indices, longest-first, so the
# worst case for memory is exercised in minute one through the real training path.
MEM_PREFLIGHT_IDX = [int(x) for x in
                     os.environ.get("MEM_PREFLIGHT_IDX", "").replace(",", " ").split()]
# LOG_MEM: peak allocated/reserved per rank, reset each interval.
LOG_MEM = os.environ.get("LOG_MEM", "1") not in ("0", "", "false", "False")
# MR_ABLATE_CHANNELS: also cross text on/off with the residue injector on/off. The five
# text conditions run with ground=None, so they describe the text pathway in isolation.
MR_ABLATE_CHANNELS = os.environ.get("MR_ABLATE_CHANNELS", "0") not in ("0", "", "false", "False")


def _wb(payload, step=None):
    """wandb.log, minus the keys whose value is not a finite number.

    An all-NaN key is still a key, so wandb draws it an empty panel; 28 of this run's charts were
    that and nothing else. Dropping the value at the call site means the panel is never created.
    """
    wb_mod = globals().get("_WB_RUN")
    if wb_mod is None:
        return
    clean = {}
    for k, v in payload.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and f not in (float("inf"), float("-inf")):   # f != f is the NaN test
            clean[k] = f
    if clean:
        wb_mod.log(clean, step=step) if step is not None else wb_mod.log(clean)
_MR_TD_KEEP = [0, 0]     # [kept, total], reported periodically
MR_RESID_SCALE = 0.0   # 3b: amplify the round-discriminative residual


# ============================== JOINT_TEXT ======================================================
QWEN = None            # the text model, with LoRA adapters, live in the training graph
QWEN_DOCS = None       # {idx: int32 token ids} for the five stages concatenated
QWEN_SPANS = None      # {idx: [(start, end)] * 5}
QWEN_HARD = None       # {idx: bool per token} True on residue-identifier tokens
QWEN_LAMBDA = 1.0      # weight on the structure loss when both objectives are summed
QWEN_LAMBDA_PENDING = [True]
MR_EVAL_TEXT_TOKENS = 576
MR_GEN_TEXT_SHIFT = 0   # control arm: rotate whose text each record is given
HARD_TOKEN_W = 1.0      # weight on residue-LETTER tokens in the text loss; 1.0 = the old behaviour
HARD_LETTER_IDS = None  # token ids that spell a residue letter, so digits keep weight 1
TEXT_LOSS_SCALE = 1.0   # constant on the text loss, so reweighting redistributes without adding
NCE_GAMMA = [0.0]        # weight on the cross-modal InfoNCE; calibrated at the first joint step
NCE_RATIO = 0.1          # target ||grad NCE|| / ||grad diffusion|| on the conditioner
NCE_PENDING = [True]
NCE_HEADS = None         # the two projection heads
NCE_WS = [1]
NCE_RANK = [0]
NCE_STAT = [0.0, 0.0, 0, 0, 0, 0.0]   # loss, top-1, rounds, candidates, length-controlled, freshness-control top-1
NCE_QSIZE = 1024         # FIFO of detached text embeddings to draw length-matched negatives from
NCE_K = 63               # negatives per anchor, so chance is 1/64
NCE_TOL = 8              # a negative counts as length-matched within this many residues
NCE_BANK = None          # frozen stage-1 text embeddings: the target and every negative
# NCE_SAME_EC: draw negatives from the anchor's own EC class. Length-matched negatives
# let class identity win the contrastive task, which is exactly what the text ablation
# found the model settling for.
NCE_SAME_EC = os.environ.get("NCE_SAME_EC", "0") not in ("0", "", "false", "False")
# DEEP_FIXED_SCALE: hold the per-layer gate at 1.0. out_proj already carries the
# magnitude, so the scalar only adds a single parameter that can switch the adapter off.
DEEP_FIXED_SCALE = os.environ.get("DEEP_FIXED_SCALE", "0") not in ("0", "", "false", "False")
# DEEP_TEXT_EC: predict the enzyme class from the text under cross-entropy and let it add a
# low-rank residual to every amino-acid logit. The class is 92.9% recoverable from the text
# by a bag-of-words retriever, yet the text is worth only +2.37 EC3 points, so the gap is in
# use rather than in availability.
DEEP_TEXT_EC = float(os.environ.get("DEEP_TEXT_EC", "0") or 0)   # CE weight; 0 = off
# DEEP_EC_LOGIT_OFF: build the class pathway and load its weights, but do not let the residual
# reach the logits. Separates "the residual is harmful" from "the adapters were starved".
DEEP_EC_LOGIT_OFF = os.environ.get("DEEP_EC_LOGIT_OFF", "0") not in ("0", "", "false", "False")
DEEP_LOGIT_RANK = int(os.environ.get("DEEP_LOGIT_RANK", "256") or 256)
TEXT_EC_STAT = [0.0, 0.0, 0]     # [loss sum, acc sum, count]
EC_LABEL_IDS = None              # published at setup; -1 where a record has no label
EC_LABEL_BY_ID = {}              # {record id: class id}; -1 where a record has no label
NCE_EC_MIN = int(os.environ.get("NCE_EC_MIN", "8") or 8)   # smaller classes fall back
QWEN_HIDDEN_MU = None   # corpus mean of Qwen hidden states; subtracted so the 0.3% signal becomes 30%
QWEN_CENTRE_EMA = None  # {stage: vec} running mean, tracks the encoder as its adapters train
QWEN_CENTRE_MOM = 0.99  # a step contributes 1%; over ~100 steps that is ~1200 records at batch 3x4


class _GradScale(torch.autograd.Function):
    """Identity forward, multiply the gradient by a constant backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def scale_grad(x, scale):
    return _GradScale.apply(x, scale) if scale != 1.0 else x


# 2048 truncated 133 of 56022 four-stage documents (0.24%), and qwen_batch keeps the FRONT, so
# what it dropped was step 5's label list -- the thing this round adds. Longest document is 2253.
def qwen_batch(idx_list, device, max_len=2304):
    """Pack the documents of a batch. Returns ids, attention, per-token stage index, hard mask."""
    docs = [QWEN_DOCS[i].long() for i in idx_list]
    L = min(max(len(d) for d in docs), max_len)
    x = torch.zeros(len(docs), L, dtype=torch.long)
    attn = torch.zeros(len(docs), L, dtype=torch.long)
    stage = torch.zeros(len(docs), L, dtype=torch.long)
    hard = torch.zeros(len(docs), L, dtype=torch.bool)
    for b, (i, d) in enumerate(zip(idx_list, docs)):
        n = min(len(d), L)
        x[b, :n] = d[:n]
        attn[b, :n] = 1
        hard[b, :n] = QWEN_HARD[i][:n]
        for k, (a, e) in enumerate(QWEN_SPANS[i], start=1):
            if e > a:
                stage[b, a:min(e, L)] = k
    return x.to(device), attn.to(device), stage.to(device), hard.to(device)


def qwen_text_loss(hidden, x, stage, hard, block=384, breakdown=False):
    """Next-token cross-entropy over stages 2..5, evaluated a block of positions at a time.

    Residue-identifier tokens carry a weight of HARD_TOKEN_W. They are about 5% of a stage and the
    only part the rest of the pipeline reads - residues are regex-extracted from this text and
    injected - yet they were the model's worst tokens at 1.0074 against 0.46-0.61 for the prose.
    The denominator is the weight sum, so this stays a weighted mean and does not rescale the loss
    that QWEN_LAMBDA is calibrated against.
    """
    head = QWEN.get_output_embeddings()
    tgt, st, hd = x[:, 1:], stage[:, 1:], hard[:, 1:]
    h_all = hidden[:, :-1]
    tot = torch.zeros((), device=hidden.device, dtype=torch.float32)
    n = 0
    per = {k: [0.0, 0] for k in (2, 3, 4, 5)}
    hard_acc = [0.0, 0]
    for s0 in range(0, h_all.shape[1], block):
        s1 = min(h_all.shape[1], s0 + block)
        sk = st[:, s0:s1]
        m = sk >= 2
        if not bool(m.any()):
            continue
        ce = F.cross_entropy(head(h_all[:, s0:s1][m]).float(), tgt[:, s0:s1][m], reduction="none")
        hsel_w = hd[:, s0:s1][m]
        if HARD_LETTER_IDS is not None:
            # only the residue letters. The mask also covers position digits - 71% of it - and the
            # end-to-end run showed indices are the part that can be wrong without much cost.
            hsel_w = hsel_w & torch.isin(tgt[:, s0:s1][m], HARD_LETTER_IDS.to(tgt.device))
        w = torch.where(hsel_w, ce.new_tensor(float(HARD_TOKEN_W)), ce.new_tensor(1.0))
        tot = tot + (ce * w).sum()
        n += float(w.sum())
        if breakdown:
            kk = sk[m]
            for k in (2, 3, 4, 5):
                sel = kk == k
                if bool(sel.any()):
                    per[k][0] += float(ce[sel].sum().detach()); per[k][1] += int(sel.sum())
            hsel = hd[:, s0:s1][m]
            if bool(hsel.any()):
                hard_acc[0] += float(ce[hsel].sum().detach()); hard_acc[1] += int(hsel.sum())
    loss_raw = tot / max(n, 1.0)
    loss = loss_raw * TEXT_LOSS_SCALE
    if not breakdown:
        return loss, None
    out = {k: (per[k][0] / per[k][1] if per[k][1] else float("nan")) for k in (2, 3, 4, 5)}
    out["hard"] = hard_acc[0] / hard_acc[1] if hard_acc[1] else float("nan")
    out["ce_raw"] = float(loss_raw.detach())   # before the scale, so the logs stay comparable
    return loss, out


def cond_report(cond, ground, tf=None, tmask=None):
    """One line describing the state of the conditioning path.

    joint_scale is the gate on the text: `emb + joint_scale * (protein - emb)`. It inits at 0.2 and a
    previous run trained it to 0.0137, which silenced the text entirely and made every downstream
    ablation read as zero. Printed with the live anisotropy of the batch's own text features, so the
    input and the gate can be watched together - a gate that closes while the input is well
    conditioned means the shortcut, not the representation, is the problem.
    """
    bits = []
    js = getattr(cond, "joint_scale", None)
    if js is not None:
        bits.append(f"joint_scale={float(js):.4f}")
    for n in ("gate_struct", "gate_aa"):
        g = getattr(ground, n, None) if ground is not None else None
        if g is not None:
            bits.append(f"{n}={float(g):+.4f}")
    if MR_STAGE_EMB is not None:
        bits.append(f"stage_emb_norm={float(MR_STAGE_EMB.weight.norm(dim=-1).mean()):.4f}")
    tf0 = globals().get("_LAST_TF0")
    if tf0 is not None and tf0.dim() == 3 and tf0.shape[0] > 2:
        with torch.no_grad():
            _t0 = tf0.float()                       # bf16 cannot accumulate 400 terms usefully
            m0 = (tmask.unsqueeze(-1).float() if tmask is not None
                  else torch.ones_like(_t0[..., :1]))
            p0 = (_t0 * m0).sum(1) / m0.sum(1).clamp_min(1e-6)
            bits.append(f"|text|={float(p0.norm(dim=-1).mean()):.3f}")
            p0 = torch.nn.functional.normalize(p0, dim=-1)
            c0 = p0 @ p0.T
            o0 = c0[~torch.eye(c0.shape[0], dtype=torch.bool, device=c0.device)]
            bits.append(f"cos_preStageEmb={float(o0.mean()):.4f}")
    if tf is not None and tf.dim() == 3 and tf.shape[0] > 2:
        with torch.no_grad():
            _t1 = tf.float()
            m = (tmask.unsqueeze(-1).float() if tmask is not None
                 else torch.ones_like(_t1[..., :1]))
            p = (_t1 * m).sum(1) / m.sum(1).clamp_min(1e-6)         # per-record pooled text
            p = torch.nn.functional.normalize(p, dim=-1)
            c = p @ p.T
            off = c[~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)]
            bits.append(f"cos_final={float(off.mean()):.4f}")
    return "[COND] " + "  ".join(bits)


def qwen_stage_states(hidden, idx_list, stage_k, max_tokens, device):
    """Hidden states inside stage k's span, right-aligned, with a mask. (B, N, H) and (B, N).

    Centred against the training corpus when a mean is supplied. Raw Qwen states are so anisotropic
    that different documents sit at cosine 0.9897 and the whole same-EC/different-EC distinction is a
    0.0029 gap; removing the shared direction rescales it to 0.2992 without changing what can be
    decoded. Padding is unaffected: only real spans are copied out, and pad slots stay exactly zero.
    """
    _dbg = not globals().get("_CENTRE_DBG_DONE", False)
    if _dbg:
        globals()["_CENTRE_DBG_DONE"] = True
        print(f"[CENTRE-DBG] stage={stage_k} mu_is_none={QWEN_HIDDEN_MU is None} "
              f"type={type(QWEN_HIDDEN_MU).__name__} "
              f"keys={sorted(QWEN_HIDDEN_MU) if isinstance(QWEN_HIDDEN_MU, dict) else 'n/a'} "
              f"hidden.shape={tuple(hidden.shape)} hidden.dtype={hidden.dtype} "
              f"|hidden|/token={float(hidden.float().norm(dim=-1).mean()):.2f}", flush=True)
    if QWEN_HIDDEN_MU is not None and QWEN_CENTRE_EMA is None:
        if MR_FULL_TEXT and QWEN_HIDDEN_MU_FULL is not None:
            _mu = QWEN_HIDDEN_MU_FULL
        else:
            _mu = QWEN_HIDDEN_MU[stage_k] if isinstance(QWEN_HIDDEN_MU, dict) else QWEN_HIDDEN_MU
        hidden = hidden - _mu.to(hidden.dtype).to(hidden.device)
        if _dbg:
            print(f"[CENTRE-DBG] |mu|={float(_mu.norm()):.2f} -> "
                  f"|hidden-mu|/token={float(hidden.float().norm(dim=-1).mean()):.2f}", flush=True)
    B = hidden.shape[0]
    Hd = hidden.shape[-1]
    keep = []
    for b, i in enumerate(idx_list):
        if MR_FULL_TEXT:
            _sp = QWEN_SPANS[i]
            a, e = _sp[0][0], _sp[-1][1]      # the whole document, stage 1 through stage 5
        else:
            a, e = QWEN_SPANS[i][stage_k - 1]
        e = min(e, hidden.shape[1]); a = min(a, e)
        keep.append(hidden[b, max(a, e - max_tokens):e])
    n = max(1, max(k.shape[0] for k in keep))
    out = hidden.new_zeros(B, n, Hd)
    msk = torch.zeros(B, n, dtype=torch.bool, device=device)
    for b, k in enumerate(keep):
        if k.shape[0]:
            out[b, -k.shape[0]:] = k          # right-aligned: the window keeps the tail
            msk[b, -k.shape[0]:] = True
        else:
            msk[b, -1] = True                 # never hand attention an all-padding row
    if QWEN_CENTRE_EMA is not None:
        # Adaptive centring, applied after the span is cut so the statistic describes exactly the
        # tokens the conditioner will see. Updated only when gradients are on, i.e. never during
        # evaluation, so scoring one checkpoint twice gives the same answer.
        with torch.no_grad():
            _mf = msk.unsqueeze(-1).float()
            _ema = QWEN_CENTRE_EMA[stage_k].to(out.device)
            # The collective goes only on the training path. Evaluation and generation run under
            # `if is_main:` - rank 0 alone - so an unconditional all_reduce there leaves three ranks
            # outside the group and NCCL's watchdog eventually kills the job.
            if torch.is_grad_enabled():
                _bm = (out.float() * _mf).sum(dim=(0, 1)) / _mf.sum(dim=(0, 1)).clamp_min(1e-6)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(_bm)
                    _bm /= torch.distributed.get_world_size()
                _ema = QWEN_CENTRE_MOM * _ema + (1.0 - QWEN_CENTRE_MOM) * _bm
                QWEN_CENTRE_EMA[stage_k] = _ema
        out = (out - _ema.to(out.dtype)) * msk.unsqueeze(-1).to(out.dtype)
    if _dbg:
        with torch.no_grad():
            _of = out.float()
            _m = msk.unsqueeze(-1).float()
            _p = (_of * _m).sum(1) / _m.sum(1).clamp_min(1e-6)
            _pn = torch.nn.functional.normalize(_p, dim=-1)
            _c = _pn @ _pn.T
            _o = _c[~torch.eye(_c.shape[0], dtype=torch.bool, device=_c.device)]
            print(f"[CENTRE-DBG] returned: out.shape={tuple(out.shape)} "
                  f"real_tokens/row={_m.sum(1).flatten().tolist()} "
                  f"|pooled|={float(_p.norm(dim=-1).mean()):.2f} "
                  f"pooled_cos={float(_o.mean()) if _o.numel() else float('nan'):.4f}", flush=True)
    return out, msk
# ============================ END JOINT_TEXT ====================================================

# ================================ GROUNDING ===================================================
SITE_PAIRS = {}        # {idx: {stage: [(aa_letter, pos0), ...]}}
SITE_AA = "ACDEFGHIKLMNPQRSTVWY"
SITE_AA_INDEX = {a: i for i, a in enumerate(SITE_AA)}
GROUND_RNG = None
GROUND_CUMULATIVE = False   # inject every stage up to k, not only k; see site_ground_ids
GROUND_NOISE_FALLBACK = 0   # borrow this stage's profile for a stage that has none (0 = fatal)


def site_ground_ids(idx_list, stage, L, device, pairs_override=None):
    """(B, L) of amino-acid indices at the positions this stage names; -1 elsewhere."""
    out = torch.full((len(idx_list), L), -1, dtype=torch.long, device=device)
    for b, ix in enumerate(idx_list):
        if pairs_override is not None:
            src = pairs_override[b]
        elif GROUND_CUMULATIVE:
            # The canvas mr_stage_sets builds is nested, so the injection has to be nested too.
            # Without this, stage 3 silently drops the core -- the step-3 text names the pocket and
            # never restates S41/D345/H348 -- and stages 4 and 5 would each carry only their own new
            # layer. Restating the labels in the text instead would cost ~4.7 tokens each and push
            # the document past qwen_batch's 2048 cap, which truncates the end: step 5's labels.
            src = [q for j in (2, 3, 4, 5) if j <= stage
                   for q in SITE_PAIRS.get(ix, {}).get(j, [])]
        else:
            src = SITE_PAIRS.get(ix, {}).get(stage, [])
        for aa, p in src:
            if 0 <= p < L:
                a = SITE_AA_INDEX.get(aa)
                if a is not None:
                    out[b, p] = a
    return out


# --------------------------- GROUND_NOISE -----------------------------------------------------
# Derived from the measured agreement between generated and reference pairs; see verify_corrupt.py,
# where the synthetic profile was checked against the real one before being used here.
SITE_NOISE_PROFILE = {
    # refit 2026-08-10 to the Qwen3.5-9B writer of q35_lam073 (refit_noise.py, 1350 test records).
    # The previous values were fitted to an older, weaker writer and would have corrupted stage 3 to
    # 58% of the reliability that writer actually achieves.
    2: {"exact": 0.232, "pos": 0.388, "within5": 0.748, "ratio": 18132 / 16041},
    3: {"exact": 0.185, "pos": 0.343, "within5": 0.737, "ratio": 24755 / 25583},
}

# RECORD_NOISE: per-record reliability, fitted to the observed mean and excess variance and then
# checked by sampling. Independence was rejected: overdispersion 3.9x and 37% empty records.
SITE_NOISE_BETA = {2: (0.366, 0.565), 3: (0.376, 0.810)}   # refit 2026-08-10; a<1 and b<1 at stage 2 = U-shaped, records are trustworthy or not, rarely in between
SITE_NOISE_PI = -1.0   # >=0 overrides the fitted mean, holding the concentration: the titration


def _noise_params(stage):
    p = SITE_NOISE_PROFILE.get(stage)
    if p is None:
        return None
    keep_exact = p["exact"]
    keep_pos = max(p["pos"] - p["exact"], 0.0)
    s_near = max(p["within5"] * p["ratio"] - p["pos"], 0.0)
    s_far = max(p["ratio"] - p["pos"] - s_near, 0.0)
    return keep_exact, keep_pos, s_near, s_far


def corrupt_site_pairs(pairs, L, stage, rng):
    """Degrade true pairs to the reliability the text model achieves, one record at a time."""
    prm = _noise_params(stage)
    if prm is None or not pairs:
        return pairs
    keep_exact, keep_pos, s_near, s_far = prm
    # RECORD_NOISE: this record's own reliability, so failures arrive together as they do in reality
    ab = SITE_NOISE_BETA.get(stage)
    if ab is not None:
        a, b = ab
        if SITE_NOISE_PI >= 0.0:                      # titration: same shape, different mean
            if SITE_NOISE_PI >= 0.999:
                return list(pairs)                    # oracle
            tot = a + b
            a, b = max(SITE_NOISE_PI * tot, 1e-3), max((1.0 - SITE_NOISE_PI) * tot, 1e-3)
        p_rec = rng.betavariate(a, b)
        base = keep_exact + keep_pos
        if base > 0:
            scale = p_rec / base
            keep_exact, keep_pos = keep_exact * scale, keep_pos * scale
        # the count the writer proposes does not shrink when it is wrong; the misses become
        # spurious placements instead
        lost = max(base - p_rec, 0.0)
        s_near, s_far = s_near + lost * 0.42, s_far + lost * 0.58
    n = len(pairs)
    truth = [p for _, p in pairs]
    tset = set(truth)
    out, used = [], set()
    for aa, p in pairs:
        u = rng.random()
        if u < keep_exact:
            out.append((aa, p)); used.add(p)
        elif u < keep_exact + keep_pos:
            out.append((SITE_AA[rng.randrange(20)], p)); used.add(p)
    # a near miss must not land on another true position: the core runs are contiguous, so base+1 is
    # often a second true residue and placing there would score as correct
    for _ in range(int(round(s_near * n))):
        for _try in range(12):
            q = truth[rng.randrange(len(truth))] + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            if 0 <= q < L and q not in used and q not in tset:
                out.append((SITE_AA[rng.randrange(20)], q)); used.add(q); break
    for _ in range(int(round(s_far * n))):
        for _try in range(12):
            q = rng.randrange(L)
            if q not in used and all(abs(q - t) > 5 for t in truth):
                out.append((SITE_AA[rng.randrange(20)], q)); used.add(q); break
    return out


def stage_pairs_noisy(ix, k, L, rng):
    """The pairs injected at stage k, each contributing stage degraded with its OWN profile.

    At inference the labels of an earlier stage come from what the writer wrote at THAT stage, so
    their reliability is that stage's. Corrupting a single restated union with one profile would be
    wrong in both directions: too harsh on the core, too kind on the newest layer.
    """
    ks = [j for j in (2, 3, 4, 5) if j <= k] if GROUND_CUMULATIVE else [k]
    out = {}
    for j in ks:
        for aa, p in corrupt_site_pairs(SITE_PAIRS.get(ix, {}).get(j, []), L, j, rng):
            out[p] = aa                      # a later stage wins the position
    return [(a, p) for p, a in out.items()]
# ------------------------- END GROUND_NOISE ---------------------------------------------------

class GroundingInjector(nn.Module):
    """Add a learned term at the index a residue identifier names.

    No attention and no positional encoding: the identifier gives the index, so the term is placed
    there. The gates start at zero so this begins as a no-op and the model has to find it useful -
    which it may not, since the positions can be wrong.
    """

    def __init__(self, hidden: int, n_aa: int = 20):
        super().__init__()
        self.emb = nn.Embedding(n_aa, hidden)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.gate_struct = nn.Parameter(torch.zeros(1))
        self.gate_aa = nn.Parameter(torch.zeros(1))

    def forward(self, emb, ground_ids, st_len):
        if ground_ids is None:
            return emb
        m = ground_ids >= 0                                   # (B, L)
        # No early return when the mask is empty. `v` is multiplied by the mask below, so an empty
        # mask already yields exactly `emb`; returning early instead would leave this module out of
        # the autograd graph on this rank alone, and allreduce_grads only reduces params whose grad
        # is not None - one rank would then issue fewer collectives than the others and NCCL
        # deadlocks. That is what killed q35_gnoise at step 3050: corruption can empty a record's
        # pair list, which never happened while training read the reference table.
        v = self.norm(self.mlp(self.emb(ground_ids.clamp_min(0))))
        v = v * m[..., None].to(v.dtype)
        out = emb.clone()
        out[:, :st_len] = out[:, :st_len] + self.gate_struct * v
        out[:, st_len:] = out[:, st_len:] + self.gate_aa * v
        return out
# ============================== END GROUNDING =================================================

def mr_stage_sets(idx, L):
    """MR_MODALITY: (struct_positions, aa_positions) that must exist by each stage; nested in both.

    Round 4 completes the backbone, round 5 completes the sequence on that backbone - the split the
    step-4/step-5 texts describe. Keying both rounds on the same position set made round 5 a copy of
    its own input.
    """
    d = MR_STAGE_POS.get(idx, {})
    core = {p for p in d.get("core", []) if 0 <= p < L}
    pocket = {p for p in d.get("pocket", []) if 0 <= p < L}
    site = core | pocket
    full = set(range(L))
    if MR_SINGLE_ROUND:
        # one round, everything, no context: stage 1 is absent so sets[stage - 1] comes back empty
        return {2: (full, full)}
    return {2: (core, core), 3: (site, site), 4: (full, site), 5: (full, full)}


def mr_build_round(model, target_row, st_len, idx, stage, device, pred_row=None):
    """Return (input_row, loss_mask_row) for one (record, stage).

    context S_{k-1} is revealed but corrupted; the increment is masked; anything outside S_k is
    masked and excluded from the loss because it is not part of this stage's design yet.
    """
    L = st_len
    sets = mr_stage_sets(idx, L)
    cur_st, cur_aa = sets[stage]
    prev_st, prev_aa = sets[stage - 1] if stage - 1 in sets else (set(), set())
    inp = target_row.clone()
    struct = inp[:st_len]
    aa = inp[st_len:]
    # MR_SPECIAL_SAFE: BOS/EOS/pad must never be replaced by MASK - every other masking path in this
    # file goes through get_non_special_symbol_mask first, and stage 2 keeps nothing, so without this
    # the whole row including the specials was overwritten (that was the stage-2 nan).
    sp_st = model.get_non_special_symbol_mask(struct.unsqueeze(0))[0]
    sp_aa = model.get_non_special_symbol_mask(aa.unsqueeze(0))[0]
    def _mk(ps):
        m = torch.zeros(L, dtype=torch.bool, device=device)
        if ps:
            m[torch.tensor(sorted(ps), dtype=torch.long, device=device)] = True
        return m

    keep_st, keep_aa = _mk(prev_st), _mk(prev_aa)
    # MR_SCHED_SAMPLE: swap part of the revealed context for what the model itself predicted last
    # round. Done before the MASK step and without touching `keep`, so these positions stay revealed
    # and stay out of the loss - the same status a committed position has at inference.
    if MR_SCHED_SAMPLE > 0 and pred_row is not None:
        _ps, _pa = pred_row[:st_len], pred_row[st_len:]
        _us = keep_st & (torch.rand(L, device=device) < MR_SCHED_SAMPLE)
        _ua = keep_aa & (torch.rand(L, device=device) < MR_SCHED_SAMPLE)
        struct = torch.where(_us & sp_st, _ps.to(struct.dtype), struct)
        aa = torch.where(_ua & sp_aa, _pa.to(aa.dtype), aa)
    # corrupt part of the given context so the model never sees a perfect history
    if MR_CTX_CORRUPT > 0:
        if keep_st.any():
            keep_st = keep_st & ~(torch.rand(L, device=device) < MR_CTX_CORRUPT)
        if keep_aa.any():
            keep_aa = keep_aa & ~(torch.rand(L, device=device) < MR_CTX_CORRUPT)
    struct = torch.where(keep_st | ~sp_st, struct, torch.full_like(struct, model.struct_mask_id))
    aa = torch.where(keep_aa | ~sp_aa, aa, torch.full_like(aa, model.aa_mask_id))
    inp = torch.cat([struct, aa], dim=0)
    # MR_MODALITY: score exactly the positions this round has to predict - inside S_k and currently
    # masked. Including the revealed context made the input equal the target there, which is a free
    # mark, and it was inflating every round (round 5 read as the easiest of the four).
    lm_st = _mk(cur_st) & ~keep_st & sp_st
    lm_aa = _mk(cur_aa) & ~keep_aa & sp_aa
    return inp, torch.cat([lm_st, lm_aa], dim=0)


def multiround_forward(model, cond, target, st_len, text_feats_by_stage, idx_list, device,
                       motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask,
                       struct_gamma=1.0, aa_gamma=1.0, stages=MR_STAGES, backward=True,
                       text_mask_by_stage=None, loss_scale=1.0,
                       ground=None, ground_pairs=None,
                       ground_noise=None, ground_drop_prob=0.0):   # GROUND_NOISE
    """One optimisation step covering every round, in parallel (teacher/diffusion forcing)."""
    _tkeep = None
    if MR_TEXT_DROPOUT > 0.0 and backward:      # training only; evaluation stays conditional
        _tkeep = (torch.rand((len(idx_list), 1), device=device)
                  >= float(MR_TEXT_DROPOUT)).to(torch.float32)
        _MR_TD_KEEP[0] += int(_tkeep.sum())
        _MR_TD_KEEP[1] += int(_tkeep.numel())
        if _MR_TD_KEEP[1] % 2000 == 0:
            print(f"  [TEXT-DROP] p={MR_TEXT_DROPOUT} kept "
                  f"{_MR_TD_KEEP[0]}/{_MR_TD_KEEP[1]} = "
                  f"{_MR_TD_KEEP[0] / max(_MR_TD_KEEP[1], 1):.3f}", flush=True)
    if MR_TRIM_PAD and target.dim() == 2:
        _pad = model.pad_id
        _sth, _aah = target[:, :st_len], target[:, st_len:]
        # encode_targets builds both halves at maxlen+2, so they are the same width; assert rather
        # than assume, because a mismatch would silently shift the struct/aa split.
        assert _aah.shape[1] == st_len, f"halves differ: {st_len} vs {_aah.shape[1]}"
        # pads are trailing and pad_id never appears inside real content, so a count is the length
        _w = int(torch.maximum((_sth != _pad).sum(1).max(), (_aah != _pad).sum(1).max()))
        if 0 < _w < st_len:
            target = torch.cat([_sth[:, :_w], _aah[:, :_w]], dim=1)
            st_len = _w
    if DPLM_CKPT_MIN_TOKENS:
        try:
            model.net.esm.encoder.gradient_checkpointing = bool(
                target.shape[1] >= DPLM_CKPT_MIN_TOKENS)
        except AttributeError:
            pass
    total = 0.0
    total_t = torch.zeros((), device=device)   # JOINT_FIX3: differentiable when backward=False
    st_sum = torch.tensor(0.0, device=device)
    aa_sum = torch.tensor(0.0, device=device)
    per_stage = {}
    hidden = None
    if MR_ROUNDS_PER_STEP and MR_ROUNDS_PER_STEP < len(stages):
        # unbiased: expectation over the sampled subset, rescaled by |stages|/R
        _perm = torch.randperm(len(stages))[:MR_ROUNDS_PER_STEP].tolist()
        _scale = len(stages) / MR_ROUNDS_PER_STEP
        stages = [stages[i] for i in sorted(_perm)]
    else:
        _scale = 1.0
    _prev_pred = None      # MR_SCHED_SAMPLE: round k-1's own output, filled in at the end of the loop
    for k in stages:
        inps, lms, tfs = [], [], []
        for b, idx in enumerate(idx_list):
            i_row, l_row = mr_build_round(model, target[b], st_len, idx, k, device,
                                          pred_row=None if _prev_pred is None else _prev_pred[b])
            inps.append(i_row); lms.append(l_row)
            tfs.append(text_feats_by_stage[k][b])
        x = torch.stack(inps); lm = torch.stack(lms); tf = torch.stack(tfs)
        if _tkeep is not None:                  # (B,1) for pooled tf, (B,1,1) for per-token tf
            tf = tf * (_tkeep if tf.dim() == 2 else _tkeep.unsqueeze(-1))
        if MR_RESID_SCALE > 0.0 and tf.dim() == 2:
            # MR_FIX4/3b: isolate what makes this round different. cos(s2,s5) is 0.95, so the shared
            # component dominates; removing the per-record mean over rounds leaves the round-specific
            # part, which is then rescaled to be usable as a condition.
            _all = torch.stack([text_feats_by_stage[j] for j in stages], dim=0)   # (K,B,D)
            tf = tf + MR_RESID_SCALE * (tf - _all.mean(dim=0))
        globals()["_LAST_TF0"] = tf.detach()          # before the stage embedding
        if MR_STAGE_EMB is not None:
            _se = MR_STAGE_EMB(torch.full((tf.shape[0],), k, dtype=torch.long, device=device))
            tf = tf + (_se if tf.dim() == 2 else _se[:, None, :])
        _gid = None
        if ground is not None:   # GROUND_WIRE: the identities this stage names, at their own index
            if ground_noise is not None:   # GROUND_NOISE
                _rng = ground_noise
                _src = []
                for ix in idx_list:
                    if ground_drop_prob > 0 and _rng.random() < ground_drop_prob:
                        _src.append([])            # nothing this time: keep the fallback alive
                    else:
                        _src.append(stage_pairs_noisy(ix, k, st_len, _rng))
                _gid = site_ground_ids(idx_list, k, st_len, device, _src)
            else:
                _gid = site_ground_ids(idx_list, k, st_len, device,
                                       None if ground_pairs is None else
                                       [ground_pairs.get((b, k), []) for b in range(len(idx_list))])
        globals()["_LAST_TF"] = tf.detach()
        globals()["_LAST_TMASK"] = (None if text_mask_by_stage is None
                                    else text_mask_by_stage.get(k))
        logits, hidden = conditioned_forward(model, cond, x, tf, st_len,
                                             motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask,
                                             text_mask=None if text_mask_by_stage is None
                                             else text_mask_by_stage.get(k),
                                             ground=ground, ground_ids=_gid)
        if NCE_HEADS is not None and torch.is_tensor(hidden):
            # before any of the data-dependent `continue`s below: those can fire on one rank and not
            # another, and an all_gather behind such a guard deadlocks
            _nl, _na, _nn, _lc = cross_modal_nce(
                hidden, x, tf,
                None if text_mask_by_stage is None else text_mask_by_stage.get(k),
                st_len, model.pad_id, idx_list)
            if backward and NCE_PENDING[0]:
                _pp = [p for p in cond.parameters() if p.requires_grad]
                _g1 = torch.autograd.grad(_joint_losses(model, logits, target, st_len,
                                                        lm[:, :st_len], lm[:, st_len:])[0]
                                          + _joint_losses(model, logits, target, st_len,
                                                          lm[:, :st_len], lm[:, st_len:])[1],
                                          _pp, retain_graph=True, allow_unused=True)
                _n1 = float(torch.sqrt(sum(((g * g).sum() for g in _g1 if g is not None),
                                           torch.zeros((), device=device))))
                _g2 = torch.autograd.grad(_nl, _pp, retain_graph=True, allow_unused=True)
                _n2 = float(torch.sqrt(sum(((g * g).sum() for g in _g2 if g is not None),
                                           torch.zeros((), device=device))))
                NCE_GAMMA[0] = (NCE_RATIO * _n1 / max(_n2, 1e-8)) if NCE_GAMMA[0] < 0 else NCE_GAMMA[0]
                NCE_PENDING[0] = False
                print(f"[NCE-CAL] gradient norms on the conditioner: diffusion {_n1:.4e} "
                      f"nce {_n2:.4e} -> gamma {NCE_GAMMA[0]:.4f} (target ratio {NCE_RATIO})",
                      flush=True)
            if backward and NCE_GAMMA[0] > 0:
                (NCE_GAMMA[0] * _nl).backward(retain_graph=True)
            NCE_STAT[0] += float(_nl.detach()); NCE_STAT[1] += _na
            NCE_STAT[2] += 1; NCE_STAT[3] = _nn; NCE_STAT[4] = _lc
        if lm.sum() == 0:
            per_stage[k] = float("nan"); continue
        if MR_SCHED_SAMPLE > 0:
            # the bit->token arithmetic is calc_metrics_bit's, not a second copy of it
            with torch.no_grad():
                _slg, _alg = split_conditioned_outputs(model, logits, st_len)
                _bw = model.struct_tokenizer.quantize.mask.to(_slg.device).long()
                _sp = (_slg.argmax(dim=-1) * _bw).sum(dim=-1) + model.struct_vocab_offset
                _prev_pred = torch.cat([_sp, _alg.argmax(dim=-1)], dim=1).detach()
        sl, al = _joint_losses(model, logits, target, st_len, lm[:, :st_len], lm[:, st_len:])
        w = MR_STAGE_W.get(k, 1.0)
        # MR_GAMMA_ACTIVE: a modality with nothing to predict returns an exact zero, so leaving its
        # gamma in the denominator would halve the round. Round 4 is backbone-only and round 5 is
        # sequence-only by construction, so this applies to both of them every step.
        _sg = struct_gamma if bool(lm[:, :st_len].any()) else 0.0
        _ag = aa_gamma if bool(lm[:, st_len:].any()) else 0.0
        if _sg == 0.0 and _ag == 0.0:
            per_stage[k] = float("nan"); continue
        lk = loss_scale * _scale * w * _combine_modality_losses(sl, al, _sg, _ag)
        if (DEEP_TEXT_EC > 0 and backward and EC_LABEL_BY_ID
                and getattr(cond, "n_ec", 0) > 0):
            # tf is still in the graph, so this reaches the conditioner and Qwen's LoRA. It has to
            # join lk before the backward below, which is per-round rather than accumulated.
            _lab = torch.tensor([EC_LABEL_BY_ID.get(str(i), -1) for i in idx_list],
                                device=tf.device, dtype=torch.long)
            _v = _lab.ge(0)
            if bool(_v.any()):
                _el = cond.ec_logits(
                    tf, None if text_mask_by_stage is None else text_mask_by_stage.get(k))
                if _el is not None:
                    _ce = F.cross_entropy(_el[_v].float(), _lab[_v])
                    TEXT_EC_STAT[0] += float(_ce.detach())
                    TEXT_EC_STAT[1] += float((_el[_v].argmax(-1) == _lab[_v]).float().mean())
                    TEXT_EC_STAT[2] += 1
                    lk = lk + DEEP_TEXT_EC * _ce
        if backward:                                   # MR_PER_ROUND_BWD
            # every round but the last retains: with a text model upstream all four back-propagate
            # through one shared forward, and freeing it early breaks the rounds that follow
            lk.backward(retain_graph=(k != stages[-1]))
            hidden = hidden.detach() if torch.is_tensor(hidden) else None
        else:
            total_t = total_t + lk
        total = total + float(lk.detach())
        st_sum = st_sum + w * sl.detach(); aa_sum = aa_sum + w * al.detach()
        per_stage[k] = float(sl.detach())
    return ((torch.tensor(total, device=device) if backward else total_t),
            st_sum, aa_sum, hidden, {"per_stage_struct": per_stage})
# ============================== END MULTIROUND ================================================

@torch.no_grad()
def multiround_evaluate(model, cond, pack, idx_list, batch_size=8, stages=MR_STAGES,
                        limit=0, text_ids=None, stage_map=None, zero_text=False,
                        ground=None):   # GROUND_WIRE
    """MR_STAGED_EVAL: per-round (bit_acc, aa_acc) using the training-time round construction."""
    target, st_len, text_feats = pack[1], pack[3], pack[4]
    motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask = pack[5], pack[6], pack[7], pack[8]
    device = target.device
    n = min(len(idx_list), target.shape[0])
    if limit:
        n = min(n, limit)
    # corruption is a training regulariser; leaving it on would make the score a random draw
    _saved = globals().get("MR_CTX_CORRUPT", 0.0)
    globals()["MR_CTX_CORRUPT"] = 0.0
    out = {}
    try:
        for k in stages:
            st_ok = st_tot = aa_ok = aa_tot = 0
            for s0 in range(0, n, batch_size):
                e0 = min(n, s0 + batch_size)
                ids = idx_list[s0:e0]
                inps, lms = [], []
                for b, ix in enumerate(ids):
                    i_row, l_row = mr_build_round(model, target[s0 + b], st_len, ix, k, device)
                    inps.append(i_row); lms.append(l_row)
                x = torch.stack(inps); lm = torch.stack(lms)
                # per-stage text features, built the same way as the training call site
                tmask = None
                if QWEN is not None:   # JOINT_EVAL
                    # JOINT_ABLATE: the text may come from a different record than the structure
                    tids = ids if text_ids is None else text_ids[s0:e0]
                    qx, qa, qs, qh = qwen_batch(tids, device)
                    qhid = QWEN(input_ids=qx, attention_mask=qa,
                                output_hidden_states=True).hidden_states[-1]
                    _kk = k if stage_map is None else stage_map.get(k, k)
                    tf, tmask = qwen_stage_states(qhid, tids, _kk, MR_EVAL_TEXT_TOKENS, device)
                    if zero_text:
                        tf = torch.zeros_like(tf)
                    tfs = None
                else:
                    tfs = {}
                    for j in stages:
                        v = []
                        for ix in ids:
                            e = MR_TEXT_EMB.get(f"{ix}#s{j}")
                            v.append(torch.as_tensor(e, dtype=torch.float32, device=device)
                                     if e is not None else text_feats[0] * 0.0)
                        tfs[j] = torch.stack(v)
                    tf = tfs[k]
                if MR_RESID_SCALE > 0.0 and tfs is not None:
                    tf = tf + MR_RESID_SCALE * (tf - torch.stack([tfs[j] for j in stages], dim=0).mean(dim=0))
                if MR_STAGE_EMB is not None:
                    _se = MR_STAGE_EMB(torch.full((tf.shape[0],), k, dtype=torch.long, device=device))
                    tf = tf + (_se if tf.dim() == 2 else _se[:, None, :])
                _gid = None
                if ground is not None:
                    if SITE_NOISE_PI >= 0.0 or GROUND_RNG is not None:   # RECORD_NOISE at eval
                        _rr = GROUND_RNG or __import__("random").Random(12345 + k)
                        _gid = site_ground_ids(
                            ids, k, st_len, device,
                            [stage_pairs_noisy(ix, k, st_len, _rr) for ix in ids])
                    else:
                        _gid = site_ground_ids(ids, k, st_len, device)
                logits, _ = conditioned_forward(model, cond, x, tf, st_len,
                                                motif_aa_ids[s0:e0], motif_aa_mask[s0:e0],
                                                motif_st_ids[s0:e0], motif_st_mask[s0:e0],
                                                text_mask=tmask, ground=ground, ground_ids=_gid)
                struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
                st_mask, aa_mask = lm[:, :st_len], lm[:, st_len:]
                if aa_mask.any():
                    aa_pred = aa_logits.argmax(dim=-1)
                    aa_ok += int((aa_pred[aa_mask] == target[s0:e0, st_len:][aa_mask]).sum())
                    aa_tot += int(aa_mask.sum())
                if st_mask.any():
                    tb = bit_struct_targets(model, target[s0:e0, :st_len])
                    bm = st_mask.unsqueeze(-1).expand_as(tb)
                    st_ok += int((struct_logits.argmax(dim=-1)[bm] == tb[bm]).sum())
                    st_tot += int(bm.sum())
            out[k] = (st_ok / st_tot if st_tot else float("nan"),
                      aa_ok / aa_tot if aa_tot else float("nan"),
                      st_tot)
    finally:
        globals()["MR_CTX_CORRUPT"] = _saved
    return out



@torch.no_grad()
def multiround_evaluate_freerun(model, cond, pack, idx_list, batch_size=8, stages=MR_STAGES,
                                limit=0, text_ids=None, stage_map=None, ground=None):
    """multiround_evaluate's accounting, in the regime generation actually runs in.

    Round k is conditioned on round k-1's own argmax instead of the truth. Batch outer, rounds
    inner, so one batch can carry its predictions forward. Costs the same forwards as the
    teacher-forced eval; the argmaxes were already being computed.
    """
    target, st_len, text_feats = pack[1], pack[3], pack[4]
    motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask = pack[5], pack[6], pack[7], pack[8]
    device = target.device
    n = len(idx_list)
    if limit:
        n = min(n, limit)
    _sc, _cc = globals().get("MR_SCHED_SAMPLE", 0.0), globals().get("MR_CTX_CORRUPT", 0.0)
    globals()["MR_SCHED_SAMPLE"] = 1.0      # always the model's own output
    globals()["MR_CTX_CORRUPT"] = 0.0       # and never MASK: this is about wrong, not absent
    acc = {k: [0, 0, 0, 0] for k in stages}     # st_ok, st_tot, aa_ok, aa_tot
    try:
        for s0 in range(0, n, batch_size):
            e0 = min(n, s0 + batch_size)
            ids = idx_list[s0:e0]
            prev_pred = None
            for k in stages:
                inps, lms = [], []
                for b, ix in enumerate(ids):
                    i_row, l_row = mr_build_round(
                        model, target[s0 + b], st_len, ix, k, device,
                        pred_row=None if prev_pred is None else prev_pred[b])
                    inps.append(i_row); lms.append(l_row)
                x = torch.stack(inps); lm = torch.stack(lms)
                tmask = None
                if QWEN is not None:
                    tids = ids if text_ids is None else text_ids[s0:e0]
                    qx, qa, _qs, _qh = qwen_batch(tids, device)
                    qhid = QWEN(input_ids=qx, attention_mask=qa,
                                output_hidden_states=True).hidden_states[-1]
                    _kk = k if stage_map is None else stage_map.get(k, k)
                    tf, tmask = qwen_stage_states(qhid, tids, _kk, MR_EVAL_TEXT_TOKENS, device)
                else:
                    v = []
                    for ix in ids:
                        e = MR_TEXT_EMB.get(f"{ix}#s{k}")
                        v.append(torch.as_tensor(e, dtype=torch.float32, device=device)
                                 if e is not None else text_feats[0] * 0.0)
                    tf = torch.stack(v)
                if MR_STAGE_EMB is not None:
                    _se = MR_STAGE_EMB(torch.full((tf.shape[0],), k, dtype=torch.long, device=device))
                    tf = tf + (_se if tf.dim() == 2 else _se[:, None, :])
                _gid = None if ground is None else site_ground_ids(ids, k, st_len, device)
                logits, _ = conditioned_forward(model, cond, x, tf, st_len,
                                                motif_aa_ids[s0:e0], motif_aa_mask[s0:e0],
                                                motif_st_ids[s0:e0], motif_st_mask[s0:e0],
                                                text_mask=tmask, ground=ground, ground_ids=_gid)
                struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
                st_mask, aa_mask = lm[:, :st_len], lm[:, st_len:]
                aa_pred = aa_logits.argmax(dim=-1)
                if aa_mask.any():
                    acc[k][2] += int((aa_pred[aa_mask] == target[s0:e0, st_len:][aa_mask]).sum())
                    acc[k][3] += int(aa_mask.sum())
                _bw = model.struct_tokenizer.quantize.mask.to(struct_logits.device).long()
                _bit = struct_logits.argmax(dim=-1)
                if st_mask.any():
                    tb = bit_struct_targets(model, target[s0:e0, :st_len])
                    bm = st_mask.unsqueeze(-1).expand_as(tb)
                    acc[k][0] += int((_bit[bm] == tb[bm]).sum())
                    acc[k][1] += int(bm.sum())
                # carry this round's own answer into the next one
                _sp = (_bit * _bw).sum(dim=-1) + model.struct_vocab_offset
                prev_pred = torch.cat([_sp, aa_pred], dim=1)
    finally:
        globals()["MR_SCHED_SAMPLE"] = _sc
        globals()["MR_CTX_CORRUPT"] = _cc
    return {k: (v[0] / v[1] if v[1] else float("nan"),
                v[2] / v[3] if v[3] else float("nan"), v[1]) for k, v in acc.items()}


def codefp_diffusion_forward(
    model,
    cond,
    target,
    st_len,
    text_feats,
    motif_aa_ids,
    motif_aa_mask,
    motif_st_ids,
    motif_st_mask,
    weighting="linear",
    text_dropout=0.0,
    struct_gamma=1.0,
    aa_gamma=1.0,
):
    """CodeFP/DPLM-2 style random-timestep absorbing diffusion training step."""
    struct_target = target[:, :st_len]
    aatype_target = target[:, st_len:]
    struct_noised, aatype_noised, single_modality_index = model.construct_x_t(struct_target, aatype_target)
    # PHASE1c: same mask ratio, staged positions (core corrupted last = revealed first)
    if P1_STAGE_MASK_P > 0.0 and P1_BATCH_IDX and torch.rand(()).item() < P1_STAGE_MASK_P:
        try:
            nx, nm = p1_restage_mask(struct_noised["x_t"], struct_target, struct_noised["mask"],
                                     model, P1_BATCH_IDX, P1_CORE_POS, P1_POCKET_POS)
            struct_noised = dict(struct_noised); struct_noised["x_t"] = nx; struct_noised["mask"] = nm
        except Exception as _e:
            print(f"[PHASE1c] restage skipped: {_e}", flush=True)
    x_t = torch.cat([struct_noised["x_t"], aatype_noised["x_t"]], dim=1)
    text_keep = torch.ones((text_feats.shape[0],), dtype=torch.bool, device=text_feats.device)
    if text_dropout and text_dropout > 0:
        keep = torch.rand((text_feats.shape[0], 1), device=text_feats.device) >= float(text_dropout)
        text_keep = keep.squeeze(1)
        text_feats = text_feats * keep.to(text_feats.dtype)
    logits, hidden = conditioned_forward(
        model,
        cond,
        x_t,
        text_feats,
        st_len,
        motif_aa_ids,
        motif_aa_mask,
        motif_st_ids,
        motif_st_mask,
        single_modality_index=single_modality_index,
    )
    num_timesteps = model.cfg.num_diffusion_timesteps
    # PHASE1b: constant treats every noise level as equally informative; Min-SNR / logit-normal put
    # the weight where light and heavy models actually disagree (the middle of the trajectory).
    _mode = P1_NOISE_W if P1_NOISE_W != "constant" else weighting
    struct_weight = p1_timestep_weight(struct_noised["t"], num_timesteps, _mode)[:, None].float()
    aatype_weight = p1_timestep_weight(aatype_noised["t"], num_timesteps, _mode)[:, None].float()
    struct_weight = struct_weight.expand_as(struct_target)
    aatype_weight = aatype_weight.expand_as(aatype_target)
    struct_loss, aa_loss = _joint_losses(model, logits, target, st_len, struct_noised["mask"], aatype_noised["mask"], struct_weight, aatype_weight)
    loss = _combine_modality_losses(struct_loss, aa_loss, struct_gamma, aa_gamma)
    return loss, struct_loss.detach(), aa_loss.detach(), hidden, {
        "struct_mask": struct_noised["mask"],
        "aa_mask": aatype_noised["mask"],
        "x_t": x_t,
        "logits": logits,
        "text_keep": text_keep,
    }


def ec_aux_loss_and_metrics(ec_head, hidden, input_ids, ec_label_ids, pad_id):
    if ec_head is None or hidden is None or ec_label_ids is None:
        zero = torch.tensor(0.0, device=hidden.device if hidden is not None else input_ids.device)
        return zero, 0.0, 0
    valid = ec_label_ids.ge(0)
    if not bool(valid.any()):
        return hidden.sum() * 0.0, 0.0, 0
    mask = input_ids.ne(pad_id)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    pooled = (hidden * mask[..., None].to(hidden.dtype)).sum(dim=1) / denom
    logits = ec_head(pooled[valid])
    target = ec_label_ids[valid]
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(dim=-1) == target).float().mean().item()
    return loss, acc, int(target.numel())


@torch.no_grad()
def evaluate_diffusion_loss(model, cond, pack, batch_size=8, weighting="linear", repeats=1):
    input_ids, target, _, st_len, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, *_ = pack
    losses = []
    st_losses = []
    aa_losses = []
    for _ in range(max(1, int(repeats))):
        for s in range(0, target.shape[0], batch_size):
            e = min(target.shape[0], s + batch_size)
            loss, st_loss, aa_loss, _, _ = codefp_diffusion_forward(
                model,
                cond,
                target[s:e],
                st_len,
                text_feats[s:e],
                motif_aa_ids[s:e],
                motif_aa_mask[s:e],
                motif_st_ids[s:e],
                motif_st_mask[s:e],
                weighting=weighting,
            )
            losses.append(float(loss.item()))
            st_losses.append(float(st_loss.item()))
            aa_losses.append(float(aa_loss.item()))
    return (
        sum(losses) / max(1, len(losses)),
        sum(aa_losses) / max(1, len(aa_losses)),
        sum(st_losses) / max(1, len(st_losses)),
    )


def lsfs_loss_and_metrics(motif_head, hidden, st_len, windows, window_mask, label_ids):
    if motif_head is None or hidden is None:
        zero = torch.tensor(0.0, device=windows.device)
        return zero, 0.0, 0
    feats = []
    labels = []
    st_hidden = hidden[:, :st_len, :]
    for b in range(windows.shape[0]):
        label = int(label_ids[b].detach().cpu())
        if label < 0:
            continue
        for k in range(windows.shape[1]):
            if not bool(window_mask[b, k]):
                continue
            start = int(windows[b, k, 0].detach().cpu())
            end = int(windows[b, k, 1].detach().cpu())
            start = max(0, min(start, st_len - 1))
            end = max(start + 1, min(end, st_len))
            if end - start <= 2:
                continue
            feats.append(st_hidden[b, start:end].mean(dim=0))
            labels.append(label)
    if not feats:
        zero = torch.tensor(0.0, device=windows.device)
        return zero, 0.0, 0
    feat = torch.stack(feats)
    target = torch.tensor(labels, dtype=torch.long, device=feat.device)
    logits = motif_head(feat)
    loss = F.cross_entropy(logits, target)
    acc = (logits.argmax(dim=-1) == target).float().mean().item()
    return loss, acc, int(target.numel())


def calc_metrics(logits, target, loss_mask, st_len):
    if isinstance(logits, dict):
        aa_pred = logits["aatype"].argmax(dim=-1)
        struct_bits = logits["struct"].argmax(dim=-1)
        struct_pred = (struct_bits * logits["struct"].new_tensor(
            model_bit_mask_for_metrics(logits["struct"].shape[-2]),
            dtype=torch.long,
        )).sum(dim=-1)
        raise RuntimeError("internal error: bit metrics require calc_metrics_bit(model, ...)")
    pred = logits.argmax(dim=-1)
    aa_mask = loss_mask[:, st_len:]
    st_mask = loss_mask[:, :st_len]
    aa_acc = (pred[:, st_len:][aa_mask] == target[:, st_len:][aa_mask]).float().mean().item()
    st_acc = (pred[:, :st_len][st_mask] == target[:, :st_len][st_mask]).float().mean().item()
    all_acc = (pred[loss_mask] == target[loss_mask]).float().mean().item()
    return all_acc, aa_acc, st_acc


def model_bit_mask_for_metrics(nbits: int):
    return [2**i for i in range(nbits)]


def calc_metrics_bit(model, logits, target, loss_mask, st_len):
    struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
    aa_pred = aa_logits.argmax(dim=-1)
    bit_pred = struct_logits.argmax(dim=-1)
    bit_weights = model.struct_tokenizer.quantize.mask.to(bit_pred.device).long()
    struct_pred = (bit_pred * bit_weights).sum(dim=-1) + model.struct_vocab_offset
    aa_mask = loss_mask[:, st_len:]
    st_mask = loss_mask[:, :st_len]
    aa_acc = (aa_pred[aa_mask] == target[:, st_len:][aa_mask]).float().mean().item()
    st_acc = (struct_pred[st_mask] == target[:, :st_len][st_mask]).float().mean().item()
    # P1_BITACC: index equality needs every bit right at once (p^d); the bit-level number is what the
    # model is trained on and what correlates with structure quality.
    try:
        _tb = bit_struct_targets(model, target[:, :st_len])
        _bm = st_mask.unsqueeze(-1).expand_as(_tb)
        bit_acc = (bit_pred[_bm] == _tb[_bm]).float().mean().item()
        n_bits = _tb.shape[-1]
    except Exception as _e:
        if not globals().get("_P1_BITERR"):
            globals()["_P1_BITERR"]=1
            import traceback; print("[P1_BITACC] failed:", repr(_e)); traceback.print_exc()
        bit_acc, n_bits = float("nan"), -1
    aa_ok = (aa_pred[aa_mask] == target[:, st_len:][aa_mask]).float().sum()
    st_ok = (struct_pred[st_mask] == target[:, :st_len][st_mask]).float().sum()
    denom = aa_mask.float().sum() + st_mask.float().sum()
    all_acc = ((aa_ok + st_ok) / denom.clamp_min(1.0)).item()
    globals()["P1_LAST_BITACC"] = (bit_acc, n_bits)
    return all_acc, aa_acc, st_acc


@torch.no_grad()
def evaluate_tokens(model, cond, pack, batch_size=8, motif_head=None, ec_head=None):
    input_ids, target, loss_mask, st_len, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, _, _, windows, window_mask, label_ids, ec_label_ids = pack
    parts = []
    lsfs_acc_sum = 0.0
    lsfs_n_sum = 0
    ec_acc_sum = 0.0
    ec_n_sum = 0
    for s in range(0, input_ids.shape[0], batch_size):
        e = min(input_ids.shape[0], s + batch_size)
        logits, hidden = conditioned_forward(
            model, cond, input_ids[s:e], text_feats[s:e], st_len,
            motif_aa_ids[s:e], motif_aa_mask[s:e], motif_st_ids[s:e], motif_st_mask[s:e],
        )
        if is_bit_model(model):
            parts.append(calc_metrics_bit(model, logits, target[s:e], loss_mask[s:e], st_len))
        else:
            parts.append(calc_metrics(logits, target[s:e], loss_mask[s:e], st_len))
        if motif_head is not None:
            _, acc, n = lsfs_loss_and_metrics(motif_head, hidden, st_len, windows[s:e], window_mask[s:e], label_ids[s:e])
            lsfs_acc_sum += acc * n
            lsfs_n_sum += n
        if ec_head is not None:
            _, acc, n = ec_aux_loss_and_metrics(ec_head, hidden, input_ids[s:e], ec_label_ids[s:e], model.pad_id)
            ec_acc_sum += acc * n
            ec_n_sum += n
    tok = tuple(sum(x[i] for x in parts) / len(parts) for i in range(3))
    return tok + ((lsfs_acc_sum / max(lsfs_n_sum, 1)), lsfs_n_sum, (ec_acc_sum / max(ec_n_sum, 1)), ec_n_sum)


def slice_pack(pack, limit: int):
    if not limit or limit <= 0:
        return pack
    n = pack[0].shape[0]
    limit = min(limit, n)
    out = []
    for item in pack:
        if torch.is_tensor(item) and item.shape[:1] == (n,):
            out.append(item[:limit])
        else:
            out.append(item)
    return tuple(out)


@torch.no_grad()
def conditioned_generate_official(
    model,
    cond,
    input_tokens,
    text_feats,
    st_len,
    motif_aa_ids,
    motif_aa_mask,
    motif_st_ids,
    motif_st_mask,
    max_iter=500,
    temperature=1.0,
    unmasking_strategy="stochastic1.0",
    sampling_strategy="annealing@2.0:1.0",
    cfg_aa_scale=1.0,
    cfg_struct_scale=1.0,
):
    model.eval()
    if QWEN is not None:
        QWEN.eval()   # lora_dropout=0.05 must not fire at inference
    initial_output_tokens, initial_output_scores = model.initialize_output_tokens(input_tokens)
    prev_decoder_out = dict(
        output_tokens=initial_output_tokens,
        output_scores=initial_output_scores,
        output_masks=None,
        attentions=None,
        step=0,
        max_step=max_iter,
        history=[initial_output_tokens.clone()],
        temperature=temperature,
        type_ids=model.get_modality_type(initial_output_tokens),
    )
    prev_decoder_out["output_masks"] = model.get_non_special_symbol_mask(prev_decoder_out["output_tokens"])

    for step in tqdm(range(max_iter), desc="Decoding", leave=False):
        output_tokens = prev_decoder_out["output_tokens"].clone()
        output_scores = prev_decoder_out["output_scores"].clone()
        output_masks = model.get_non_special_symbol_mask(output_tokens)

        logits, _ = conditioned_forward(
            model,
            cond,
            output_tokens,
            text_feats,
            st_len,
            motif_aa_ids,
            motif_aa_mask,
            motif_st_ids,
            motif_st_mask,
        )
        if is_bit_model(model):
            struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
            aa_logits[:, :, :4] = -math.inf
            aa_logits[:, :, 24:] = -math.inf
            aa_logits = top_k_top_p_filtering(aa_logits, top_p=0.95)
            cur_temp = temperature
            if sampling_strategy == "argmax":
                _tokens, _scores = model.sample_from_logits(aa_logits, struct_logits, temperature=0.0)
            elif sampling_strategy.startswith("annealing"):
                max_temp, min_temp = map(float, sampling_strategy.split("@")[1].split(":"))
                rate = 1 - step / max_iter
                cur_temp = min_temp + (max_temp - min_temp) * rate
                _tokens, _scores = model.sample_from_logits(aa_logits, struct_logits, temperature=cur_temp)
            else:
                _tokens, _scores = model.sample_from_logits(aa_logits, struct_logits, temperature=cur_temp)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])

            non_special_sym_mask = model.get_non_special_symbol_mask(prev_decoder_out["output_tokens"])
            output_masks, result_tokens, result_scores = model._reparam_decoding(
                output_tokens=prev_decoder_out["output_tokens"].clone(),
                output_scores=prev_decoder_out["output_scores"].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev_decoder_out["output_masks"],
                type_ids=prev_decoder_out["type_ids"].clone(),
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
            )
            prev_decoder_out.update(
                output_masks=output_masks,
                output_tokens=result_tokens,
                output_scores=result_scores,
                step=step + 1,
            )
            continue
        if float(cfg_aa_scale) != 1.0 or float(cfg_struct_scale) != 1.0:
            uncond_logits, _ = conditioned_forward(
                model,
                cond,
                output_tokens,
                torch.zeros_like(text_feats),
                st_len,
                motif_aa_ids,
                motif_aa_mask,
                motif_st_ids,
                motif_st_mask,
            )
            cond_logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
            uncond_logits = torch.nan_to_num(uncond_logits, neginf=-1e9, posinf=1e9)
            struct_scale = float(cfg_struct_scale)
            aa_scale = float(cfg_aa_scale)
            logits = uncond_logits.clone()
            logits[:, :st_len] = uncond_logits[:, :st_len] + struct_scale * (cond_logits[:, :st_len] - uncond_logits[:, :st_len])
            logits[:, st_len:] = uncond_logits[:, st_len:] + aa_scale * (cond_logits[:, st_len:] - uncond_logits[:, st_len:])
            logits = apply_modality_vocab_mask(model, logits, st_len)
        logits = F.log_softmax(logits, dim=-1)
        logits = top_k_top_p_filtering(logits, top_p=0.95)

        cur_temp = temperature
        if sampling_strategy == "argmax":
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == "gumbel_argmax":
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=0.0, noise_scale=cur_temp)
            _tokens.masked_scatter_(~output_masks, output_tokens[~output_masks])
        elif sampling_strategy.startswith("annealing"):
            max_temp, min_temp = map(float, sampling_strategy.split("@")[1].split(":"))
            rate = 1 - step / max_iter
            cur_temp = min_temp + (max_temp - min_temp) * rate
            _tokens, _scores = sample_from_categorical(logits, temperature=cur_temp)
        else:
            _tokens, _scores = sample_from_categorical(logits, temperature=cur_temp)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        non_special_sym_mask = model.get_non_special_symbol_mask(prev_decoder_out["output_tokens"])
        output_masks, result_tokens, result_scores = model._reparam_decoding(
            output_tokens=prev_decoder_out["output_tokens"].clone(),
            output_scores=prev_decoder_out["output_scores"].clone(),
            cur_tokens=output_tokens.clone(),
            cur_scores=output_scores.clone(),
            decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
            xt_neq_x0=prev_decoder_out["output_masks"],
            type_ids=prev_decoder_out["type_ids"].clone(),
            non_special_sym_mask=non_special_sym_mask,
            t=step + 1,
            max_step=max_iter,
        )
        prev_decoder_out.update(
            output_masks=output_masks,
            output_tokens=result_tokens,
            output_scores=result_scores,
            step=step + 1,
        )
    if is_bit_model(model):
        non_special_sym_mask = model.get_non_special_symbol_mask(prev_decoder_out["output_tokens"])
        return model.prepare_for_struct_tokenizer(prev_decoder_out, non_special_sym_mask)
    return prev_decoder_out["output_tokens"]


@torch.no_grad()
@torch.no_grad()
def mr_generate_batch(model, cond, x0, idx_list, st_len, motif, iters_total=500,
                      temperature=1.0, sampling_strategy="annealing@1.1:0.1",
                      unmasking_strategy="stochastic1.0", stages=MR_STAGES, verbose=False,
                      cfg_aa=1.0, cfg_struct=1.0, cfg_null_k=0, ground=None):
    """Decode one batch round by round, each round conditioned on its own stage text.

    Returns the final decoder state, shaped for prepare_for_struct_tokenizer.
    """
    device = x0.device
    B = x0.shape[0]
    m_aa, m_aam, m_st, m_stm = motif
    twoL = x0.shape[1]

    # specials are fixed for the whole decode and must never be masked or predicted
    nonspec = model.get_non_special_symbol_mask(x0)

    # a fully masked row to fall back on for anything this round has not committed
    blank = x0.clone()
    blank[:, :st_len] = model.struct_mask_id
    blank[:, st_len:] = model.aa_mask_id
    blank = torch.where(nonspec, blank, x0)

    # per-record position sets, expanded to the [struct ; aa] layout
    allowed = {}
    for k in stages:
        a = torch.zeros(B, twoL, dtype=torch.bool, device=device)
        for b, ix in enumerate(idx_list):
            s_pos, a_pos = mr_stage_sets(ix, st_len)[k]
            if s_pos:
                a[b, torch.tensor(sorted(s_pos), dtype=torch.long, device=device)] = True
            if a_pos:
                a[b, st_len + torch.tensor(sorted(a_pos), dtype=torch.long, device=device)] = True
        allowed[k] = a & nonspec

    # split the iteration budget by how much each round actually has to write
    committed = torch.zeros(B, twoL, dtype=torch.bool, device=device)
    todo = {}
    seen = torch.zeros_like(committed)
    for k in stages:
        todo[k] = allowed[k] & ~seen
        seen = seen | allowed[k]
    tot = sum(int(todo[k].sum()) for k in stages) or 1
    iters = {k: max(8, int(round(iters_total * int(todo[k].sum()) / tot))) for k in stages}
    print("[MR-BUDGET] " + json.dumps({"ids": idx_list, "requested": iters_total, "iterations_by_round": iters, "total_iterations": sum(iters.values())}), flush=True)

    cur = blank.clone()
    prev_out = None
    for k in stages:
        predict = allowed[k] & ~committed & nonspec
        if not bool(predict.any()):
            continue
        # anything not yet committed goes back to MASK: outside S_k is not this round's business
        x = torch.where(committed, cur, blank)
        partial = ~predict                      # True = frozen (dplm2.py:562-563)

        _tids = idx_list
        if MR_GEN_TEXT_SHIFT:      # give each record a neighbour's description instead of its own
            n = len(idx_list); sh = MR_GEN_TEXT_SHIFT % max(n, 1)
            _tids = idx_list[sh:] + idx_list[:sh]
        tf, tmask = mr_stage_text(_tids, k, device)
        # the residues this stage names, at their own indices - the same call training makes
        _gid = None if ground is None else site_ground_ids(idx_list, k, st_len, device)
        if ground is not None and verbose:
            _n = 0 if _gid is None else int((_gid >= 0).sum())
            print(f"    [round {k}] injecting {_n} named residues "
                  f"(gates struct={float(ground.gate_struct):+.4f} aa={float(ground.gate_aa):+.4f})",
                  flush=True)

        out_tok, out_sc = model.initialize_output_tokens(x, partial_masks=partial)
        T = iters[k]
        prev = dict(output_tokens=out_tok, output_scores=out_sc, output_masks=None,
                    attentions=None, step=0, max_step=T, history=[out_tok.clone()],
                    temperature=temperature, type_ids=model.get_modality_type(out_tok))
        prev["output_masks"] = model.get_non_special_symbol_mask(out_tok, partial_masks=partial)

        for step in range(T):
            output_tokens = prev["output_tokens"].clone()
            output_scores = prev["output_scores"].clone()
            output_masks = model.get_non_special_symbol_mask(output_tokens, partial_masks=partial)
            logits, _ = conditioned_forward(model, cond, output_tokens, tf, st_len,
                                            m_aa, m_aam, m_st, m_stm, text_mask=tmask,
                                            ground=ground, ground_ids=_gid)
            struct_logits, aa_logits = split_conditioned_outputs(model, logits, st_len)
            if float(cfg_aa) != 1.0 or float(cfg_struct) != 1.0:
                # The null is the zero vector: with the features centred, that IS the average
                # document, so the unconditional branch is "no particular enzyme" rather than an
                # out-of-distribution input. Extrapolation happens on raw logits - the -inf masking
                # and top-p below would otherwise produce inf - inf.
                if cfg_null_k and cfg_null_k > 0 and tf.shape[0] > 2:
                    # p(x) = E_c[p(x|c)], estimated over real descriptions. Offsets are spread across
                    # the batch because the shards are length-sorted and adjacent records are often
                    # homologs of the same class.
                    _B = tf.shape[0]
                    _offs = [max(1, round((j + 1) * _B / (cfg_null_k + 1))) % _B or 1
                             for j in range(cfg_null_k)]
                    _sa, _ss = [], []
                    for _o in _offs:
                        _tfn = torch.roll(tf, _o, dims=0)
                        _tmn = None if tmask is None else torch.roll(tmask, _o, dims=0)
                        _lg, _ = conditioned_forward(model, cond, output_tokens, _tfn, st_len,
                                                     m_aa, m_aam, m_st, m_stm, text_mask=_tmn,
                                                     ground=ground, ground_ids=_gid)
                        _s, _a = split_conditioned_outputs(model, _lg, st_len)
                        _sa.append(torch.log_softmax(_a.float(), dim=-1))
                        _ss.append(torch.log_softmax(_s.float(), dim=-1))
                    # a mean of distributions, not of logits
                    u_aa = torch.logsumexp(torch.stack(_sa), 0) - math.log(len(_sa))
                    u_struct = torch.logsumexp(torch.stack(_ss), 0) - math.log(len(_ss))
                    aa_logits = torch.log_softmax(aa_logits.float(), dim=-1)
                    struct_logits = torch.log_softmax(struct_logits.float(), dim=-1)
                else:
                    u_logits, _ = conditioned_forward(model, cond, output_tokens,
                                                      torch.zeros_like(tf), st_len,
                                                      m_aa, m_aam, m_st, m_stm, text_mask=tmask,
                                                      ground=ground, ground_ids=_gid)
                    u_struct, u_aa = split_conditioned_outputs(model, u_logits, st_len)
                aa_logits = u_aa + float(cfg_aa) * (aa_logits - u_aa)
                struct_logits = u_struct + float(cfg_struct) * (struct_logits - u_struct)
            aa_logits[:, :, :4] = -math.inf
            aa_logits[:, :, 24:] = -math.inf
            aa_logits = top_k_top_p_filtering(aa_logits, top_p=0.95)
            cur_temp = temperature
            if sampling_strategy.startswith("annealing"):
                mx, mn = map(float, sampling_strategy.split("@")[1].split(":"))
                cur_temp = mn + (mx - mn) * (1 - step / max(T, 1))
            elif sampling_strategy == "argmax":
                cur_temp = 0.0
            _tokens, _scores = model.sample_from_logits(aa_logits, struct_logits, temperature=cur_temp)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            nss = model.get_non_special_symbol_mask(prev["output_tokens"], partial_masks=partial)
            output_masks, result_tokens, result_scores = model._reparam_decoding(
                output_tokens=prev["output_tokens"].clone(),
                output_scores=prev["output_scores"].clone(),
                cur_tokens=output_tokens.clone(), cur_scores=output_scores.clone(),
                decoding_strategy=f"reparam-uncond-{unmasking_strategy}-linear",
                xt_neq_x0=prev["output_masks"], type_ids=prev["type_ids"].clone(),
                non_special_sym_mask=nss, t=step + 1, max_step=T)
            prev.update(output_masks=output_masks, output_tokens=result_tokens,
                        output_scores=result_scores, step=step + 1)

        cur = prev["output_tokens"]
        committed = committed | predict
        prev_out = prev
        if verbose:
            left = int((nonspec & ~committed).sum())
            still = int(((cur == model.struct_mask_id) | (cur == model.aa_mask_id)).sum())
            print(f"    [round {k}] {T:3d} iters, wrote {int(predict.sum()):5d} positions, "
                  f"{left:5d} not yet in any stage set, {still:5d} still MASK", flush=True)

    # a position that never entered any stage set was never written; round 5 covers everything, so
    # a nonzero count here means MR_STAGE_POS was missing for some record
    leftover = int((nonspec & ~committed).sum())
    if leftover:
        print(f"  [WARN] {leftover} non-special positions were never assigned to a round "
              f"and remain MASK", flush=True)

    prev_out["output_tokens"] = cur
    nss_final = model.get_non_special_symbol_mask(cur)
    return model.prepare_for_struct_tokenizer(prev_out, nss_final)


def qwen_hidden_only(qx, qa):
    """hidden_states[-1] without paying for the vocabulary head.

    The causal-LM wrapper materialises logits over 152k tokens and casts them to fp32 before we
    discard them; at batch 64 that is 55 GiB. The trunk yields the same states - PEFT patches the
    layers, not the head - so the adapters are still applied.
    """
    trunk = QWEN
    for _ in range(4):
        if hasattr(trunk, "layers") and not hasattr(trunk, "lm_head"):
            break
        nxt = getattr(trunk, "model", None)
        if nxt is None or nxt is trunk:
            break
        trunk = nxt
    if hasattr(trunk, "layers") and not hasattr(trunk, "lm_head"):
        return trunk(input_ids=qx, attention_mask=qa).last_hidden_state
    # layout not as expected: chunk the full model so the logits never exceed one chunk
    outs = []
    for i in range(0, qx.shape[0], 4):
        outs.append(QWEN(input_ids=qx[i:i + 4], attention_mask=qa[i:i + 4],
                         output_hidden_states=True).hidden_states[-1])
    return torch.cat(outs, dim=0)


_MR_QHID = {"key": None, "val": None}          # one batch of Qwen hidden states, reused across rounds
_MR_QHID_OFF = bool(int(__import__("os").environ.get("MR_QWEN_NOCACHE", "0")))


def mr_stage_text(idx_list, k, device):
    """Stage-k text features, built exactly as the training call site builds them (hybrid.py:1750)."""
    if QWEN is None:
        v = []
        for ix in idx_list:
            e = MR_TEXT_EMB.get(f"{ix}#s{k}")
            v.append(torch.as_tensor(e, dtype=torch.float32, device=device))
        tf, tmask = torch.stack(v), None
    else:
        # qwen_batch/qwen_hidden_only depend on idx_list only - not on k - so the multiround loop was
        # running the same 9B forward over the same ~1400-token document once per round. Cache it for
        # the duration of one batch: same inputs, same weights, so the states are bit-identical and
        # only qwen_stage_states (which does depend on k) still runs per round. Skipped when
        # gradients are on: reusing a cached activation across calls would backward through the same
        # graph twice. MR_QWEN_NOCACHE=1 restores the old behaviour for an A/B.
        _key = None if _MR_QHID_OFF else (tuple(idx_list), int(device.index or 0)
                                          if hasattr(device, "index") else str(device))
        if _key is not None and torch.is_grad_enabled():
            _key = None
        if _key is not None and _MR_QHID["key"] == _key:
            qhid = _MR_QHID["val"]
        else:
            qx, qa, _qs, _qh = qwen_batch(idx_list, device)
            qhid = qwen_hidden_only(qx, qa)
            if _key is not None:
                _MR_QHID["key"], _MR_QHID["val"] = _key, qhid
        tf, tmask = qwen_stage_states(qhid, idx_list, k, MR_EVAL_TEXT_TOKENS, device)
    if MR_STAGE_EMB is not None:
        se = MR_STAGE_EMB(torch.full((tf.shape[0],), k, dtype=torch.long, device=device))
        tf = tf + (se if tf.dim() == 2 else se[:, None, :])
    return tf, tmask


@torch.no_grad()
def multiround_generate_pack(model, cond, pack, rows, out_dir, batch_size=8, max_iter=500,
                             sampling_strategy="annealing@1.1:0.1", stages=MR_STAGES,
                             cfg_aa=1.0, cfg_struct=1.0, cfg_null_k=0, ground=None):
    """The multiround counterpart of generate_pack."""
    input_ids, _t, _l, st_len, text_feats, m_aa, m_aam, m_st, m_stm, *_rest = pack
    n = input_ids.shape[0]
    print(f"[MR-GEN] {n} records, stages={stages}, {max_iter} iterations split across rounds, "
          f"batch={batch_size}  cfg_aa={cfg_aa} cfg_struct={cfg_struct}"
          + ((f"  (guidance ON, null = mean over {cfg_null_k} real texts, "
               f"{1 + max(cfg_null_k, 1)} forwards per step)" if cfg_null_k
              else "  (guidance ON, null = zero vector, 2 forwards per step)")
             if (cfg_aa != 1.0 or cfg_struct != 1.0) else "  (guidance off)")
          + (f"  GROUNDING ON (gates {float(ground.gate_struct):+.4f}/{float(ground.gate_aa):+.4f})"
             if ground is not None else "  grounding off"), flush=True)
    outs = []
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        idx_list = [r["idx"] for r in rows[s:e]]
        o = mr_generate_batch(model, cond, input_ids[s:e], idx_list, st_len,
                              (m_aa[s:e], m_aam[s:e], m_st[s:e], m_stm[s:e]),
                              iters_total=max_iter, sampling_strategy=sampling_strategy,
                              stages=stages, verbose=(s == 0),
                              cfg_aa=cfg_aa, cfg_struct=cfg_struct, cfg_null_k=cfg_null_k,
                              ground=ground)
        outs.append(o)
        print(f"[MR-GEN] {e}/{n}", flush=True)
    outputs = {
        "output_tokens": torch.cat([o["output_tokens"] for o in outs], dim=0),
        "res_mask": torch.cat([o["res_mask"] for o in outs], dim=0),
        "final_struct_feature": torch.cat([o["final_struct_feature"] for o in outs], dim=0),
    }
    # numpy has no bfloat16; the structure decoder is fp32 and gains nothing from the cast anyway
    outputs = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in outputs.items()}
    with torch.autocast(device_type="cuda", enabled=False):
        save_results(tokenizer=model.tokenizer, struct_tokenizer=model.struct_tokenizer,
                     save_dir=str(out_dir), task="co_generation", outputs=outputs,
                     headers=[r["idx"] for r in rows], save_pdb=True)


def generate_pack(model, cond, pack, rows, out_dir, batch_size=8, max_iter=500, sampling_strategy="annealing@2.0:1.0", cfg_aa_scale=1.0, cfg_struct_scale=1.0):
    input_ids, _, _, st_len, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, st_prefix_lens, aa_prefix_lens, *_ = pack
    outs = []
    strip_prefix = bool(st_prefix_lens.sum().item() or aa_prefix_lens.sum().item())
    for s in range(0, input_ids.shape[0], batch_size):
        e = min(input_ids.shape[0], s + batch_size)
        out = conditioned_generate_official(
            model,
            cond,
            input_ids[s:e],
            text_feats[s:e],
            st_len,
            motif_aa_ids[s:e], motif_aa_mask[s:e], motif_st_ids[s:e], motif_st_mask[s:e],
            max_iter=max_iter,
            sampling_strategy=sampling_strategy,
            cfg_aa_scale=cfg_aa_scale,
            cfg_struct_scale=cfg_struct_scale,
        )
        outs.append(out)
    if is_bit_model(model):
        output_tokens = torch.cat([o["output_tokens"].cpu() for o in outs], dim=0)
        bit_res_mask = torch.cat([o["res_mask"].cpu() for o in outs], dim=0)
        bit_final_struct = torch.cat([o["final_struct_feature"].cpu() for o in outs], dim=0)
    else:
        output_tokens = torch.cat([o.cpu() for o in outs], dim=0)
    if strip_prefix:
        stripped = []
        max_st = 0
        max_aa = 0
        for i in range(output_tokens.shape[0]):
            sp = int(st_prefix_lens[i].cpu())
            ap = int(aa_prefix_lens[i].cpu())
            st_part = output_tokens[i, :st_len][sp:]
            aa_part = output_tokens[i, st_len:][ap:]
            stripped.append((st_part, aa_part))
            max_st = max(max_st, st_part.numel())
            max_aa = max(max_aa, aa_part.numel())
        final = torch.full((len(stripped), max_st + max_aa), model.pad_id, dtype=torch.long)
        for i, (st_part, aa_part) in enumerate(stripped):
            final[i, : st_part.numel()] = st_part
            final[i, max_st : max_st + aa_part.numel()] = aa_part
        output_tokens = final
    output_tokens = output_tokens.to(next(model.parameters()).device)
    outputs = {"output_tokens": output_tokens}
    if is_bit_model(model):
        outputs["res_mask"] = bit_res_mask.to(next(model.parameters()).device)
        outputs["final_struct_feature"] = bit_final_struct.to(next(model.parameters()).device)
    save_results(
        tokenizer=model.tokenizer,
        struct_tokenizer=model.struct_tokenizer,
        save_dir=str(out_dir),
        task="co_generation",
        outputs=outputs,
        headers=[r["idx"] for r in rows],
        save_pdb=True,
    )


def make_pack(model, cond_rows, hash_dim, device, use_retrieval_prefix: bool = False):
    input_ids, target, loss_mask, st_len, st_prefix_lens, aa_prefix_lens = encode_targets(model, cond_rows, use_retrieval_prefix)
    if cond_rows and "_text_emb" in cond_rows[0]:
        text_feats = torch.stack([torch.as_tensor(r["_text_emb"], dtype=torch.float32) for r in cond_rows])
    else:
        text_feats = torch.stack([text_hash(r["comment"], hash_dim) for r in cond_rows])
    motif_aa_ids, motif_aa_mask = pad_id_lists(cond_rows, "motif_aa_ids", model.pad_id)
    motif_st_ids, motif_st_mask = pad_id_lists(cond_rows, "motif_st_ids", model.pad_id)
    windows, window_mask, label_ids = pad_windows(cond_rows)
    ec_label_ids = pad_ec_aux_labels(cond_rows)
    globals()["EC_LABEL_IDS"] = ec_label_ids
    # keyed by the string id multiround_forward actually carries, so no row ordering
    # has to line up between the train, val and test packs
    EC_LABEL_BY_ID.update({str(r["idx"]): int(v)
                           for r, v in zip(cond_rows, ec_label_ids.tolist())})
    return (
        input_ids.to(device),
        target.to(device),
        loss_mask.to(device),
        st_len,
        text_feats.to(device),
        motif_aa_ids.to(device),
        motif_aa_mask.to(device),
        motif_st_ids.to(device),
        motif_st_mask.to(device),
        st_prefix_lens.to(device),
        aa_prefix_lens.to(device),
        windows.to(device),
        window_mask.to(device),
        label_ids.to(device),
        ec_label_ids.to(device),
    )


def setup_wandb(args):
    if not args.wandb_project:
        return None
    try:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run or Path(args.out_dir).name, config=vars(args))
        return wandb
    except Exception as exc:
        print(f"wandb disabled: {exc}", flush=True)
        return None


def attach_text_embeddings(rows, text_embed_map: dict | None):
    if text_embed_map is None:
        return rows
    missing = []
    out = []
    for r in rows:
        rr = dict(r)
        key = str(rr["idx"])
        emb = text_embed_map.get(key)
        if emb is None:
            missing.append(key)
        else:
            rr["_text_emb"] = emb
        out.append(rr)
    if missing:
        raise RuntimeError(f"text embeddings missing for {len(missing)} rows, first={missing[:5]}")
    return out


def load_text_embedding_map(path: str):
    if not path:
        return None
    obj = torch.load(path, map_location="cpu")
    emb = obj.get("embeddings", obj)
    return {str(k): torch.as_tensor(v, dtype=torch.float32) for k, v in emb.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_rows", required=True)
    ap.add_argument("--split_json")
    ap.add_argument("--val_rows", required=True)
    ap.add_argument("--test_rows", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--codefp_manifest", default="/data/workspace/caiqingran/protein_design/text2struct_dist/runs/codefp_style_motif_db_20260617_043046/motif_prototypes_manifest.jsonl")
    ap.add_argument("--metadata_parquet", default="/data/workspace/caiqingran/protein_design/text2struct_dist/runs/v70_train/data/metadata_v5_named.parquet")
    ap.add_argument("--anchors_jsonl", default="/data/workspace/caiqingran/protein_design/text2struct_dist/data/v207_uniprot_functional_anchor_targets/functional_anchor_targets.jsonl")
    ap.add_argument("--model_name", default="airkingbd/dplm2_bit_650m")
    ap.add_argument("--bit_model", action="store_true", default=True)
    ap.add_argument(
        "--struct_tokenizer_name",
        default="/home/caiqingran/.cache/huggingface/hub/models--airkingbd--struct_tokenizer/snapshots/dd9b86fff06889d74ad00d45377a58a17df2d2c2",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze_last", type=int, default=4)
    ap.add_argument("--hash_dim", type=int, default=4096)
    ap.add_argument("--text_embed_pt", default="")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0)
    ap.add_argument("--train_eval_limit", type=int, default=0)
    ap.add_argument("--eval_interval", type=int, default=25)
    ap.add_argument("--loss_log_interval", type=int, default=0)
    ap.add_argument("--skip_step1_eval", action="store_true")
    ap.add_argument("--use_retrieval_prefix", action="store_true")
    ap.add_argument("--use_codefp_cross_attn", action="store_true")
    ap.add_argument("--cross_text_tokens", type=int, default=1)
    ap.add_argument("--use_mmdit_text", action="store_true")
    ap.add_argument("--mmdit_layers", type=int, default=1)
    ap.add_argument("--qwen_path", default="", help="text model to train jointly; empty keeps the offline vectors")
    ap.add_argument("--qwen_docs", default="", help="tokenised five-stage documents with spans")
    ap.add_argument("--qwen_lr", type=float, default=1e-4)
    ap.add_argument("--qwen_lora_r", type=int, default=32)
    ap.add_argument("--qwen_freeze_steps", type=int, default=0, help="keep the text model fixed for this many steps so the conditioner fits a stationary target")
    ap.add_argument("--text_lambda", type=float, default=-1.0, help="weight on the structure loss; <0 calibrates it from gradient norms")
    ap.add_argument("--site_pairs", default="", help="(residue, position) pairs parsed from each stage text")
    ap.add_argument("--ground_noise_pi", type=float, default=-1.0, help="override the writer accuracy used for corruption, holding the failure shape; the titration")
    ap.add_argument("--ground_noise", action="store_true", help="degrade the injected pairs to the reliability the text model achieves")
    ap.add_argument("--ground_cumulative", action="store_true", help="inject every stage up to k, not only k; nested like the canvas")
    ap.add_argument("--ground_noise_fallback", type=int, default=0, help="a stage with no fitted noise profile borrows this stage's; 0 makes it fatal instead of silently oracle")
    ap.add_argument("--ground_drop_prob", type=float, default=0.0, help="probability of injecting nothing at all, so the model keeps a path that works without it")
    ap.add_argument("--ground_gate_scale", type=float, default=1.0, help="multiply the injector gates after loading; the inference-time trust knob")
    ap.add_argument("--ground_eval_only", action="store_true", help="score the staged metrics on test with the loaded pair table, then exit")
    ap.add_argument("--ground_mode", default="none", choices=["none", "gated", "inpaint"], help="how the named residues enter: not at all, as a gated term at their index, or written into the input")
    ap.add_argument("--per_token_text", action="store_true", help="condition on the text token sequence instead of one pooled vector")
    ap.add_argument("--max_text_tokens", type=int, default=576, help="tokens kept from a stage span, counted from the end (residues are named late)")
    ap.add_argument("--mmdit_text_tokens", type=int, default=4)
    ap.add_argument("--use_logit_residual", action="store_true")
    ap.add_argument("--logit_residual_rank", type=int, default=256)
    ap.add_argument("--logit_residual_init_scale", type=float, default=0.02)
    ap.add_argument("--use_deep_text_cross", action="store_true")
    ap.add_argument("--deep_cross_layers", type=int, default=8, help="last N DPLM2 layers get text cross-attn; <=0 means all layers")
    ap.add_argument("--deep_cross_text_tokens", type=int, default=4)
    ap.add_argument("--use_deep_per_token_cross", action="store_true",
                    help="cross-attend the full text token sequence at the last --deep_cross_layers layers; "
                         "requires --per_token_text")
    ap.add_argument("--no_retrieval", action="store_true")
    ap.add_argument("--lsfs_gamma", type=float, default=0.0)
    ap.add_argument("--anchor_radius", type=int, default=12)
    ap.add_argument("--max_anchor_groups", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--generate_batch_size", type=int, default=8)
    ap.add_argument("--diffusion_weighting", choices=["linear", "constant"], default="linear")
    ap.add_argument(
        "--objective",
        choices=[
            "codefp_diffusion",
            "fullmask_anchor",
            "hybrid_anchor_diffusion",
            "hybrid_alternating",
            "hybrid_anchor_codefp",
            "multiround",
        ],
        default="codefp_diffusion",
    )
    ap.add_argument("--fullmask_gamma", type=float, default=1.0)
    ap.add_argument("--high_noise_gamma", type=float, default=0.35)
    ap.add_argument("--bsc_prob", type=float, default=0.0, help="Bitwise Self-Correction: max fraction of already-decoded struct bits flipped in the context during training (0 = off)")
    ap.add_argument("--perstage_emb_pt", default="", help="MULTIROUND: {idx#sK: vec} per-stage text embeddings")
    ap.add_argument("--mr_ablate_text", action="store_true", help="MULTIROUND: also score with shuffled and zeroed text, then exit")
    ap.add_argument("--mr_eval_limit", type=int, default=256, help="MULTIROUND: records used by the staged evaluation (0=all); it costs 4 forwards each")
    ap.add_argument("--resid_scale", type=float, default=0.0, help="MULTIROUND 3b: amplify the round-discriminative part of the text embedding (0=off)")
    ap.add_argument("--rounds_per_step", type=int, default=0, help="MULTIROUND: train on R randomly chosen rounds per step (0=all 4); halves cost at R=2")
    ap.add_argument("--ctx_corrupt", type=float, default=0.15, help="MULTIROUND: Diffusion-Forcing corruption rate of the given context")
    ap.add_argument("--stage_weights", default="", help="MULTIROUND: e.g. 2:8,3:4,4:1,5:1 (default balances by position count)")
    ap.add_argument("--core_loss_boost", type=float, default=0.0, help="PHASE1a: extra weight on catalytic-core positions (0=off)")
    ap.add_argument("--noise_weighting", default="constant", choices=["constant","linear","minsnr","logitnorm"], help="PHASE1b: per-timestep loss weighting")
    ap.add_argument("--stage_mask_prob", type=float, default=0.0, help="PHASE1c: probability of stage-ordered masking instead of uniform (0=off)")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw","muon"], help="PHASE1d: optimiser")
    ap.add_argument("--core_pos_json", default="", help="PHASE1: {idx: [core positions]} produced by make_core_pos.py")
    ap.add_argument("--struct_loss_gamma", type=float, default=1.0, help="Relative weight for structure-token CE inside generation losses")
    ap.add_argument("--aa_loss_gamma", type=float, default=1.0, help="Relative weight for amino-acid CE inside generation losses")
    ap.add_argument("--hybrid_high_every", type=int, default=4, help="For hybrid_alternating, run one high-noise joint step every N training steps")
    ap.add_argument("--high_t_min_frac", type=float, default=0.75)
    ap.add_argument("--cond_lr", type=float, default=0.0, help="separate lr for randomly initialised modules (cond/MMDiT/heads); 0 = one lr for everything")
    ap.add_argument("--init_checkpoint", default="", help="Optional checkpoint with cond/model_trainable states to warm-start from")
    ap.add_argument("--text_dropout", type=float, default=0.0, help="conditional dropout probability for CFG training")
    ap.add_argument("--cfg_aa_scale", type=float, default=1.0, help="CFG scale for amino-acid logits during generation")
    ap.add_argument("--cfg_struct_scale", type=float, default=1.0, help="CFG scale for structure-token logits during generation")
    ap.add_argument("--ec_aux_gamma", type=float, default=0.0)
    ap.add_argument("--ec_aux_depth", type=int, default=3)
    ap.add_argument("--ec_vocab_json", type=str, default="",
                    help="load the EC label vocabulary from a training run ec_aux_label_vocab.json instead of rebuilding it from train_rows, which generation truncates")
    ap.add_argument("--eval_diffusion_repeats", type=int, default=1)
    ap.add_argument("--sample_max_iter", type=int, default=500)
    ap.add_argument("--sample_strategy", default="annealing@2.0:1.0")
    ap.add_argument("--final_generate_train_limit", type=int, default=0)
    ap.add_argument("--skip_final_generate", action="store_true")
    ap.add_argument("--tf32", action="store_true",
                    help="allow tensor-float matmuls; off by default in this torch build, which is "
                         "why an A800 sits at its fp32 peak instead of well above it")
    ap.add_argument("--gen_bf16", action="store_true",
                    help="run the final generation under bfloat16 autocast; master weights stay fp32 "
                         "so nothing downstream needs a dtype change")
    ap.add_argument("--nce_gamma", type=float, default=0.0,
                    help="weight on the cross-modal InfoNCE; <0 calibrates it from gradient norms, "
                         "0 = off")
    ap.add_argument("--nce_ratio", type=float, default=0.1,
                    help="target ||grad NCE|| / ||grad diffusion|| on the conditioner when calibrating")
    ap.add_argument("--nce_proj", type=int, default=256)
    ap.add_argument("--nce_k", type=int, default=63,
                    help="length-matched negatives per anchor; chance top-1 is 1/(1+k)")
    ap.add_argument("--nce_tol", type=int, default=8,
                    help="a negative counts as length-matched within this many residues. Every "
                         "stage-1 document states the exact residue count, so without this the "
                         "objective is a length-matching task a trivial baseline solves at 0.966")
    ap.add_argument("--qwen_grad_ckpt", action="store_true",
                    help="checkpoint the text model. Measured useless on Qwen3.5-9B (linear "
                         "attention keeps its memory in the recurrence) and 30-40%% slower")
    ap.add_argument("--nce_bank", default="",
                    help="frozen stage-1 text embeddings; required whenever the InfoNCE is on")
    ap.add_argument("--text_loss_scale", type=float, default=1.0,
                    help="constant on the text loss. Upweighting a subset of tokens raises the whole "
                         "text gradient (0.845 -> 2.740 measured at unfreezing); scaling by 0.3084 "
                         "puts it back, so the reweighting redistributes pressure without adding any")
    ap.add_argument("--hard_token_weight", type=float, default=1.0,
                    help="weight on residue-identifier tokens in the text CE. They are ~5%% of a "
                         "stage and the only part downstream reads, yet scored 1.0074 against "
                         "0.46-0.61 for the prose around them")
    ap.add_argument("--qwen_hidden_mean", default="",
                    help="corpus mean of Qwen hidden states, subtracted from the per-token text "
                         "features; without it the conditioner sees vectors at pairwise cosine 0.99 "
                         "and learns to shut the text gate")
    ap.add_argument("--cfg_null_k", type=int, default=0,
                    help="draws used to marginalise the unconditional branch; 0 keeps the zero "
                         "vector, which this model has never seen and treats as an unrelated "
                         "instruction rather than as the absence of one")
    ap.add_argument("--null_probe", type=int, default=0,
                    help="diagnose the CFG null: run N records under their own text, a zero vector "
                         "and another record's text, report entropy and divergence, then exit")
    ap.add_argument("--adaptive_centre", action="store_true",
                    help="track the per-stage mean with a running average instead of fixing it; a "
                         "fixed mean stops working the moment the Qwen adapters unfreeze, measured "
                         "as cos_final going from 0.2 back to 0.95 within 400 steps of unfreezing")
    ap.add_argument("--centre_momentum", type=float, default=0.99)
    ap.add_argument("--mr_gen_text_shift", type=int, default=0,
                    help="control arm: rotate the text within each batch so every record is designed "
                         "from a different record's description")
    ap.add_argument("--multiround_generate", action="store_true",
                    help="decode round by round the way the multiround objective trains, carrying "
                         "the model's own output forward; the single-shot path expects a pooled "
                         "text vector this architecture does not produce")
    ap.add_argument("--generate_only_test", action="store_true",
                    help="skip the train and validation decodes; 252 validation samples cost more "
                         "than the test split they are not being compared against")
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--wandb_run", default="")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.text_loss_scale != 1.0:
        globals()["TEXT_LOSS_SCALE"] = float(args.text_loss_scale)
        print(f"[TEXT] text loss scaled by {args.text_loss_scale}; the unfreezing line must then "
              f"report a text gradient near 8.45e-01 and lambda near 0.73, or this is not a control",
              flush=True)
    if args.hard_token_weight != 1.0:
        globals()["HARD_TOKEN_W"] = float(args.hard_token_weight)
        from transformers import AutoTokenizer as _AT
        _tk = _AT.from_pretrained(args.qwen_path)
        _lids = sorted({i for a in "ACDEFGHIKLMNPQRSTVWY"
                        for s in (a, " " + a)
                        for i in [_tk.encode(s, add_special_tokens=False)]
                        if len(i) == 1 for i in i})
        globals()["HARD_LETTER_IDS"] = torch.tensor(_lids, dtype=torch.long)
        print(f"[TEXT] {len(_lids)} residue-letter token ids will carry the weight; "
              f"position digits stay at 1", flush=True)
        print(f"[TEXT] residue-identifier tokens weighted x{args.hard_token_weight} in the text loss",
              flush=True)
    if args.qwen_hidden_mean:
        if not os.path.exists(args.qwen_hidden_mean):
            raise SystemExit(f"[CENTRE] {args.qwen_hidden_mean} not found")
        _mu = torch.load(args.qwen_hidden_mean, map_location="cpu")
        if isinstance(_mu, dict) and "stage_means" in _mu:
            globals()["QWEN_HIDDEN_MU"] = {int(k): v for k, v in _mu["stage_means"].items()}
            if MR_FULL_TEXT:
                _cnt = {int(k): float(v) for k, v in (_mu.get("counts") or {}).items()}
                if _cnt and set(_cnt) >= set(QWEN_HIDDEN_MU):
                    _tot = sum(_cnt[k] for k in QWEN_HIDDEN_MU)
                    globals()["QWEN_HIDDEN_MU_FULL"] = sum(
                        QWEN_HIDDEN_MU[k].float() * (_cnt[k] / _tot) for k in sorted(QWEN_HIDDEN_MU))
                    # is_main is not bound yet at this point; the neighbouring [CENTRE] print
                    # is unguarded for the same reason
                    print("[CENTRE] MR_FULL_TEXT: whole-document mean from token counts "
                          f"{ {k: int(_cnt[k]) for k in sorted(_cnt)} }, "
                          f"|mu_full|={float(QWEN_HIDDEN_MU_FULL.norm()):.2f}", flush=True)
                else:
                    raise SystemExit("[CENTRE] MR_FULL_TEXT needs per-stage token counts in "
                                     "qwen35_stage_means.pt; centring on one stage's mean would be wrong")
            _n = {k: round(float(v.norm()), 2) for k, v in sorted(QWEN_HIDDEN_MU.items())}
            print(f"[CENTRE] per-STAGE centring active; |mu| by stage {_n}", flush=True)
            if args.adaptive_centre:
                globals()["QWEN_CENTRE_EMA"] = {k: v.clone().cuda()
                                                for k, v in QWEN_HIDDEN_MU.items()}
                globals()["QWEN_CENTRE_MOM"] = float(args.centre_momentum)
                print(f"[CENTRE] adaptive: warm-started from the fixed means, momentum "
                      f"{args.centre_momentum}; it will follow the adapters as they train",
                      flush=True)
        else:
            globals()["QWEN_HIDDEN_MU"] = _mu["mean"] if isinstance(_mu, dict) else _mu
            print(f"[CENTRE] single global mean (dim {QWEN_HIDDEN_MU.numel()}, "
                  f"norm {QWEN_HIDDEN_MU.norm():.4f}) - NOTE this was measured to leave the "
                  f"conditioner at cosine 0.9887; a per-stage mean is what works", flush=True)
    else:
        print("[CENTRE] NOT centring the per-token text features - the conditioner will see "
              "near-identical vectors (measured cosine 0.9897) and is expected to shut the gate",
              flush=True)
    if args.mr_gen_text_shift:
        globals()["MR_GEN_TEXT_SHIFT"] = int(args.mr_gen_text_shift)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    global P1_CORE_BOOST, P1_NOISE_W, P1_STAGE_MASK_P, P1_CORE_POS
    P1_CORE_BOOST = args.core_loss_boost
    P1_NOISE_W = args.noise_weighting
    P1_STAGE_MASK_P = args.stage_mask_prob
    if args.core_pos_json and os.path.exists(args.core_pos_json):
        import json as _j; _d = _j.load(open(args.core_pos_json))
        P1_CORE_POS = {k: set(v.get("core", [])) for k, v in _d.items()}
        globals()["P1_POCKET_POS"] = {k: set(v.get("pocket", [])) for k, v in _d.items()}
    if (args.core_loss_boost > 0.0 or args.stage_mask_prob > 0.0) and not P1_CORE_POS:
        raise SystemExit("[PHASE1] core_loss_boost/stage_mask_prob set but core_pos_json is empty - "
                         "the flag would silently do nothing.")
    # ---- MR_WIRED: multiround setup -------------------------------------------------------
    if args.objective == "multiround":
        import json as _mj
        if not args.core_pos_json or not os.path.exists(args.core_pos_json):
            raise SystemExit("[MULTIROUND] --core_pos_json is required (defines each stage's positions)")
        _need_emb = not args.qwen_path      # only the frozen-text path reads these
        if _need_emb and (not args.perstage_emb_pt or not os.path.exists(args.perstage_emb_pt)):
            raise SystemExit("[MULTIROUND] --perstage_emb_pt is required when no text model is "
                             "loaded (it is the frozen-text fallback)")
        if args.qwen_path and args.perstage_emb_pt:
            print("[MULTIROUND] a text model is loaded, so the pre-computed embeddings are "
                  "ignored; not reading them", flush=True)
            args.perstage_emb_pt = ""
        globals()["MR_STAGE_POS"] = _mj.load(open(args.core_pos_json))
        _mr = torch.load(args.perstage_emb_pt, map_location="cpu") if args.perstage_emb_pt else {}
        globals()["MR_TEXT_EMB"] = _mr["embeddings"] if isinstance(_mr, dict) and "embeddings" in _mr else _mr
        if MR_TEXT_EMB:
            _w = next(iter(MR_TEXT_EMB.values()))
            _w = int(torch.as_tensor(_w).shape[-1])
            print(f"[MULTIROUND] frozen text embeddings loaded, width {_w}", flush=True)
            globals()["_MR_EMB_WIDTH"] = _w
        globals()["MR_CTX_CORRUPT"] = float(args.ctx_corrupt)
        globals()["MR_ROUNDS_PER_STEP"] = int(args.rounds_per_step)
        globals()["MR_RESID_SCALE"] = float(args.resid_scale)
        print(f"[MULTIROUND] stage_pos={len(MR_STAGE_POS)} stage_text_emb={len(MR_TEXT_EMB)} "
              f"ctx_corrupt={MR_CTX_CORRUPT}", flush=True)
        if int(os.environ.get("RANK", "0")) == 0 and MR_TEXT_EMB and QWEN is None:   # MR_EMB_ANISO: offline path only
            import torch.nn.functional as _Fn
            _ids = sorted({k.split("#")[0] for k in MR_TEXT_EMB})[:400]
            _v = [MR_TEXT_EMB[f"{i}#s2"] for i in _ids if f"{i}#s2" in MR_TEXT_EMB]
            if len(_v) > 8:
                _V = _Fn.normalize(torch.stack([torch.as_tensor(x).float() for x in _v]), dim=1)
                _C = _V @ _V.T
                _off = _C[~torch.eye(len(_C), dtype=torch.bool)]
                print(f"[MULTIROUND] text embedding anisotropy: cosine between DIFFERENT records "
                      f"mean={_off.mean():.4f} p95={_off.quantile(0.95):.4f} "
                      f"({'USABLE' if _off.mean() < 0.9 else 'DEGENERATE - the model cannot read these'})",
                      flush=True)

    print(f"[PHASE1] core_boost={P1_CORE_BOOST} noise_w={P1_NOISE_W} stage_mask_p={P1_STAGE_MASK_P} opt={args.optimizer} core_pos={len(P1_CORE_POS)}", flush=True)
    global BSC_PROB
    BSC_PROB = args.bsc_prob
    print(f"[BSC] bitwise self-correction prob = {BSC_PROB}", flush=True)
    local_rank, rank, world_size = init_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_train = load_json(args.train_rows)
    split_obj = load_json(args.split_json) if args.split_json else {}
    train_ids = split_obj.get("train_idx") if split_obj else None
    forbidden_ids = set(map(str, split_obj.get("val_idx", []))) | set(map(str, split_obj.get("test_idx", [])))
    train_rows = select_rows(all_train, train_ids, args.train_limit, max_len=args.max_len)
    val_rows = select_rows(load_json(args.val_rows), limit=args.val_limit, max_len=args.max_len)
    test_rows = select_rows(load_json(args.test_rows), limit=args.test_limit, max_len=args.max_len)
    if args.ec_vocab_json:
        _ev = load_json(args.ec_vocab_json)
        if int(_ev["depth"]) != args.ec_aux_depth:
            raise SystemExit("[EC-VOCAB] %s is depth %s, --ec_aux_depth is %d"
                             % (args.ec_vocab_json, _ev["depth"], args.ec_aux_depth))
        ec_aux_labels = list(_ev["labels"])
    else:
        ec_aux_labels = (build_ec_label_vocab(train_rows, args.ec_aux_depth)
                         if (args.ec_aux_gamma > 0 or DEEP_TEXT_EC > 0) else [])
    ec_aux_label_to_id = {label: i for i, label in enumerate(ec_aux_labels)}
    if is_main:
        print(
            f"rank={rank}/{world_size} local_rank={local_rank} rows train={len(train_rows)} "
            f"val={len(val_rows)} test={len(test_rows)} forbidden={len(forbidden_ids)}",
            flush=True,
        )
        json.dump(train_rows, open(out / "train_rows.json", "w"), indent=2)
        json.dump(val_rows, open(out / "val_rows.json", "w"), indent=2)
        json.dump(test_rows, open(out / "test_rows.json", "w"), indent=2)
        json.dump({"depth": args.ec_aux_depth, "labels": ec_aux_labels}, open(out / "ec_aux_label_vocab.json", "w"), indent=2)
    barrier(world_size)

    model_cls = DPLM2Bit if args.bit_model else DPLM2
    model = model_cls.from_pretrained(
        args.model_name,
        cfg_override={
            "tokenizer": {"vocab_file": args.model_name},
            "struct_tokenizer": {"exp_path": args.struct_tokenizer_name},
            # DPLM_GRAD_CKPT: byprot gates activation checkpointing on cfg.gradient_ckpt
            # (dplm2.py:90). Off by default so nothing changes for the existing runs; needed once
            # max_len is uncapped, where a 997-residue draw is 1998 joint tokens in one step.
            **({"gradient_ckpt": True}
               if os.environ.get("DPLM_GRAD_CKPT", "0") not in ("0", "", "false", "False") else {}),
        },
    ).cuda()
    if issubclass(type(model.net), PeftModel):
        model.net = model.net.merge_and_unload()
    model.train()
    if QWEN is not None:
        QWEN.train()   # lora_dropout=0.05 must not fire at inference
    for p in model.parameters():
        p.requires_grad_(False)
    if args.unfreeze_last:
        for layer in model.net.esm.encoder.layer[-args.unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in model.net.lm_head.parameters():
            p.requires_grad_(True)
        if hasattr(model.net, "lm_head_struct"):
            for p in model.net.lm_head_struct.parameters():
                p.requires_grad_(True)
        for name in ("quant2emb", "struct_bos_emb", "struct_eos_emb", "struct_mask_emb"):
            obj = getattr(model.net, name, None)
            if obj is None:
                continue
            if torch.is_tensor(obj):
                obj.requires_grad_(True)
            else:
                for p in obj.parameters():
                    p.requires_grad_(True)

    if args.no_retrieval:
        manifest = {}
        labels = ["NO_RETRIEVAL"]
        if args.lsfs_gamma != 0:
            print("no_retrieval: forcing lsfs_gamma=0", flush=True)
            args.lsfs_gamma = 0.0
    else:
        manifest = read_manifest(args.codefp_manifest)
        label_set = set()
        for row in train_rows:
            proto = pick_prototype(manifest, row)
            if proto is not None:
                label_set.add(proto["_picked_label"])
        labels = sorted(label_set)
        if not labels:
            raise RuntimeError("empty LSFS label vocabulary")
    label_to_id = {label: i for i, label in enumerate(labels)}
    if is_main:
        json.dump({"labels": labels}, open(out / "lsfs_label_vocab.json", "w"), indent=2)
    anchor_map = read_anchor_map(args.anchors_jsonl) if args.lsfs_gamma > 0 else None
    metadata_by_idx, metadata_by_uid = load_metadata_index(args.metadata_parquet)
    if is_main:
        pass
    train_tok = tokenize_pdbs_distributed(model, train_rows, out / "train_dplm2_tokens.pt", rank, world_size)
    val_tok = tokenize_pdbs_distributed(model, val_rows, out / "val_dplm2_tokens.pt", rank, world_size)
    test_tok = tokenize_pdbs_distributed(model, test_rows, out / "test_dplm2_tokens.pt", rank, world_size)
    barrier(world_size)

    if is_main:
        if args.no_retrieval:
            train_ret = build_no_retrieval_rows(train_tok, out / "train_no_retrieval_tokens.pt")
            val_ret = build_no_retrieval_rows(val_tok, out / "val_no_retrieval_tokens.pt")
            test_ret = build_no_retrieval_rows(test_tok, out / "test_no_retrieval_tokens.pt")
        else:
            train_ret = build_retrieval_rows(model, train_tok, manifest, metadata_by_idx, metadata_by_uid, out / "train_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)
            val_ret = build_retrieval_rows(model, val_tok, manifest, metadata_by_idx, metadata_by_uid, out / "val_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)
            test_ret = build_retrieval_rows(model, test_tok, manifest, metadata_by_idx, metadata_by_uid, out / "test_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)
        json.dump(
            {
                "train": [{k: r.get(k) for k in ("idx", "ec_list", "retrieval_label", "retrieval_medoid_idx", "retrieval_start0", "retrieval_end0", "retrieval_support_n", "retrieval_motif_len")} for r in train_ret],
                "val": [{k: r.get(k) for k in ("idx", "ec_list", "retrieval_label", "retrieval_medoid_idx", "retrieval_start0", "retrieval_end0", "retrieval_support_n", "retrieval_motif_len")} for r in val_ret],
                "test": [{k: r.get(k) for k in ("idx", "ec_list", "retrieval_label", "retrieval_medoid_idx", "retrieval_start0", "retrieval_end0", "retrieval_support_n", "retrieval_motif_len")} for r in test_ret],
            },
            open(out / "retrieval_summary.json", "w"),
            indent=2,
        )
    barrier(world_size)
    train_tok = tokenize_pdbs(model, train_rows, out / "train_dplm2_tokens.pt")
    val_tok = tokenize_pdbs(model, val_rows, out / "val_dplm2_tokens.pt")
    test_tok = tokenize_pdbs(model, test_rows, out / "test_dplm2_tokens.pt")
    if args.no_retrieval:
        train_ret = build_no_retrieval_rows(train_tok, out / "train_no_retrieval_tokens.pt")
        val_ret = build_no_retrieval_rows(val_tok, out / "val_no_retrieval_tokens.pt")
        test_ret = build_no_retrieval_rows(test_tok, out / "test_no_retrieval_tokens.pt")
    else:
        train_ret = build_retrieval_rows(model, train_tok, manifest, metadata_by_idx, metadata_by_uid, out / "train_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)
        val_ret = build_retrieval_rows(model, val_tok, manifest, metadata_by_idx, metadata_by_uid, out / "val_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)
        test_ret = build_retrieval_rows(model, test_tok, manifest, metadata_by_idx, metadata_by_uid, out / "test_retrieval_tokens.pt", out / "retrieval_pdbs", forbidden_ids, label_to_id, anchor_map, args.anchor_radius, args.max_anchor_groups)

    device = next(model.parameters()).device
    text_embed_map = load_text_embedding_map(args.text_embed_pt)
    train_ret = attach_text_embeddings(train_ret, text_embed_map)
    val_ret = attach_text_embeddings(val_ret, text_embed_map)
    test_ret = attach_text_embeddings(test_ret, text_embed_map)
    train_ret = attach_ec_aux_labels(train_ret, ec_aux_label_to_id, args.ec_aux_depth)
    val_ret = attach_ec_aux_labels(val_ret, ec_aux_label_to_id, args.ec_aux_depth)
    test_ret = attach_ec_aux_labels(test_ret, ec_aux_label_to_id, args.ec_aux_depth)
    train_pack = make_pack(model, train_ret, args.hash_dim, device, args.use_retrieval_prefix)
    val_pack = make_pack(model, val_ret, args.hash_dim, device, args.use_retrieval_prefix)
    MR_EVAL_IDX = {"val": [str(r["idx"]) for r in val_ret],
                   "test": [str(r["idx"]) for r in test_ret]}   # MR_STAGED_EVAL
    test_pack = make_pack(model, test_ret, args.hash_dim, device, args.use_retrieval_prefix)
    train_eval_pack = slice_pack(train_pack, args.train_eval_limit)

    if args.site_pairs:   # GROUNDING
        _sp = torch.load(args.site_pairs, map_location="cpu")
        globals()["SITE_PAIRS"] = _sp["pairs"]
        if is_main:
            _n2 = sum(len(v.get(2, [])) for v in _sp["pairs"].values()) / max(len(_sp["pairs"]), 1)
            print(f"[GROUND] {len(_sp['pairs'])} records, {_n2:.1f} pairs per record at stage 2, "
                  f"mode={args.ground_mode}", flush=True)

    def _qwen_hidden_size(q):
        """Qwen2.5 puts hidden_size at the top of the config; Qwen3.5 is multimodal and puts it
        under text_config. Reading the wrong one silently builds the conditioner at the wrong
        width."""
        c = q.config
        for probe in (c, getattr(c, "text_config", None)):
            v = getattr(probe, "hidden_size", None) if probe is not None else None
            if v:
                return int(v)
        raise RuntimeError("cannot determine the text model's hidden size")


    # ---------------- JOINT_LOOP: the text model enters the graph --------------------------------
    if args.qwen_path:
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        _q = AutoModelForCausalLM.from_pretrained(
            args.qwen_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        _q = get_peft_model(_q, LoraConfig(
            r=args.qwen_lora_r, lora_alpha=2 * args.qwen_lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        if args.qwen_grad_ckpt:
            # measured on Qwen3.5-9B: no memory saved, 30-40% slower. 24 of its 32 layers are
            # linear-attention and their memory is in the recurrence, not in the block activations
            # checkpointing recomputes. Kept for architectures where it does help.
            _q.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            _q.enable_input_require_grads()
            print("[JOINT] gradient checkpointing ON for the text model", flush=True)
        _q.cuda()
        globals()["QWEN"] = _q
        _d = torch.load(args.qwen_docs, map_location="cpu")
        globals()["QWEN_DOCS"] = _d["ids"]
        globals()["QWEN_SPANS"] = _d["spans"]
        globals()["QWEN_HARD"] = _d["hard"]
        if is_main:
            _tr = sum(p.numel() for p in _q.parameters() if p.requires_grad)
            print(f"[JOINT] text model {args.qwen_path.split('/')[-3] if '/' in args.qwen_path else args.qwen_path}"
                  f" hidden={_qwen_hidden_size(_q)} LoRA r={args.qwen_lora_r} "
                  f"trainable={_tr/1e6:.1f}M docs={len(_d['ids'])}", flush=True)

    text_dim = _qwen_hidden_size(QWEN) if QWEN is not None else int(train_pack[4].shape[1])

    def _check_width(name, got):
        """Any artefact narrower or wider than the text model is from a different model."""
        if got is not None and int(got) != int(text_dim):
            raise SystemExit(
                f"[WIDTH] {name} is {int(got)}-dimensional but the text model is {int(text_dim)}. "
                f"This file was built for a different text model; rebuild it before training.")

    if QWEN_HIDDEN_MU is not None:
        _mu0 = (next(iter(QWEN_HIDDEN_MU.values())) if isinstance(QWEN_HIDDEN_MU, dict)
                else QWEN_HIDDEN_MU)
        _check_width("--qwen_hidden_mean", torch.as_tensor(_mu0).shape[-1])
    if MR_TEXT_EMB:
        _check_width("--perstage_emb_pt", globals().get("_MR_EMB_WIDTH"))
    if is_main:
        print(f"[WIDTH] text model {text_dim}; every loaded text artefact matches", flush=True)
    ground = None
    if args.ground_mode == "gated":   # GROUNDING
        ground = GroundingInjector(model.net.config.hidden_size).cuda()
        if args.ground_noise_pi >= 0.0:
            globals()["SITE_NOISE_PI"] = args.ground_noise_pi
        globals()["GROUND_CUMULATIVE"] = bool(args.ground_cumulative)
        globals()["GROUND_NOISE_FALLBACK"] = int(args.ground_noise_fallback)
        if args.ground_noise_fallback:
            _f = args.ground_noise_fallback
            assert _f in SITE_NOISE_PROFILE, f"--ground_noise_fallback {_f} has no profile either"
            for _k in (2, 3, 4, 5):
                if _k not in SITE_NOISE_PROFILE:
                    SITE_NOISE_PROFILE[_k] = dict(SITE_NOISE_PROFILE[_f])
                    SITE_NOISE_BETA[_k] = SITE_NOISE_BETA[_f]
                    print(f"[GROUND_NOISE] stage {_k} has no fitted profile; BORROWING stage {_f}'s "
                          f"({SITE_NOISE_PROFILE[_f]['exact']:.3f} exact). Refit before the joint "
                          f"stage or this arm reports a writer accuracy it never measured.",
                          flush=True)
        if args.ground_noise:
            # a stage whose pairs exist but whose profile does not is returned UNCHANGED by
            # corrupt_site_pairs, i.e. injected as an oracle, with no error. Crash instead.
            _have = {k for d in SITE_PAIRS.values() for k in d}
            _miss = sorted(k for k in _have if k in (2, 3, 4, 5) and k not in SITE_NOISE_PROFILE)
            assert not _miss, (f"--ground_noise is on and stages {_miss} carry pairs with no fitted "
                              f"noise profile: they would be injected as an ORACLE. Run refit_noise.py "
                              f"for those stages, or pass --ground_noise_fallback 3.")
        if args.ground_noise:
            import random as _pyrandom
            globals()["GROUND_RNG"] = _pyrandom.Random(args.seed + 4231 * rank)
            if is_main:
                print(f"[GROUND] pairs degraded per record (beta-binomial, "
                      + (f"pi={args.ground_noise_pi}" if args.ground_noise_pi >= 0
                         else "fitted to the measured profile")
                      + f"); dropped entirely with probability {args.ground_drop_prob}", flush=True)
    if args.use_deep_per_token_cross:
        if not args.per_token_text:
            raise SystemExit("--use_deep_per_token_cross needs --per_token_text: it cross-attends "
                             "the text token sequence, and without that flag the features are pooled")
        cond = DeepPerTokenCrossConditioner(
            text_dim,
            model.net.config.hidden_size,
            model.net.config.num_attention_heads,
            model.net.config.num_hidden_layers,
            args.deep_cross_layers,
            args.max_text_tokens,
            len(ec_aux_labels) if DEEP_TEXT_EC > 0 else 0,
            model.net.config.vocab_size if DEEP_TEXT_EC > 0 else 0,
        ).cuda()
        if DEEP_TEXT_EC > 0 and is_main:
            print(f"[TEXT-EC] class head on {len(ec_aux_labels)} EC labels, "
                  f"logit residual rank {DEEP_LOGIT_RANK} into vocab "
                  f"{model.net.config.vocab_size}, CE weight {DEEP_TEXT_EC}", flush=True)
        if DEEP_TEXT_EC > 0 and not ec_aux_labels:
            raise SystemExit("[TEXT-EC] DEEP_TEXT_EC > 0 but the EC label vocabulary is empty; "
                             "the head would be built with n_ec=0 and silently do nothing")
        if is_main:
            _n = model.net.config.num_hidden_layers
            _sel = sorted(cond.layer_ids)
            print(f"[COND] deep per-token text cross-attention: {len(_sel)} of {_n} layers "
                  f"({_sel[0]}..{_sel[-1]}), up to {args.max_text_tokens} text tokens "
                  f"(text_dim={text_dim})", flush=True)
    elif args.use_deep_text_cross:
        cond = DeepTextCrossConditioner(
            text_dim,
            model.net.config.hidden_size,
            model.net.config.num_attention_heads,
            model.net.config.num_hidden_layers,
            args.deep_cross_text_tokens,
            args.deep_cross_layers,
        ).cuda()
    elif args.use_mmdit_text and args.per_token_text:
        cond = MMDiTPerTokenConditioner(
            text_dim,
            model.net.config.hidden_size,
            model.net.config.num_attention_heads,
            args.max_text_tokens,
            args.mmdit_layers,
        ).cuda()
        if is_main:
            print(f"[COND] per-token text conditioning, up to {args.max_text_tokens} tokens "
                  f"(text_dim={text_dim})", flush=True)
    elif args.use_mmdit_text:
        if args.use_logit_residual:
            cond = MMDiTLogitConditioner(
                text_dim,
                model.net.config.hidden_size,
                model.net.config.num_attention_heads,
                model.net.config.vocab_size,
                args.mmdit_text_tokens,
                args.mmdit_layers,
                args.logit_residual_rank,
                args.logit_residual_init_scale,
            ).cuda()
        else:
            cond = MMDiTTextConditioner(
                text_dim,
                model.net.config.hidden_size,
                model.net.config.num_attention_heads,
                args.mmdit_text_tokens,
                args.mmdit_layers,
            ).cuda()
    elif args.use_codefp_cross_attn:
        cond = CodeFPCrossConditioner(
            text_dim,
            model.net.config.hidden_size,
            model.net.config.num_attention_heads,
            args.cross_text_tokens,
        ).cuda()
    else:
        cond = TextRetrievalConditioner(text_dim, model.net.config.hidden_size).cuda()
    motif_head = MotifLabelHead(model.net.config.hidden_size, len(labels)).cuda() if args.lsfs_gamma > 0 else None
    ec_head = ECLabelHead(model.net.config.hidden_size, len(ec_aux_labels)).cuda() if args.ec_aux_gamma > 0 and ec_aux_labels else None
    if args.ec_aux_gamma > 0 and ec_head is None:
        raise RuntimeError("ec_aux_gamma > 0 but EC auxiliary label vocabulary is empty")
    if args.nce_gamma != 0.0:
        globals()["NCE_HEADS"] = CrossModalNCE(model.net.config.hidden_size, text_dim,
                                               proj=args.nce_proj).cuda()
        NCE_GAMMA[0] = float(args.nce_gamma)
        globals()["NCE_RATIO"] = float(args.nce_ratio)
        globals()["NCE_K"] = int(args.nce_k); globals()["NCE_TOL"] = int(args.nce_tol)
        assert args.nce_bank, "--nce_gamma needs --nce_bank: a live text target reintroduces the "\
                              "recency shortcut the freshness control already caught"
        globals()["NCE_BANK"] = load_text_bank(args.nce_bank, "cuda", train_rows)
        _check_width("--nce_bank", NCE_BANK["z"].shape[-1])   # checked where it exists
        if is_main:
            print(f"[NCE] frozen text bank: {NCE_BANK['z'].shape[0]} records, "
                  f"dim {NCE_BANK['z'].shape[1]}, from {os.path.basename(args.nce_bank)}", flush=True)
        NCE_WS[0] = int(world_size); NCE_RANK[0] = int(rank)
        if is_main:
            _np = sum(p.numel() for p in NCE_HEADS.parameters()) / 1e6
            print(f"[NCE] cross-modal InfoNCE on, {_np:.1f}M projection params, gathered batch "
                  f"{world_size * args.batch}; negatives are {args.nce_k} texts within "
                  f"{args.nce_tol} residues, chance top-1 {1.0/(1+args.nce_k):.4f}; "
                  + ("gamma calibrated at the first joint step" if args.nce_gamma < 0
                     else f"gamma pinned at {args.nce_gamma}"), flush=True)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        if QWEN is not None:   # JOINT_SAVE_LORA
            _lw = ckpt.get("qwen_lora")
            if _lw:
                _qs = QWEN.state_dict()
                _bad = [k for k, v in _lw.items() if k not in _qs or _qs[k].shape != v.shape]
                assert not _bad, f"Qwen restore mismatch: {_bad[:10]}"
                _miss, _unexp = QWEN.load_state_dict(_lw, strict=False)
                assert not _unexp, _unexp
                # LAMBDA_PENDING_FIX: a stage-1 checkpoint carries adapters but no lambda,
                # and leaving calibration off would pin it at 1.0 for a freshly built conditioner
                if "qwen_lambda" in ckpt:
                    globals()["QWEN_LAMBDA"] = float(ckpt["qwen_lambda"])
                    QWEN_LAMBDA_PENDING[0] = False
                if is_main:
                    print(f"[JOINT] restored {len(_lw)} adapter tensors; "
                          + (f"lambda={QWEN_LAMBDA:.3f} from the file"
                             if not QWEN_LAMBDA_PENDING[0]
                             else "no lambda in the file, so it is calibrated at unfreezing"),
                          flush=True)
            elif is_main:
                print("[JOINT] this checkpoint carries no adapters: the conditioner will be "
                      "paired with the base text model, which is not the trained pair", flush=True)
        if ckpt.get("ground") is not None and ground is not None:   # SAVE_COMPLETENESS
            ground.load_state_dict(ckpt["ground"])
            if args.ground_gate_scale != 1.0:   # GATE_SCALE
                with torch.no_grad():
                    ground.gate_struct.mul_(args.ground_gate_scale)
                    ground.gate_aa.mul_(args.ground_gate_scale)
            if is_main:
                print(f"[GROUND] restored injector, gates "
                      f"struct={float(ground.gate_struct):+.4f} aa={float(ground.gate_aa):+.4f}"
                      + (f"  (scaled by {args.ground_gate_scale})"
                         if args.ground_gate_scale != 1.0 else ""),
                      flush=True)
        elif ground is not None and is_main:
            print("[GROUND] this checkpoint carries no injector: its gates start at zero, which "
                  "is an exact identity, so grounding will do nothing", flush=True)
        # MR_STAGE_EMB does not exist yet - it is built below, once QWEN's width is known - so this
        # restore is deferred rather than attempted here, where its guard would silently fail.
        globals()["_PENDING_STAGE_EMB"] = ckpt.get("stage_emb")
        if "cond" in ckpt:
            missing, unexpected = cond.load_state_dict(ckpt["cond"], strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            _want = len(cond.state_dict())
            _got = _want - len(missing)
            if _got < max(1, _want // 2):
                _ck = sorted(ckpt["cond"].keys())
                _mk = sorted(cond.state_dict().keys())
                raise SystemExit(
                    "[COND-MISMATCH] the checkpoint's conditioner is not the one that was built: "
                    f"matched {_got} of {_want} tensors "
                    f"({len(missing)} missing, {len(unexpected)} unexpected).\n"
                    f"  checkpoint has: {_ck[:5]} ... ({len(_ck)} tensors)\n"
                    f"  built instead : {_mk[:5]} ... ({len(_mk)} tensors)\n"
                    "  A checkpoint trained with --use_deep_per_token_cross must be loaded with that "
                    "flag too (plus --deep_cross_layers), and one trained with --use_mmdit_text "
                    "without it. Loading would otherwise proceed with a randomly initialised "
                    "conditioner and every number after this point would be meaningless."
                )
            if is_main:
                print(
                    f"loaded cond from {args.init_checkpoint}; missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if "model_trainable" in ckpt:
            _ms = model.state_dict()
            _bad = [k for k, v in ckpt["model_trainable"].items() if k not in _ms or _ms[k].shape != v.shape]
            assert not _bad, f"DPLM restore mismatch: {_bad[:10]}"
            missing, unexpected = model.load_state_dict(ckpt["model_trainable"], strict=False)
            assert not unexpected, unexpected
            if is_main:
                print(
                    f"loaded model_trainable from {args.init_checkpoint}; missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        if motif_head is not None and ckpt.get("motif_head") is not None:
            motif_head.load_state_dict(ckpt["motif_head"], strict=False)
        if NCE_HEADS is not None and ckpt.get("nce_heads") is not None:
            NCE_HEADS.load_state_dict(ckpt["nce_heads"], strict=False)
            if ckpt.get("nce_gamma"):
                NCE_GAMMA[0] = float(ckpt["nce_gamma"]); NCE_PENDING[0] = False
            if is_main:
                print(f"[NCE] projections restored, gamma {NCE_GAMMA[0]:.4f} from the file",
                      flush=True)
    if args.objective == "multiround":
        # JOINT_FIX3: size the stage bias to whatever actually feeds the conditioner. With a text
        # model in the graph that is its hidden width, not the width of the offline vectors, which
        # are no longer used at all in that configuration.
        _td = _qwen_hidden_size(QWEN) if QWEN is not None else train_pack[4].shape[-1]
        globals()["MR_STAGE_EMB"] = torch.nn.Embedding(6, _td).to(device)
        torch.nn.init.normal_(MR_STAGE_EMB.weight, std=0.02)  # MR_FIX4: zero-init left the model
        _pend = globals().get("_PENDING_STAGE_EMB")
        if _pend is not None:
            _before = MR_STAGE_EMB.weight.detach().clone()
            MR_STAGE_EMB.load_state_dict(_pend)
            _delta = float((MR_STAGE_EMB.weight - _before).abs().mean())
            if is_main:
                print(f"[STAGE-EMB] restored from checkpoint; mean|delta| vs fresh init {_delta:.4f} "
                      f"(a delta near 0 would mean the tensor never arrived)", flush=True)
        elif is_main:
            print("[STAGE-EMB] no stage embedding in the checkpoint: rounds start indistinguishable",
                  flush=True)
        # with no way to tell the rounds apart, since the per-stage texts are 95% identical after pooling
        # default weights equalise the per-round supervision volume
        if args.stage_weights:
            globals()["MR_STAGE_W"] = {int(k): float(v) for k, v in
                                       (kv.split(":") for kv in args.stage_weights.split(","))}
        else:
            _L = train_pack[3]
            _c = max(1.0, sum(len(v.get("core", [])) for v in MR_STAGE_POS.values()) / max(len(MR_STAGE_POS), 1))
            _p = _c + max(1.0, sum(len(v.get("pocket", [])) for v in MR_STAGE_POS.values()) / max(len(MR_STAGE_POS), 1))
            globals()["MR_STAGE_W"] = {2: 1.5, 3: 1.2, 4: 1.0, 5: 1.0}   # MR_FIX4
        _w = MR_STAGE_W; _m = sum(_w.values()) / max(len(_w), 1)
        globals()["MR_STAGE_W"] = {k: v / _m for k, v in _w.items()}   # MR_NORMW: mean 1, ratios kept
        print(f"[MULTIROUND] stage weights (raw) = { {k: round(v,2) for k,v in _w.items()} }", flush=True)
        print(f"[MULTIROUND] stage weights (norm) = { {k: round(v,3) for k,v in MR_STAGE_W.items()} }", flush=True)
    def _trainable_of(m):
        """Only what the optimiser is allowed to update; frozen tensors stay out of every group."""
        return [] if m is None else [p for p in m.parameters() if p.requires_grad]

    params = (
        _trainable_of(cond)
        + _trainable_of(NCE_HEADS)
        + _trainable_of(ground)
        + _trainable_of(MR_STAGE_EMB)
        + _trainable_of(motif_head)
        + _trainable_of(ec_head)
        + [p for p in model.parameters() if p.requires_grad]
    )
    broadcast_params(params, world_size)
    torch.manual_seed(args.seed + 1009 * rank)
    np.random.seed(args.seed + 1009 * rank)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    if args.cond_lr > 0:
        # MR_LR_GROUPS: randomly initialised modules need a far higher rate than pretrained layers
        _new = (_trainable_of(cond)
                + _trainable_of(NCE_HEADS)
                + _trainable_of(ground)
                + _trainable_of(MR_STAGE_EMB)
                + _trainable_of(motif_head)
                + _trainable_of(ec_head))
        _qwen_p = [p for p in (QWEN.parameters() if QWEN is not None else []) if p.requires_grad]
        _new_ids = {id(p) for p in _new} | {id(p) for p in _qwen_p}
        _back = [p for p in params if id(p) not in _new_ids]
        _groups = [{"params": _new, "lr": args.cond_lr},
                   {"params": _back, "lr": args.lr}]
        if _qwen_p:
            _groups.append({"params": _qwen_p, "lr": args.qwen_lr})
        opt = torch.optim.AdamW(_groups, weight_decay=0.01)
        if is_main:
            print(f"[LR] cond/new modules lr={args.cond_lr} ({sum(p.numel() for p in _new)/1e6:.1f}M params) | "
                  f"backbone lr={args.lr} ({sum(p.numel() for p in _back)/1e6:.1f}M params)"
                  + (f" | text model lr={args.qwen_lr} ({sum(p.numel() for p in _qwen_p)/1e6:.1f}M params)"
                     if _qwen_p else ""), flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    wb = setup_wandb(args) if is_main else None
    # _wb() runs at module scope and cannot see this local; publishing it is what
    # makes the helper work at all. Without this every log call is a silent no-op.
    globals()["_WB_RUN"] = wb

    def _check_checkpoint_covers_trainables():
        """Every tensor the optimiser updates must appear in the file, by shape and count."""
        saved = 0
        for grp in ([cond] + ([] if ground is None else [ground])
                    + ([] if MR_STAGE_EMB is None else [MR_STAGE_EMB])
                    + ([] if motif_head is None else [motif_head])
                    + ([] if ec_head is None else [ec_head])
                    + ([] if NCE_HEADS is None else [NCE_HEADS])):
            saved += sum(p.numel() for p in grp.parameters() if p.requires_grad)
        saved += sum(v.numel() for k, v in model.state_dict().items() if k in trainable_names)
        if QWEN is not None:
            saved += sum(p.numel() for p in QWEN.parameters() if p.requires_grad)
        trained = sum(p.numel() for p in params)
        if QWEN is not None:
            trained += sum(p.numel() for p in QWEN.parameters() if p.requires_grad)
        if abs(saved - trained) > 0:
            raise RuntimeError(
                f"the checkpoint would omit trained parameters: {trained/1e6:.2f}M are updated, "
                f"{saved/1e6:.2f}M would be written. A file that cannot reconstruct the model is "
                f"worse than none, because it looks usable.")
        if is_main:
            print(f"[CKPT] every trained tensor is covered: {trained/1e6:.2f}M", flush=True)

    _check_checkpoint_covers_trainables()

    log = out / "train.log"
    n = train_pack[0].shape[0]
    best_val = -1.0
    P1_ROW2IDX = [str(r["idx"]) for r in train_rows]
    if args.ground_eval_only and is_main:   # GROUND_EVAL_ONLY
        model.eval()
        if QWEN is not None:
            QWEN.eval()   # lora_dropout=0.05 must not fire at inference
        _n2 = sum(len(v.get(2, [])) for v in SITE_PAIRS.values()) / max(len(SITE_PAIRS), 1)
        print(f"[GROUND-EVAL] pair table: {len(SITE_PAIRS)} records, {_n2:.1f} at stage 2, "
              f"mode={args.ground_mode}", flush=True)
        _r = multiround_evaluate(model, cond, test_pack, MR_EVAL_IDX["test"],
                                 batch_size=args.eval_batch_size, limit=args.mr_eval_limit,
                                 ground=ground)
        print("  [GROUND-EVAL] test " +
              " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}" for k, v in sorted(_r.items())), flush=True)
        print("  [GROUND-EVAL] done", flush=True)
        sys.exit(0)

    if args.mr_ablate_text and is_main:   # MR_ABLATE_PRELOOP
        # Same weights, same inputs, three conditioning vectors. "shuffled" gives the model another
        # enzyme's text - fluent, in-distribution, and wrong - which is what separates "reads the
        # text" from "learned that a sparse reveal looks like a catalytic core".
        import random as _rnd
        model.eval()
        if QWEN is not None:
            QWEN.eval()   # lora_dropout=0.05 must not fire at inference
        _ids = sorted({k.split("#")[0] for k in MR_TEXT_EMB})
        _perm = _ids[:]; _rnd.Random(0).shuffle(_perm)
        _map = dict(zip(_ids, _perm))
        _shuf = {}
        for _k, _v in MR_TEXT_EMB.items():
            _i, _st = _k.split("#")
            _shuf[f"{_map.get(_i, _i)}#{_st}"] = _v
        _saved = MR_TEXT_EMB
        # MR_ABLATE5: same enzyme, text of another stage - tests whether the stage matters
        _rot = {2: 4, 3: 5, 4: 2, 5: 3}
        _stg = {}
        for _k in MR_TEXT_EMB:
            _i, _st = _k.split("#")
            _n = int(_st[1:]) if _st[1:].isdigit() else 0
            _src = f"{_i}#s{_rot.get(_n, _n)}"
            _stg[_k] = MR_TEXT_EMB.get(_src, MR_TEXT_EMB[_k])

        # MR_ABLATE5: another enzyme of the SAME EC class - tests chemistry against family prior
        _ev = MR_EVAL_IDX["test"][:args.mr_eval_limit or None]
        _ec = {}
        for _i in _ev:
            _p = os.path.join(MR_CLAIMS_ROOT, f"cot_{_i}", "claims.json")
            try:
                _ec[_i] = json.load(open(_p)).get("ec_resolved") or json.load(open(_p)).get("ec") or "?"
            except Exception:
                _ec[_i] = "?"
        _unk = sum(1 for _v in _ec.values() if _v == "?")
        assert _unk < 0.1 * max(len(_ec), 1), (
            f"{_unk}/{len(_ec)} eval records have no readable claims.json under "
            f"MR_CLAIMS_ROOT={MR_CLAIMS_ROOT}; the same-EC ablation would quietly swap with a "
            f"RANDOM protein and still print a number")
        _byec = {}
        for _i, _e in _ec.items():
            _byec.setdefault(_e, []).append(_i)
        _ecmap, _npair = {}, 0
        for _e, _grp in _byec.items():
            if len(_grp) < 2 or _e == "?":
                for _i in _grp:
                    _ecmap[_i] = _i
                continue
            _rot2 = _grp[1:] + _grp[:1]          # every record gets a different partner of its class
            for _a, _b in zip(_grp, _rot2):
                _ecmap[_a] = _b; _npair += 1
        _ecemb = dict(MR_TEXT_EMB)
        for _k in list(MR_TEXT_EMB):
            _i, _st = _k.split("#")
            _j = _ecmap.get(_i)
            if _j and _j != _i:
                _src = f"{_j}#{_st}"
                if _src in MR_TEXT_EMB:
                    _ecemb[_k] = MR_TEXT_EMB[_src]
        print(f"  [MR-ABLATE] same-EC: {_npair}/{len(_ev)} evaluated records had a same-class "
              f"partner to swap with ({len(_byec)} classes)", flush=True)

        if QWEN is not None:   # JOINT_ABLATE: swap documents, not table entries
            _ev = MR_EVAL_IDX["test"][:args.mr_eval_limit or None]
            _perm2 = _ev[:]; _rnd.Random(1).shuffle(_perm2)
            _ec = {}
            for _i in _ev:
                try:
                    _m = json.load(open(os.path.join(MR_CLAIMS_ROOT, f"cot_{_i}", "claims.json")))
                    _ec[_i] = _m.get("ec_resolved") or _m.get("ec") or "?"
                except Exception:
                    _ec[_i] = "?"
            _unk = sum(1 for _v in _ec.values() if _v == "?")
            assert _unk < 0.1 * max(len(_ec), 1), (
                f"{_unk}/{len(_ec)} eval records have no readable claims.json under "
                f"MR_CLAIMS_ROOT={MR_CLAIMS_ROOT}; the document-swap ablation would quietly swap "
                f"with a RANDOM protein and still print a number")
            _byec = {}
            for _i, _e in _ec.items():
                _byec.setdefault(_e, []).append(_i)
            _ecpair, _npair = {}, 0
            for _e, _grp in _byec.items():
                if len(_grp) < 2 or _e == "?":
                    for _i in _grp:
                        _ecpair[_i] = _i
                    continue
                for _a, _b in zip(_grp, _grp[1:] + _grp[:1]):
                    _ecpair[_a] = _b; _npair += 1
            print(f"  [MR-ABLATE] same-EC: {_npair}/{len(_ev)} records had a same-class partner "
                  f"({len(_byec)} classes)", flush=True)
            _rot = {2: 4, 3: 5, 4: 2, 5: 3}
            _conds = [
                ("real",       None,                          None, False),
                ("shuffled",   _perm2,                        None, False),
                ("stage-shuf", None,                          _rot, False),
                ("same-EC",    [_ecpair[i] for i in _ev],     None, False),
                ("zero",       None,                          None, True),
            ]
            for _tag, _tids, _smap, _zero in _conds:
                _a = multiround_evaluate(model, cond, test_pack, MR_EVAL_IDX["test"],
                                         batch_size=args.eval_batch_size,
                                         limit=args.mr_eval_limit,
                                         text_ids=_tids, stage_map=_smap, zero_text=_zero)
                print(f"  [MR-ABLATE] test/{_tag:11s} " +
                      " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}" for k, v in sorted(_a.items())),
                      flush=True)
        if MR_ABLATE_CHANNELS:
            # ground is the regex pathway: (residue, position) pairs read out of the text and written
            # straight into the input embedding. It and the text carry overlapping information, so
            # each one measured alone reads higher than its marginal contribution.
            print(f"  [MR-CHANNELS] injector present: {ground is not None}", flush=True)
            for _tag, _zero, _g in (("text+inject ", False, ground),
                                    ("inject only ", True,  ground),
                                    ("text only   ", False, None),
                                    ("neither     ", True,  None)):
                if _g is None and ground is None and _tag.startswith("text+"):
                    continue
                _a = multiround_evaluate(model, cond, test_pack, MR_EVAL_IDX["test"],
                                         batch_size=args.eval_batch_size,
                                         limit=args.mr_eval_limit,
                                         zero_text=_zero, ground=_g)
                print(f"  [MR-CHANNELS] test/{_tag} " +
                      " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}" for k, v in sorted(_a.items())),
                      flush=True)
        print("  [MR-ABLATE] done", flush=True)
        sys.exit(0)

    if args.null_probe and is_main:
        # ---------------------------------------------------------------- CFG null diagnostic
        import torch.nn.functional as _F
        _n = min(int(args.null_probe), test_pack[0].shape[0])
        _tgt, _stl = test_pack[1][:_n].to(device), test_pack[3]
        _ids = [r["idx"] for r in test_ret[:_n]]
        _rot = _ids[3:] + _ids[:3]                      # a different record's text, in distribution
        _ma, _mam, _ms, _msm = (test_pack[5][:_n].to(device), test_pack[6][:_n].to(device),
                                test_pack[7][:_n].to(device), test_pack[8][:_n].to(device))
        print(f"[NULL-PROBE] {_n} records, stage 2, comparing three text conditions", flush=True)
        _gm = globals().get("MR_CTX_CORRUPT")
        globals()["MR_CTX_CORRUPT"] = 0.0               # deterministic context, so only text varies
        _rows = [mr_build_round(model, _tgt[b], _stl, _ids[b], 2, device) for b in range(_n)]
        globals()["MR_CTX_CORRUPT"] = _gm
        _x = torch.stack([r[0] for r in _rows]); _lm = torch.stack([r[1] for r in _rows])
        _lm_aa = _lm[:, _stl:]

        def _pred(tf, tmask):
            lg, _ = conditioned_forward(model, cond, _x, tf, _stl, _ma, _mam, _ms, _msm,
                                        text_mask=tmask)
            _, aa = split_conditioned_outputs(model, lg, _stl)
            return _F.log_softmax(aa[:, :, 4:24].float(), dim=-1)

        with torch.no_grad():
            # Qwen over ~1500 tokens per record; outside no_grad this retains activations for a
            # backward that never happens, and 24 records exhaust an 80 GiB card.
            _tf_r, _tm_r = mr_stage_text(_ids, 2, device)
            _tf_o, _tm_o = mr_stage_text(_rot, 2, device)
            _lp = {"real": _pred(_tf_r, _tm_r),
                   "zero": _pred(torch.zeros_like(_tf_r), _tm_r),
                   "other": _pred(_tf_o, _tm_o)}
        _m = _lm_aa.unsqueeze(-1)
        print(f"[NULL-PROBE] scored at {int(_lm_aa.sum())} masked amino-acid positions")
        print(f"[NULL-PROBE] {'condition':<8} {'entropy':>9} {'sym-KL vs real':>16} {'argmax==real':>14}")
        _ar = _lp["real"].argmax(-1)
        for _k in ("real", "zero", "other"):
            _p = _lp[_k]
            _ent = float((-(_p.exp() * _p).sum(-1) * _lm_aa).sum() / _lm_aa.sum())
            if _k == "real":
                _kl, _ag = 0.0, 1.0
            else:
                _a, _b = _lp["real"], _p
                _d = 0.5 * ((_a.exp() * (_a - _b)).sum(-1) + (_b.exp() * (_b - _a)).sum(-1))
                _kl = float((_d * _lm_aa).sum() / _lm_aa.sum())
                _ag = float(((_p.argmax(-1) == _ar) & _lm_aa).sum() / _lm_aa.sum())
            print(f"[NULL-PROBE] {_k:<8} {_ent:>9.4f} {_kl:>16.4f} {_ag:>13.3f}")
        print("[NULL-PROBE] READING: if `zero` has both the highest entropy and a larger sym-KL than")
        print("[NULL-PROBE] `other`, the null is out of distribution and condition dropout is the fix.")
        print("[NULL-PROBE] If `zero` sits between `real` and `other`, the null was fine.")
        import sys as _sys
        _sys.exit(0)

    def _ckpt_state(_step, _best):
        """The dict both save sites write, so a periodic file matches best.pt exactly."""
        return {
            "cond": cond.state_dict(),
            "motif_head": None if motif_head is None else motif_head.state_dict(),
            "ec_head": None if ec_head is None else ec_head.state_dict(),
            "nce_heads": None if NCE_HEADS is None else NCE_HEADS.state_dict(),
            "nce_gamma": NCE_GAMMA[0],
            "ec_aux_labels": ec_aux_labels,
            "model_trainable": {k: v.cpu() for k, v in model.state_dict().items()
                                if k in trainable_names},
            "step": _step,
            "best_val": _best,
            "qwen_lora": (None if QWEN is None else
                          {k: v.detach().cpu() for k, v in QWEN.state_dict().items()
                           if "lora_" in k}),
            "qwen_lambda": QWEN_LAMBDA,
            "ground": None if ground is None else ground.state_dict(),
            "stage_emb": None if MR_STAGE_EMB is None else MR_STAGE_EMB.state_dict(),
        }

    _EPOCH_STATE = {"perm": None, "epoch": -1}
    for step in range(1, args.steps + 1):
        if EPOCH_SAMPLER:
            _per_rank = (n + world_size - 1) // world_size
            _spe = max(1, (_per_rank + args.batch - 1) // args.batch)   # steps per epoch, per rank
            _ep = (step - 1) // _spe
            if _EPOCH_STATE["epoch"] != _ep:
                # same seed on every rank, so perm[rank::world_size] partitions the corpus
                _g = torch.Generator().manual_seed(int(args.seed) * 1000003 + _ep)
                _EPOCH_STATE["perm"] = torch.randperm(n, generator=_g)
                _EPOCH_STATE["epoch"] = _ep
                if is_main:
                    print(f"[EPOCH] epoch {_ep} begins at step {step}: {n} records, "
                          f"{_per_rank}/rank, {_spe} steps/epoch", flush=True)
            _mine = _EPOCH_STATE["perm"][rank::world_size]
            _t = (step - 1) % _spe
            _sel = _mine[_t * args.batch:(_t + 1) * args.batch]
            if _sel.numel() < args.batch:      # ragged tail: wrap inside this rank's own slice
                _sel = torch.cat([_sel, _mine[:args.batch - _sel.numel()]])
            idx = _sel.to(device)
        else:
            idx = torch.randint(0, n, (args.batch,), device=device)
        if MEM_PREFLIGHT_IDX and step <= len(MEM_PREFLIGHT_IDX):
            # every rank takes the same row: the point is the per-rank worst case, and making the
            # ranks differ would only test whichever one happened to draw the longest
            _pf = min(MEM_PREFLIGHT_IDX[step - 1], n - 1)
            idx = torch.full((args.batch,), _pf, dtype=torch.long, device=device)
            if is_main:
                print(f"[PREFLIGHT] step {step}: forcing row {_pf}", flush=True)
        globals()["P1_BATCH_IDX"] = ([P1_ROW2IDX[i] for i in idx.tolist()]
                                     if (P1_CORE_BOOST > 0.0 or P1_STAGE_MASK_P > 0.0 or args.objective == "multiround") else [])
        input_ids, target, loss_mask, st_len, text_feats, motif_aa_ids, motif_aa_mask, motif_st_ids, motif_st_mask, _, _, windows, window_mask, label_ids, ec_label_ids = train_pack
        anchor_loss = torch.tensor(0.0, device=device)
        high_loss = torch.tensor(0.0, device=device)
        if args.objective == "codefp_diffusion":
            gen_loss, struct_loss, aa_loss, hidden, aux = codefp_diffusion_forward(
                model,
                cond,
                target[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                weighting=args.diffusion_weighting,
                text_dropout=args.text_dropout,
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            anchor_struct_loss = struct_loss
            anchor_aa_loss = aa_loss
            high_struct_loss = torch.tensor(0.0, device=device)
            high_aa_loss = torch.tensor(0.0, device=device)
        elif args.objective == "fullmask_anchor":
            gen_loss, struct_loss, aa_loss, hidden, aux = fullmask_forward(
                model,
                cond,
                input_ids[idx],
                target[idx],
                loss_mask[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            anchor_loss = gen_loss
            anchor_struct_loss = struct_loss
            anchor_aa_loss = aa_loss
            high_struct_loss = torch.tensor(0.0, device=device)
            high_aa_loss = torch.tensor(0.0, device=device)
        elif args.objective == "hybrid_anchor_diffusion":
            anchor_loss, anchor_struct_loss, anchor_aa_loss, anchor_hidden, anchor_aux = fullmask_forward(
                model,
                cond,
                input_ids[idx],
                target[idx],
                loss_mask[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            high_loss, high_struct_loss, high_aa_loss, hidden, aux = joint_high_noise_diffusion_forward(
                model,
                cond,
                target[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                timestep_min_frac=args.high_t_min_frac,
                weighting=args.diffusion_weighting,
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            gen_loss = args.fullmask_gamma * anchor_loss + args.high_noise_gamma * high_loss
            struct_loss = args.fullmask_gamma * anchor_struct_loss + args.high_noise_gamma * high_struct_loss
            aa_loss = args.fullmask_gamma * anchor_aa_loss + args.high_noise_gamma * high_aa_loss
        elif args.objective == "multiround":
            opt.zero_grad(set_to_none=True)
            _idxs = [P1_ROW2IDX[i] for i in idx.tolist()]
            _txt_loss = torch.zeros((), device=device)
            _txt_brk = None
            _tmask = None
            _frozen = QWEN is not None and step <= args.qwen_freeze_steps   # JOINT_WARMUP
            if QWEN is not None:   # JOINT_LOOP
                _qx, _qa, _qs, _qh = qwen_batch(_idxs, device)
                _want = (step % max(1, args.loss_log_interval) == 0)
                if _frozen:
                    # a fixed feature extractor: no graph is kept, so the conditioner fits a
                    # stationary target and the step costs less than it will after unfreezing
                    with torch.no_grad():
                        _qhid = QWEN(input_ids=_qx, attention_mask=_qa,
                                     output_hidden_states=True).hidden_states[-1]
                        _txt_loss, _txt_brk = qwen_text_loss(_qhid, _qx, _qs, _qh, breakdown=_want)
                    _txt_loss = _txt_loss.detach()
                else:
                    _qhid = QWEN(input_ids=_qx, attention_mask=_qa,
                                 output_hidden_states=True).hidden_states[-1]
                    _txt_loss, _txt_brk = qwen_text_loss(_qhid, _qx, _qs, _qh, breakdown=_want)
                _tfs, _tmask = {}, {}
                for _k in (2, 3, 4, 5):
                    _tfs[_k], _tmask[_k] = qwen_stage_states(
                        _qhid, _idxs, _k, args.max_text_tokens, device)
                    if not _frozen:
                        _tfs[_k] = scale_grad(_tfs[_k], QWEN_LAMBDA)
            if QWEN is None:
              _tfs = {}
              for _k in (2, 3, 4, 5):
                  _v = []
                  for _ix in _idxs:
                      _e = MR_TEXT_EMB.get(f"{_ix}#s{_k}")
                      _v.append(torch.as_tensor(_e, dtype=torch.float32, device=device)
                                if _e is not None else text_feats[0] * 0.0)
                  _tfs[_k] = torch.stack(_v)
            if QWEN is not None and not _frozen and QWEN_LAMBDA_PENDING[0]:   # JOINT_WARMUP:
                # measured the moment the adapters unfreeze. At step 1 the conditioner is still
                # random, so the structure gradient through it carried no information.
                # JOINT_LAMBDA_FIX: the adapters are where both objectives meet. The text loss
                # reaches them through the language-model head, the structure loss through the
                # conditioner and back along the hidden states; the conditioner alone sees only one.
                _probe = [p for p in QWEN.parameters() if p.requires_grad]
                _gt = torch.autograd.grad(_txt_loss, _probe, retain_graph=True, allow_unused=True)
                _nt = float(torch.sqrt(sum(((g * g).sum() for g in _gt if g is not None),
                                           torch.zeros((), device=device))))
                # CALIB_MEM: one sample only. This forward keeps its whole four-round graph
                # (backward=False), and on a 9B text model the sum of it, the two grad probes and
                # the real step exceeds 80 GB. lambda is a ratio of two norms that both scale with
                # the batch, so a single sample measures the same quantity at half the peak.
                _c = slice(0, 1)
                _tfs_c = {k: v[_c] for k, v in _tfs.items()}
                _tmask_c = (None if _tmask is None else
                            {k: (None if v is None else v[_c]) for k, v in _tmask.items()})
                torch.cuda.empty_cache()
                _s0, _, _, _, _ = multiround_forward(
                    model, cond, target[idx][_c], st_len, _tfs_c, _idxs[_c], device,
                    motif_aa_ids[idx][_c], motif_aa_mask[idx][_c],
                    motif_st_ids[idx][_c], motif_st_mask[idx][_c],
                    struct_gamma=args.struct_loss_gamma, aa_gamma=args.aa_loss_gamma,
                    backward=False, text_mask_by_stage=_tmask_c, ground=ground,
                    ground_noise=GROUND_RNG, ground_drop_prob=args.ground_drop_prob)
                _gs = torch.autograd.grad(_s0, _probe, retain_graph=False, allow_unused=True)
                _ns = float(torch.sqrt(sum(((g * g).sum() for g in _gs if g is not None),
                                           torch.zeros((), device=device))))
                globals()["QWEN_LAMBDA"] = (_nt / max(_ns, 1e-8)) if args.text_lambda < 0 else args.text_lambda
                QWEN_LAMBDA_PENDING[0] = False
                opt.zero_grad(set_to_none=True)
                del _gt, _gs, _s0
                torch.cuda.empty_cache()   # release the probe graphs before the real step
                if is_main:
                    print(f"[JOINT] gradient norms on the conditioner: text {_nt:.4e} "
                          f"structure {_ns:.4e} -> lambda {QWEN_LAMBDA:.4f}", flush=True)
                _qout = QWEN(input_ids=_qx, attention_mask=_qa, output_hidden_states=True)
                _qhid = _qout.hidden_states[-1]
                _txt_loss, _txt_brk = qwen_text_loss(_qhid, _qx, _qs, _qh, breakdown=False)
                for _k in (2, 3, 4, 5):
                    _tfs[_k], _tmask[_k] = qwen_stage_states(
                        _qhid, _idxs, _k, args.max_text_tokens, device)
                    _tfs[_k] = scale_grad(_tfs[_k], QWEN_LAMBDA)
            if QWEN is not None and not _frozen:
                (_txt_loss).backward(retain_graph=True)
            gen_loss, struct_loss, aa_loss, hidden, aux = multiround_forward(
                model, cond, target[idx], st_len, _tfs, _idxs, device,
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                struct_gamma=args.struct_loss_gamma, aa_gamma=args.aa_loss_gamma,
                text_mask_by_stage=_tmask,
                loss_scale=1.0,   # lambda rides on the gradient into the text model instead
                ground=ground, ground_noise=GROUND_RNG,        # NOISE_WIRE_FIX
                ground_drop_prob=args.ground_drop_prob,
            )
            anchor_loss = gen_loss
            anchor_struct_loss = struct_loss; anchor_aa_loss = aa_loss
            high_struct_loss = torch.tensor(0.0, device=device)
            high_aa_loss = torch.tensor(0.0, device=device)
            if step % max(1, args.loss_log_interval) == 0 and is_main:
                if QWEN is not None and _txt_brk is not None:   # JOINT_LOG
                    _tl = (f"  [TEXT] step={step:05d} "
                           f"{'frozen' if _frozen else 'joint '} "
                           f"ce={_txt_brk.get('ce_raw', float(_txt_loss)):.4f} "
                           + " ".join(f"s{k}={_txt_brk[k]:.3f}" for k in (2, 3, 4, 5))
                           + f" residue_tokens={_txt_brk['hard']:.4f} lambda={QWEN_LAMBDA:.2f}")
                    print(_tl, flush=True)
                    if TEXT_EC_STAT[2]:
                        _tn = TEXT_EC_STAT[2]
                        print(f"  [TEXT-EC] step={step:05d} ce={TEXT_EC_STAT[0]/_tn:.4f} "
                              f"acc={TEXT_EC_STAT[1]/_tn:.4f} over {_tn} rounds", flush=True)
                        _wb({"text_ec_ce": TEXT_EC_STAT[0] / _tn,
                             "text_ec_acc": TEXT_EC_STAT[1] / _tn}, step=step)
                        TEXT_EC_STAT[0] = 0.0; TEXT_EC_STAT[1] = 0.0; TEXT_EC_STAT[2] = 0
                    if NCE_STAT[2]:
                        print(f"  [NCE] step={step:05d} loss={NCE_STAT[0]/NCE_STAT[2]:.4f} "
                              f"top1={NCE_STAT[1]/NCE_STAT[2]:.4f} (chance "
                              f"{1.0/max(NCE_STAT[3],1):.4f}) gamma={NCE_GAMMA[0]:.4f} "
                              f"over {NCE_STAT[2]} rounds", flush=True)
                        _wb({"nce_loss": NCE_STAT[0] / NCE_STAT[2],
                             "nce_top1": NCE_STAT[1] / NCE_STAT[2],
                             "nce_chance": 1.0 / max(NCE_STAT[3], 1),
                             "nce_gamma": NCE_GAMMA[0]}, step=step)
                        NCE_STAT[0] = 0.0; NCE_STAT[1] = 0.0; NCE_STAT[2] = 0
                        NCE_STAT[5] = 0.0
                    with open(log, "a") as fh:
                        fh.write(_tl + "\n")
                    if wb is not None:
                        _wb({"text_ce": float(_txt_loss),
                                "text_ce_residue": _txt_brk["hard"],
                                **{f"text_ce_s{k}": _txt_brk[k] for k in (2, 3, 4, 5)}}, step=step)
                print(f"  [MR] per-stage struct loss: " +
                      " ".join(f"s{k}={v:.4f}" for k, v in sorted(aux["per_stage_struct"].items())), flush=True)
        elif args.objective == "hybrid_anchor_codefp":
            anchor_loss, anchor_struct_loss, anchor_aa_loss, anchor_hidden, anchor_aux = fullmask_forward(
                model,
                cond,
                input_ids[idx],
                target[idx],
                loss_mask[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            high_loss, high_struct_loss, high_aa_loss, hidden, aux = codefp_diffusion_forward(
                model,
                cond,
                target[idx],
                st_len,
                text_feats[idx],
                motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                weighting=args.diffusion_weighting,
                text_dropout=args.text_dropout,
                struct_gamma=args.struct_loss_gamma,
                aa_gamma=args.aa_loss_gamma,
            )
            gen_loss = args.fullmask_gamma * anchor_loss + args.high_noise_gamma * high_loss
            struct_loss = args.fullmask_gamma * anchor_struct_loss + args.high_noise_gamma * high_struct_loss
            aa_loss = args.fullmask_gamma * anchor_aa_loss + args.high_noise_gamma * high_aa_loss
        else:
            high_every = max(1, int(args.hybrid_high_every))
            if step % high_every == 0:
                high_loss, high_struct_loss, high_aa_loss, hidden, aux = joint_high_noise_diffusion_forward(
                    model,
                    cond,
                    target[idx],
                    st_len,
                    text_feats[idx],
                    motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                    timestep_min_frac=args.high_t_min_frac,
                    weighting=args.diffusion_weighting,
                    struct_gamma=args.struct_loss_gamma,
                    aa_gamma=args.aa_loss_gamma,
                )
                gen_loss = args.high_noise_gamma * high_loss
                struct_loss = args.high_noise_gamma * high_struct_loss
                aa_loss = args.high_noise_gamma * high_aa_loss
                anchor_struct_loss = torch.tensor(0.0, device=device)
                anchor_aa_loss = torch.tensor(0.0, device=device)
            else:
                anchor_loss, anchor_struct_loss, anchor_aa_loss, hidden, aux = fullmask_forward(
                    model,
                    cond,
                    input_ids[idx],
                    target[idx],
                    loss_mask[idx],
                    st_len,
                    text_feats[idx],
                    motif_aa_ids[idx], motif_aa_mask[idx], motif_st_ids[idx], motif_st_mask[idx],
                    struct_gamma=args.struct_loss_gamma,
                    aa_gamma=args.aa_loss_gamma,
                )
                gen_loss = args.fullmask_gamma * anchor_loss
                struct_loss = args.fullmask_gamma * anchor_struct_loss
                aa_loss = args.fullmask_gamma * anchor_aa_loss
                high_struct_loss = torch.tensor(0.0, device=device)
                high_aa_loss = torch.tensor(0.0, device=device)
        lsfs_loss, lsfs_acc, lsfs_n = lsfs_loss_and_metrics(motif_head, hidden, st_len, windows[idx], window_mask[idx], label_ids[idx])
        ec_labels_for_loss = ec_label_ids[idx]
        if ec_head is not None and "text_keep" in aux:
            ec_labels_for_loss = ec_labels_for_loss.clone()
            ec_labels_for_loss[~aux["text_keep"]] = -1
        ec_loss, ec_acc, ec_n = ec_aux_loss_and_metrics(ec_head, hidden, target[idx], ec_labels_for_loss, model.pad_id)
        # the class term lives inside multiround_forward, where tf is still in the graph
        loss = gen_loss + args.lsfs_gamma * lsfs_loss + args.ec_aux_gamma * ec_loss
        if args.objective != "multiround":             # MR_ZG_FIX: multiround already zeroed and
            opt.zero_grad(set_to_none=True)            # accumulated its per-round gradients; zeroing
            loss.backward()                            # here would discard all of them
        allreduce_grads(params, world_size)
        # JOINT_CLIP: separate thresholds, so a large gradient in one half cannot scale the other down
        _gn_main = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        _gn_text = 0.0
        if QWEN is not None:
            _gn_text = float(torch.nn.utils.clip_grad_norm_(
                [p for p in QWEN.parameters() if p.requires_grad], 1.0))
        if is_main and args.loss_log_interval and step % max(1, args.loss_log_interval) == 0:
            _gl = (f"  [GRAD] step={step:05d} main={_gn_main:.3f} text={_gn_text:.3f} (clip 1.0)"
               + (f" gate_struct={float(ground.gate_struct):+.4f} "
                  f"gate_aa={float(ground.gate_aa):+.4f}" if ground is not None else ""))
            print(_gl, flush=True)
            with open(log, "a") as fh:
                fh.write(_gl + "\n")
            if wb is not None:
                _wb({"grad_norm_main": _gn_main, "grad_norm_text": _gn_text}, step=step)
        opt.step()
        if CKPT_EVERY and is_main and step % CKPT_EVERY == 0:
            _p = out / f"step{step:06d}.pt"
            torch.save(_ckpt_state(step, best_val), _p)
            print(f"  [CKPT] step {step} -> {_p.name}", flush=True)

        do_eval = ((step == 1 and not args.skip_step1_eval) or step % args.eval_interval == 0)
        do_loss_log = bool(args.loss_log_interval and step % args.loss_log_interval == 0)
        if do_eval:
            avg_loss = reduce_float(loss.item(), device, world_size)
            avg_gen = reduce_float(gen_loss.item(), device, world_size)
            avg_struct = reduce_float(struct_loss.item(), device, world_size)
            avg_aa = reduce_float(aa_loss.item(), device, world_size)
            avg_anchor = reduce_float(anchor_loss.item(), device, world_size)
            avg_high = reduce_float(high_loss.item(), device, world_size)
            avg_lsfs = reduce_float(lsfs_loss.item(), device, world_size)
            avg_ec = reduce_float(ec_loss.item(), device, world_size)
            if is_main:
                model.eval()
                if QWEN is not None:
                    QWEN.eval()   # lora_dropout=0.05 must not fire at inference
                if QWEN is not None:   # JOINT_EVAL: these read a pooled vector the
                    # sequence conditioner cannot take, and they score a task this objective
                    # never trains; the staged numbers below are the ones that apply.
                    _z = (float("nan"),) * 8
                    tr_diff = va_diff = te_diff = _z
                    tr = va = te = _z
                else:
                    tr_diff = evaluate_diffusion_loss(model, cond, train_eval_pack, batch_size=args.eval_batch_size, weighting=args.diffusion_weighting, repeats=args.eval_diffusion_repeats)
                    va_diff = evaluate_diffusion_loss(model, cond, val_pack, batch_size=args.eval_batch_size, weighting=args.diffusion_weighting, repeats=args.eval_diffusion_repeats)
                    te_diff = evaluate_diffusion_loss(model, cond, test_pack, batch_size=args.eval_batch_size, weighting=args.diffusion_weighting, repeats=args.eval_diffusion_repeats)
                    tr = evaluate_tokens(model, cond, train_eval_pack, batch_size=args.eval_batch_size, motif_head=motif_head, ec_head=ec_head)
                    va = evaluate_tokens(model, cond, val_pack, batch_size=args.eval_batch_size, motif_head=motif_head, ec_head=ec_head)
                    te = evaluate_tokens(model, cond, test_pack, batch_size=args.eval_batch_size, motif_head=motif_head, ec_head=ec_head)
                if args.objective == "multiround":   # MR_STAGED_EVAL: the task we actually train on
                    _mv = multiround_evaluate(model, cond, val_pack, MR_EVAL_IDX["val"],
                                              batch_size=args.eval_batch_size, limit=args.mr_eval_limit,
                                              ground=ground)
                    if MR_EVAL_SKIP_TEST:
                        _mt = {}
                    else:
                        _mt = multiround_evaluate(model, cond, test_pack, MR_EVAL_IDX["test"],
                                                  batch_size=args.eval_batch_size, limit=args.mr_eval_limit,
                                                  ground=ground)
                    _mf = {}
                    if MR_FREERUN_EVAL:
                        # the same records, scored in the regime generation runs in: round k is
                        # conditioned on round k-1's own argmax rather than on the truth
                        _mf = multiround_evaluate_freerun(
                            model, cond, val_pack, MR_EVAL_IDX["val"],
                            batch_size=args.eval_batch_size, limit=args.mr_eval_limit, ground=ground)
                    _ml = ("  [MR-EVAL] val  " + " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}(n={v[2]})" for k, v in sorted(_mv.items()))
                           + "\n  [MR-EVAL] test " + " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}(n={v[2]})" for k, v in sorted(_mt.items())))
                    if _mf:
                        _ml += ("\n  [MR-FREE] val  "
                                + " ".join(f"s{k}={v[0]:.4f}/{v[1]:.4f}(n={v[2]})" for k, v in sorted(_mf.items())))
                    print(_ml, flush=True)
                    with open(log, "a") as fh:
                        fh.write(_ml + "\n")
                    if wb is not None:
                        _wb({**{f"mr_val_bit_s{k}": v[0] for k, v in _mv.items()},
                                **{f"mr_test_bit_s{k}": v[0] for k, v in _mt.items()},
                                # the aa half moved 0.18 -> 0.51 over the 512 run and was dropped;
                                # train/val/test_acc are nan under multiround so this is the only
                                # validation curve there is
                                **{f"mr_val_aa_s{k}": v[1] for k, v in _mv.items()},
                                **{f"mr_test_aa_s{k}": v[1] for k, v in _mt.items()},
                                # kept under their own names: mixing them into mr_val_* would break
                                # comparability with every run logged before this switch existed
                                **{f"mr_free_bit_s{k}": v[0] for k, v in _mf.items()},
                                **{f"mr_free_aa_s{k}": v[1] for k, v in _mf.items()}}, step=step)
                model.train()
                if QWEN is not None:
                    QWEN.train()   # lora_dropout=0.05 must not fire at inference
                line = (
                    f"step={step:05d} loss={avg_loss:.4f} diff={avg_gen:.4f} aa_loss={avg_aa:.4f} st_loss={avg_struct:.4f} "
                    f"anchor={avg_anchor:.4f} high={avg_high:.4f} "
                    f"lsfs={avg_lsfs:.4f}/{lsfs_acc:.3f}/{lsfs_n} ec={avg_ec:.4f}/{ec_acc:.3f}/{ec_n} "
                    f"diff_eval={tr_diff[0]:.4f}/{va_diff[0]:.4f}/{te_diff[0]:.4f} "
                    f"train_acc={tr[0]:.4f}/{tr[1]:.4f}/{tr[2]:.4f} "
                    f"val_acc={va[0]:.4f}/{va[1]:.4f}/{va[2]:.4f} "
                    f"test_acc={te[0]:.4f}/{te[1]:.4f}/{te[2]:.4f} bit_acc={P1_LAST_BITACC[0]:.4f}(d={P1_LAST_BITACC[1]}) "
                    f"lsfs_eval={tr[3]:.3f}/{va[3]:.3f}/{te[3]:.3f} n={tr[4]}/{va[4]}/{te[4]} "
                    f"ec_eval={tr[5]:.3f}/{va[5]:.3f}/{te[5]:.3f} n={tr[6]}/{va[6]}/{te[6]}"
                )
                print(line, flush=True)
                with open(log, "a") as fh:
                    fh.write(line + "\n")
                if wb is not None:
                    _wb({
                        "loss": avg_loss,
                        "diff_loss": avg_gen,
                        "aa_loss": avg_aa,
                        "struct_loss": avg_struct,
                        "anchor_loss": avg_anchor,
                        "high_noise_loss": avg_high,
                        "lsfs_loss": avg_lsfs,
                        "ec_aux_loss": avg_ec,
                        "lsfs_batch_acc": lsfs_acc,
                        "ec_aux_batch_acc": ec_acc,
                        "train_diff_eval": tr_diff[0], "val_diff_eval": va_diff[0], "test_diff_eval": te_diff[0],
                        "train_diff_aa": tr_diff[1], "val_diff_aa": va_diff[1], "test_diff_aa": te_diff[1],
                        "train_diff_struct": tr_diff[2], "val_diff_struct": va_diff[2], "test_diff_struct": te_diff[2],
                        "train_acc": tr[0], "train_aa_acc": tr[1], "train_st_acc": tr[2],
                        "val_acc": va[0], "val_aa_acc": va[1], "val_st_acc": va[2],
                        "test_acc": te[0], "test_aa_acc": te[1], "test_st_acc": te[2],
                        "train_lsfs_acc": tr[3], "val_lsfs_acc": va[3], "test_lsfs_acc": te[3],
                        "train_ec_aux_acc": tr[5], "val_ec_aux_acc": va[5], "test_ec_aux_acc": te[5],
                    }, step=step)
                _sel = va[0]
                if args.objective == "multiround":   # MR_BEST_SEL: score the task being trained
                    import math as _m
                    _vals = [x for k, v in _mv.items() for x in (v[0], v[1]) if not _m.isnan(x)]
                    _sel = sum(_vals) / len(_vals) if _vals else -1.0
                if _sel > best_val:
                    best_val = _sel
                    torch.save(
                        {
                            "cond": cond.state_dict(),
                            "motif_head": None if motif_head is None else motif_head.state_dict(),
                            "ec_head": None if ec_head is None else ec_head.state_dict(),
                            "nce_heads": None if NCE_HEADS is None else NCE_HEADS.state_dict(),
                            "nce_gamma": NCE_GAMMA[0],
                            "ec_aux_labels": ec_aux_labels,
                            "model_trainable": {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names},
                            "step": step,
                            "best_val": best_val,
                            # JOINT_SAVE_LORA: without these the conditioner would be restored
                            # against a text model it was never fitted to
                            "qwen_lora": (None if QWEN is None else
                                          {k: v.detach().cpu() for k, v in QWEN.state_dict().items()
                                           if "lora_" in k}),
                            "qwen_lambda": QWEN_LAMBDA,
                            # SAVE_COMPLETENESS: modules added to training after the first version
                            "ground": None if ground is None else ground.state_dict(),
                            "stage_emb": None if MR_STAGE_EMB is None else MR_STAGE_EMB.state_dict(),
                        },
                        out / "best.pt",
                    )
            barrier(world_size)
        elif do_loss_log:
            avg_loss = reduce_float(loss.item(), device, world_size)
            avg_gen = reduce_float(gen_loss.item(), device, world_size)
            avg_struct = reduce_float(struct_loss.item(), device, world_size)
            avg_aa = reduce_float(aa_loss.item(), device, world_size)
            avg_anchor = reduce_float(anchor_loss.item(), device, world_size)
            avg_high = reduce_float(high_loss.item(), device, world_size)
            avg_lsfs = reduce_float(lsfs_loss.item(), device, world_size)
            avg_ec = reduce_float(ec_loss.item(), device, world_size)
            if is_main:
                if LOG_MEM and torch.cuda.is_available():
                    _pk = torch.cuda.max_memory_allocated() / 2**30
                    _rs = torch.cuda.max_memory_reserved() / 2**30
                    _tot = torch.cuda.get_device_properties(device).total_memory / 2**30
                    _jl = int(globals().get("_LAST_JOINT_LEN", 0))
                    print(f"step={step:05d} rank={rank} mem_peak={_pk:.1f}G reserved={_rs:.1f}G "
                          f"of {_tot:.0f}G  headroom={_tot - _rs:.1f}G  last_joint_len={_jl}", flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    if wb is not None and is_main:
                        _wb({"mem_peak_gb": _pk, "mem_reserved_gb": _rs}, step=step)
                line = f"step={step:05d} loss={avg_loss:.4f} diff={avg_gen:.4f} aa_loss={avg_aa:.4f} st_loss={avg_struct:.4f} anchor={avg_anchor:.4f} high={avg_high:.4f} lsfs={avg_lsfs:.4f} ec={avg_ec:.4f} no_eval=1"
                print(line, flush=True)
                _cr = cond_report(cond, ground, globals().get("_LAST_TF"), globals().get("_LAST_TMASK"))
                print(f"step={step:05d} {_cr}", flush=True)
                with open(log, "a") as fh:
                    fh.write(f"step={step:05d} {_cr}\n")
                if wb is not None:
                    _js = getattr(cond, "joint_scale", None)
                    if _js is not None:
                        _wb({"joint_scale": float(_js)}, step=step)
                    # the deep path has no joint_scale; its read-out is one gate per adapted layer
                    _sc = getattr(cond, "scales", None)
                    if _sc is not None:
                        _ids = sorted(getattr(cond, "layer_ids", range(len(_sc))))
                        _vals = [float(p) for p in _sc]
                        _wb({"deep_scale_mean": sum(_vals) / max(len(_vals), 1),
                                "deep_scale_absmean": sum(abs(v) for v in _vals) / max(len(_vals), 1),
                                **{f"deep_scale_L{_ids[i]}": v for i, v in enumerate(_vals)}},
                               step=step)
                        print(f"step={step:05d} deep_scales=["
                              + " ".join(f"L{_ids[i]}:{v:+.3f}" for i, v in enumerate(_vals))
                              + "]", flush=True)
                    # with the zero-init the gate is fixed near 1.0; the growing quantity, and so
                    # the one that says whether the text is being used, is the output projection
                    _ad = getattr(cond, "adapters", None)
                    if _ad is not None:
                        _ids2 = sorted(getattr(cond, "layer_ids", range(len(_ad))))
                        _wn = []
                        for _b in _ad:
                            _w = _b["attn"].out_proj.weight
                            _wn.append(float(_w.detach().float().norm()))
                        _wb({"deep_wout_mean": sum(_wn) / max(len(_wn), 1),
                             **{f"deep_wout_L{_ids2[i]}": v for i, v in enumerate(_wn)}}, step=step)
                        print(f"step={step:05d} deep_wout=["
                              + " ".join(f"L{_ids2[i]}:{v:.4f}" for i, v in enumerate(_wn))
                              + "]", flush=True)
                    _lu = getattr(cond, "logit_up", None)
                    if _lu is not None:
                        _ln = float(_lu.weight.detach().float().norm())
                        _en = float(cond.ec_emb.weight.detach().float().norm())
                        _wb({"deep_logit_norm": _ln, "deep_ecemb_norm": _en}, step=step)
                        print(f"step={step:05d} logit_up_norm={_ln:.4f} ec_emb_norm={_en:.4f}",
                              flush=True)
                with open(log, "a") as fh:
                    fh.write(line + "\n")
                if wb is not None:
                    _wb({"loss": avg_loss, "diff_loss": avg_gen, "aa_loss": avg_aa, "struct_loss": avg_struct, "anchor_loss": avg_anchor, "high_noise_loss": avg_high, "lsfs_loss": avg_lsfs, "ec_aux_loss": avg_ec}, step=step)

    if is_main:
        model.eval()
        cond.eval()
        if ground is not None: ground.eval()
        if MR_STAGE_EMB is not None: MR_STAGE_EMB.eval()
        if QWEN is not None:
            QWEN.eval()   # lora_dropout=0.05 must not fire at inference
        if not args.skip_final_generate:
            import contextlib as _ctx
            _amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.gen_bf16 else _ctx.nullcontext())
            print(f"[GEN] bf16_autocast={args.gen_bf16} tf32={torch.backends.cuda.matmul.allow_tf32} "
                  f"batch={args.generate_batch_size} max_iter={args.sample_max_iter} "
                  f"only_test={args.generate_only_test}", flush=True)
            with _amp:
                if not args.generate_only_test:
                    train_gen_pack = slice_pack(train_pack, args.final_generate_train_limit)
                    train_gen_rows = train_ret[: args.final_generate_train_limit] if args.final_generate_train_limit else train_ret
                    generate_pack(model, cond, train_gen_pack, train_gen_rows, out / "generated_train", batch_size=args.generate_batch_size, max_iter=args.sample_max_iter, sampling_strategy=args.sample_strategy, cfg_aa_scale=args.cfg_aa_scale, cfg_struct_scale=args.cfg_struct_scale)
                    generate_pack(model, cond, val_pack, val_ret, out / "generated_val", batch_size=args.generate_batch_size, max_iter=args.sample_max_iter, sampling_strategy=args.sample_strategy, cfg_aa_scale=args.cfg_aa_scale, cfg_struct_scale=args.cfg_struct_scale)
                if args.multiround_generate:
                    multiround_generate_pack(model, cond, test_pack, test_ret, out / "generated_test",
                                             batch_size=args.generate_batch_size,
                                             max_iter=args.sample_max_iter,
                                             sampling_strategy=args.sample_strategy,
                                             cfg_aa=args.cfg_aa_scale,
                                             cfg_struct=args.cfg_struct_scale,
                                             cfg_null_k=args.cfg_null_k,
                                             ground=ground)
                else:
                    generate_pack(model, cond, test_pack, test_ret, out / "generated_test", batch_size=args.generate_batch_size, max_iter=args.sample_max_iter, sampling_strategy=args.sample_strategy, cfg_aa_scale=args.cfg_aa_scale, cfg_struct_scale=args.cfg_struct_scale)
        torch.save({"done": True, "best_val": best_val}, out / "done.pt")
        if wb is not None:
            wb.finish()
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
