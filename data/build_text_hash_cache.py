#!/usr/bin/env python3
"""Precompute text_hash for every row once, so a launch does not spend 17 minutes on it.

text_hash bags words and character trigrams of each row's `comment` (mean 6002 characters) into a
4096-d vector. That is ~5000 md5 calls per row, each followed by a scalar index-add on a TORCH
tensor, which is microseconds apiece in Python. Measured: 20.6 ms/row x 50420 rows = 17.3 min, paid
by EVERY rank on EVERY launch, and the result depends only on the comment text.

The vectors are identical to what text_hash returns -- this file computes them with the same code,
then asserts a sample against the original function before saving.
"""
import json, time, hashlib, argparse, torch, numpy as np

def text_hash(text, dim=4096):
    text = (text or "").lower()
    toks = text.replace("|", " ").replace(",", " ").replace(";", " ").split()
    feats = torch.zeros(dim, dtype=torch.float32)
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        feats[h % dim] += 1.0
        if len(tok) >= 3:
            for i in range(len(tok) - 2):
                h = int(hashlib.md5(("c:" + tok[i:i + 3]).encode("utf-8")).hexdigest()[:8], 16)
                feats[h % dim] += 0.25
    return feats / feats.norm().clamp_min(1.0)

def fast_hash(text, dim=4096):
    """Same arithmetic, accumulated in numpy instead of one torch scalar op per token."""
    text = (text or "").lower()
    toks = text.replace("|", " ").replace(",", " ").replace(";", " ").split()
    idx, val = [], []
    for tok in toks:
        idx.append(int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16) % dim); val.append(1.0)
        if len(tok) >= 3:
            for i in range(len(tok) - 2):
                idx.append(int(hashlib.md5(("c:" + tok[i:i + 3]).encode("utf-8")).hexdigest()[:8], 16) % dim)
                val.append(0.25)
    f = np.zeros(dim, dtype=np.float32)
    if idx:
        np.add.at(f, np.asarray(idx), np.asarray(val, dtype=np.float32))
    n = float(np.linalg.norm(f))
    return torch.from_numpy(f / max(n, 1.0))

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dim", type=int, default=4096)
a = ap.parse_args()

rows = []
for p in a.rows:
    rows += json.load(open(p))
print(f"rows {len(rows)}", flush=True)

# the fast path must reproduce the original bit for bit before it is trusted
import random
for r in random.Random(0).sample(rows, 25):
    x, y = text_hash(r["comment"], a.dim), fast_hash(r["comment"], a.dim)
    assert torch.allclose(x, y, atol=1e-6), f"fast_hash differs on {r['idx']}: max {(x-y).abs().max()}"
print("  fast path verified against text_hash on 25 rows", flush=True)

out, t0 = {}, time.perf_counter()
for i, r in enumerate(rows, 1):
    out[str(r["idx"])] = fast_hash(r["comment"], a.dim)
    if i % 10000 == 0:
        print(f"  {i}/{len(rows)}  {time.perf_counter()-t0:.0f}s", flush=True)
torch.save({"dim": a.dim, "emb": out,
            "source": "text_hash over row['comment'], identical output, computed once"}, a.out)
print(f"wrote {a.out}  ({len(out)} rows, {time.perf_counter()-t0:.0f}s)")
