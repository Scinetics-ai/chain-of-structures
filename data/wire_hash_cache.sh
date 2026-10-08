#!/bin/bash
# Wait for the precomputed text_hash table, put it under the key the trainer already reads, assert
# it covers every row, and add the flag to the launch scripts. No trainer code changes: --text_embed_pt
# and attach_text_embeddings already set row["_text_emb"], and make_pack skips the hash when it is there.
set -u
D=/data/caiqingran/enzymecot; S=$D/cot_split_v2
for i in $(seq 1 60); do [ -f "$S/text_hash_cache.pt" ] && break; sleep 10; done
[ -f "$S/text_hash_cache.pt" ] || { echo "ABORT: cache never appeared"; exit 1; }
sleep 3
/data/caiqingran/env/py/bin/python - <<'PY' || exit 1
import json, torch
S="/data/caiqingran/enzymecot/cot_split_v2"
d=torch.load(f"{S}/text_hash_cache.pt", map_location="cpu")
emb=d["emb"]
ids=set()
for f in ("train_rows_cotfull.json","val_rows_cotfull.json","test_rows_cotfull.json"):
    ids |= {str(r["idx"]) for r in json.load(open(f"{S}/{f}"))}
miss = ids - set(emb)
print(f"  rows {len(ids)}   cached {len(emb)}   missing {len(miss)}")
assert not miss, f"{len(miss)} rows have no cached vector, e.g. {sorted(miss)[:3]}; make_pack only checks row 0 and would KeyError mid-stack"
v=next(iter(emb.values()))
print(f"  vector {tuple(v.shape)} {v.dtype}  |v|={float(v.norm()):.4f}")
assert v.shape[-1]==4096 and abs(float(v.norm())-1.0)<1e-3
torch.save({"embeddings": emb, "source": d.get("source")}, f"{S}/text_embed_hash.pt")
print(f"  wrote {S}/text_embed_hash.pt")
PY
for f in $D/scripts/run_stage.sh $D/scripts/run_stage_2node.sh; do
  grep -q "text_embed_pt" "$f" && { echo "  already wired: $f"; continue; }
  sed -i 's#  --qwen_docs #  --text_embed_pt $S/text_embed_hash.pt \\\n  --qwen_docs #' "$f"
  grep -n "text_embed_pt" "$f" | head -1
done
echo "=== wired $(date +%H:%M:%S) ==="
