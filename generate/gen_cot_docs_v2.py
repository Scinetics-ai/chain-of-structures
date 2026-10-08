#!/usr/bin/env python3
"""Have Qwen write stages 2-5 from stage 1, so inference stops being handed the answer.

Every number measured so far read the REFERENCE stage-2/3 text at generation time, and that text
states the annotation verbatim: 16041/16041 pairs appear as one-letter code plus one-based index, and
a regex with no model scores 1.0000 on it. Stage 1 is clean - 0/16041, and only 1 of 1347 documents
contains any residue-index token at all - so it is the one legitimate input.

The loop the design calls for feeds the generated structure back into the next stage's text. Qwen was
never trained for that: `qwen_text_loss` is plain next-token cross-entropy over stages 2-5, with no
structure input anywhere. So writing all four stages autoregressively from stage 1 is not a shortcut
here, it is exactly what this model is able to do, and the missing feedback is a limitation of the
training rather than of this script.

The output is a drop-in replacement for qwen25_1p5b_docs.pt - same ids/spans/hard layout - so the
whole downstream pipeline runs unchanged by pointing --qwen_docs at it. Residues are also extracted
into site_pairs format with the regex that scores 1.0 on reference text, which makes the extractor a
controlled quantity: whatever it recovers from generated text is what the model actually wrote.
"""
import argparse, json, os, re, sys, time

import torch
try:
    import fla.modules as _fla  # noqa: F401
except Exception:
    _fla = None

S = os.environ.get("SPLIT_DIR", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2")
import glob as _glob
QP = os.environ.get("QWEN_PATH") or _glob.glob(
    "/data/workspace/caiqingran/home_migration/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/*")[0]
TOK = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="/data/workspace/caiqingran/protein_design/DISCO_repo/runs/ground_ab/ground_A_adaptive/best.pt")
    ap.add_argument("--rows", default=f"{S}/len320_test_rows_cotfull.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=1400)
    # the old value was 512, which silently dropped every label past residue 512 while the arm
    # it measured trained with --max_len 9999 and a median length of 371 (p90 565, max 989)
    ap.add_argument("--max_pos", type=int, default=9999)
    ap.add_argument("--out", default="/tmp/gen_docs")
    args = ap.parse_args()
    dev = "cuda"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(QP)
    q = AutoModelForCausalLM.from_pretrained(QP, dtype=torch.bfloat16, trust_remote_code=True)
    q = get_peft_model(q, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                     task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"]))
    lw = torch.load(args.lora, map_location="cpu").get("qwen_lora")
    assert lw, "no qwen_lora"
    state = q.state_dict()
    bad = [k for k, v in lw.items() if k not in state or state[k].shape != v.shape]
    assert not bad, f"LoRA keys/shapes mismatch: {bad[:10]}"
    loaded = q.load_state_dict(lw, strict=False)
    assert not loaded.unexpected_keys, loaded.unexpected_keys
    assert all(torch.equal(q.state_dict()[k].cpu(), v.to(q.state_dict()[k].dtype)) for k, v in lw.items()), "LoRA value mismatch"
    q = q.to(dev).eval()
    print(f"  restored {len(lw)} adapter tensors", flush=True)

    docs = torch.load(os.environ.get("DOCS_PT", f"{S}/qwen35_9b_docs.pt"), map_location="cpu")
    IDS, SPANS = docs["ids"], docs["spans"]
    rows = [r["idx"] for r in json.load(open(args.rows)) if r["idx"] in IDS]
    rows = rows[args.shard::args.nshard]
    print(f"  shard {args.shard}/{args.nshard}: {len(rows)} records", flush=True)

    # the marker each stage begins with, as token ids, so spans can be recovered without re-tokenising
    marks = {k: tok.encode(f"# Step {k}", add_special_tokens=False) for k in (2, 3, 4, 5)}
    print(f"  stage markers: " + " ".join(f"{k}:{len(v)}tok" for k, v in marks.items()), flush=True)

    def find_sub(seq, sub, start=0):
        n, m = len(seq), len(sub)
        for i in range(start, n - m + 1):
            if seq[i:i + m] == sub:
                return i
        return -1

    out_ids, out_spans, out_hard, out_pairs = {}, {}, {}, {}
    t0 = time.perf_counter()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for s0 in range(0, len(rows), args.batch):
        chunk = rows[s0:s0 + args.batch]
        # the prefix is stage 1 only - the request, and nothing the model should have produced
        prefixes = [IDS[i][:SPANS[i][0][1]].tolist() for i in chunk]
        L = max(len(p) for p in prefixes)
        x = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        m = torch.zeros((len(chunk), L), dtype=torch.long)
        for b, p in enumerate(prefixes):
            x[b, L - len(p):] = torch.tensor(p, dtype=torch.long)
            m[b, L - len(p):] = 1
        with torch.no_grad():
            g = q.generate(input_ids=x.to(dev), attention_mask=m.to(dev),
                           max_new_tokens=args.max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        for b, idx in enumerate(chunk):
            pref = prefixes[b]
            full = pref + g[b, L:].tolist()
            # trim trailing pads
            while full and full[-1] == tok.pad_token_id:
                full.pop()
            # spans: stage 1 is the prefix; each later stage starts at its marker
            starts = [0, len(pref)]
            cur = len(pref)
            ok = True
            for k in (3, 4, 5):
                j = find_sub(full, marks[k], cur + 1)
                if j < 0:
                    ok = False
                    break
                starts.append(j); cur = j
            if not ok:
                # the model stopped early; give the remaining stages empty spans at the end
                while len(starts) < 5:
                    starts.append(len(full))
            sp = [(starts[i], starts[i + 1] if i + 1 < 5 else len(full)) for i in range(5)]
            out_ids[idx] = torch.tensor(full, dtype=torch.long)
            out_spans[idx] = sp
            out_hard[idx] = torch.zeros(len(full), dtype=torch.bool)
            # residues, read out of what the model wrote, with the extractor that is exact on reference
            d = {}
            for k in (2, 3, 4, 5):
                a, e = sp[k - 1]
                if e <= a:
                    continue
                txt = tok.decode(full[a:e], skip_special_tokens=True)
                seen, lst = set(), []
                for mm in TOK.finditer(txt):
                    aa, p1 = mm.group(1), int(mm.group(2))
                    if 0 < p1 <= args.max_pos and (p1 - 1) not in seen:
                        seen.add(p1 - 1); lst.append((aa, p1 - 1))   # text is one-based
                if lst:
                    d[k] = sorted(lst, key=lambda t: t[1])
            if d:
                out_pairs[idx] = d
        if s0 % (args.batch * 4) == 0:
            done = s0 + len(chunk)
            el = time.perf_counter() - t0
            print(f"  {done}/{len(rows)}  {el:.0f}s  eta {el/max(done,1)*(len(rows)-done):.0f}s",
                  flush=True)

    torch.save({"model": docs.get("model"), "max_len": docs.get("max_len"),
                "ids": out_ids, "spans": out_spans, "hard": out_hard},
               f"{args.out}_docs_{args.shard}.pt")
    torch.save({"pairs": out_pairs, "aa_alphabet": "ACDEFGHIKLMNPQRSTVWY",
                "source": "parsed from Qwen-generated stage 2/3/4/5 text"},
               f"{args.out}_pairs_{args.shard}.pt")
    print(f"  wrote {len(out_ids)} documents", flush=True)
    for k in (2, 3, 4, 5):
        got = sum(1 for i in out_pairs if k in out_pairs[i])
        cnt = [len(out_pairs[i][k]) for i in out_pairs if k in out_pairs[i]]
        med = sorted(cnt)[len(cnt) // 2] if cnt else 0
        print(f"    stage {k}: {got}/{len(out_ids)} documents name residues "
              f"({got/max(len(out_ids),1):.3f}), median {med} labels", flush=True)


if __name__ == "__main__":
    main()
