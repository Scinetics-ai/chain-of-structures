#!/usr/bin/env python3
"""Per-stage hidden-state means for the NEW text, rebuilt exactly as the old file was made.

Why this is not optional: QWEN_HIDDEN_MU is subtracted from the Qwen states before they reach the
conditioner -- the comment in the trainer calls it "so the 0.3% signal becomes 30%".  The label list
appended to step 4 and step 5 is about a third of those spans' tokens, so their true mean moves, and
the old file would mis-centre exactly the two stages this round adds.

Reproduces the recorded provenance of qwen35_stage_means.pt: "Qwen3.5-9B trunk, stage1_q35 adapters,
per-stage span mean", n_docs 4000, whole span (no token cap).  The LoRA target modules are derived
from the checkpoint's own key names rather than guessed.
"""
import os, re, json, time, argparse, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--qwen", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()

D = torch.load(a.docs, map_location="cpu")
IDS, SPANS = D["ids"], D["spans"]
train = [i for i in json.load(open(a.split))["train"] if i in IDS][: a.n]
print(f"docs {len(IDS)}  train sample {len(train)}", flush=True)

ck = torch.load(a.lora, map_location="cpu")
sd = ck["qwen_lora"]; r = int(ck.get("lora_r", 32))
tm = sorted({m.group(1) for k in sd for m in [re.search(r"layers\.\d+\.(?:self_attn|mlp)\.([a-z_]+)\.lora_", k)] if m})
print(f"lora r={r} target_modules={tm}  keys={len(sd)}", flush=True)
assert tm, "could not derive target modules from the checkpoint keys"

q = AutoModelForCausalLM.from_pretrained(a.qwen, torch_dtype=torch.bfloat16, device_map=None).cuda()
q = get_peft_model(q, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                                 task_type="CAUSAL_LM", target_modules=tm))
miss, unexp = q.load_state_dict(sd, strict=False)
# every LoRA tensor in the checkpoint must land: a silently unmatched adapter means we are centring
# on the base model, not on the writer the old file was built from
assert not unexp, f"{len(unexp)} checkpoint keys did not match the model: {unexp[:3]}"
nl = sum(1 for k in q.state_dict() if "lora" in k)
print(f"lora tensors in model {nl}, checkpoint supplied {len(sd)}", flush=True)
assert nl == len(sd), "adapter count mismatch"
q.eval()

acc = {k: torch.zeros(q.config.hidden_size, dtype=torch.float64) for k in range(1, 6)}
cnt = {k: 0 for k in range(1, 6)}
t0 = time.perf_counter()
with torch.no_grad():
    for s in range(0, len(train), a.batch):
        ch = train[s:s + a.batch]
        ds = [IDS[i] for i in ch]
        L = max(len(x) for x in ds)
        x = torch.zeros(len(ch), L, dtype=torch.long)
        m = torch.zeros(len(ch), L, dtype=torch.long)
        for b, d in enumerate(ds):
            x[b, :len(d)] = d; m[b, :len(d)] = 1
        h = q(input_ids=x.cuda(), attention_mask=m.cuda(), output_hidden_states=True).hidden_states[-1]
        for b, i in enumerate(ch):
            for k in range(1, 6):
                p, e = SPANS[i][k - 1]
                e = min(e, h.shape[1])
                if e > p:
                    acc[k] += h[b, p:e].double().mean(0).cpu()
                    cnt[k] += e - p                   # token count, as the old file records
        if s % (a.batch * 100) == 0:
            print(f"  {s + len(ch)}/{len(train)}  {time.perf_counter()-t0:.0f}s", flush=True)

# the old file divides by the number of DOCUMENTS contributing, and stores token counts separately
ndoc = {k: 0 for k in range(1, 6)}
for i in train:
    for k in range(1, 6):
        p, e = SPANS[i][k - 1]
        if e > p: ndoc[k] += 1
mu = {k: (acc[k] / max(ndoc[k], 1)).float() for k in range(1, 6)}
print("norms:", {k: round(float(v.norm()), 2) for k, v in mu.items()}, flush=True)
torch.save({"stage_means": mu, "n_docs": len(train), "counts": cnt,
            "hidden": int(q.config.hidden_size), "max_tokens": None,
            "source": "Qwen3.5-9B trunk, stage1_q35 adapters, per-stage span mean (v2 text)"}, a.out)
print("wrote", a.out)
