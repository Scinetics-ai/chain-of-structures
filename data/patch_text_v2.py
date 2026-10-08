#!/usr/bin/env python3
"""Give step 4 and step 5 a residue-naming sentence, so the discrete channel exists at all 4 stages.

Nothing existing is modified.  The patched text goes to step4.v2.md / step5.v2.md and the patched
claims to claims.v2.json, beside the originals, so the 2-stage arm stays bit-identical and both arms
are reproducible from the same tree.

The sentence is rendered deterministically, not written by an LLM: the writer contract in
DATA_PIPELINE.md is "an LLM is a surface realizer, not a source of facts", and there is no new fact
here that needs realizing -- only a list.  The phrasing mirrors the step-3 sentence that already
carries labels ("beyond the core, 12 further residues lie within 5 A of the substrate: ...") so the
two stages read in one voice.

WHY THE TEXT NAMES ONLY THE NEW LAYER: site_ground_ids reads stage k's own pairs, so a naive design
would restate every earlier label in step 4 and step 5.  That costs ~4.7 tokens per label and pushes
the document past qwen_batch's 2048 cap, whose truncation keeps the FRONT -- it would silently delete
step 5's labels.  Accumulation is done in the model instead (--ground_cumulative), which is also the
more faithful path: at inference the earlier labels come from what the writer already wrote, with the
reliability already fitted for those stages.

Two guards, both for silent failures that have bitten before:
  * a non-standard residue letter (25 of them across the corpus) is not in the extractor's alphabet:
    it would be written into the text and then vanish from the parse. Filtered here.
  * after writing, the extractor build_pairs.py uses is re-run on the new text and the parsed pair
    set must equal the intended set EXACTLY. A stray "H2O" or "C4" in the prose would otherwise add
    a false label at position 1 or 3.
"""
import os, json, re, sys
V2 = os.environ.get("RUNS_V2", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs_v2")
S  = os.environ.get("SPLIT_DIR", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2")
SH = json.load(open(os.environ.get("SHELLS", f"{S}/shells_v1.json")))
AA1 = "ACDEFGHIKLMNPQRSTVWY"
RES = re.compile(rf"\b([{AA1}])(\d{{1,4}})\b")
DRY = "--write" not in sys.argv

LAYERS = ((4, "shell2", "O4S2",
           "Beyond that first shell, {n} further residues lie between 5 and 8 angstroms of the "
           "substrate and make up the second shell that the backbone has to place: {lst}."),
          (5, "anchor", "O5AN",
           "Away from the site, the {n} most buried positions of the chain are {lst}, and their "
           "side chains pack the interior that holds the core segments at the separation the site "
           "requires."))


def obs_of(t):
    i = t.find("## OBSERVATION")
    return t[i:] if i >= 0 else t


def parse(t):
    seen, out = set(), []
    for m in RES.finditer(obs_of(t)):
        aa, p = m.group(1), int(m.group(2))
        if p > 0 and (aa, p) not in seen:
            seen.add((aa, p))
            out.append((aa, p))
    return out


def labels_of(rec, field):
    return [(a, int(q)) for a, q in rec[field] if a in AA1]


stats = {"ok": 0, "no_step": 0, "empty_layer": 0, "already_had_pairs": 0, "mismatch": 0, "err": 0}
ex = []
for idx, rec in SH.items():
    try:
        wrote_any = False
        for stage, field, cid, tpl in LAYERS:
            p = f"{V2}/cot_{idx}/step{stage}.md"
            if not os.path.exists(p):
                stats["no_step"] += 1
                continue
            old = open(p).read()
            had = parse(old)
            if had:                      # step4/step5 name nothing today; if one does, leave it alone
                stats["already_had_pairs"] += 1
                if len(ex) < 4:
                    ex.append((idx, stage, "already had", had[:4]))
                continue
            lbls = labels_of(rec, field)
            if not lbls:
                stats["empty_layer"] += 1
                continue
            new = old.rstrip() + " " + tpl.format(
                n=len(lbls), lst=", ".join(f"{a}{q}" for a, q in lbls)) + "\n"
            got = parse(new)
            if got != lbls:              # THE guard
                stats["mismatch"] += 1
                if len(ex) < 8:
                    ex.append((idx, stage, "parse != intended",
                               [x for x in got if x not in lbls][:5]))
                continue
            if not DRY:
                open(f"{V2}/cot_{idx}/step{stage}.v2.md", "w").write(new)
            stats["ok"] += 1
            wrote_any = True
        if wrote_any and not DRY:
            cj = json.load(open(f"{V2}/cot_{idx}/claims.json"))
            for stage, field, cid, tpl in LAYERS:
                lbls = labels_of(rec, field)
                if not lbls or not os.path.exists(f"{V2}/cot_{idx}/step{stage}.v2.md"):
                    continue
                cj["claims"].append({
                    "stage": f"step{stage}", "part": "OBS", "id": cid,
                    "fact": tpl.format(n=len(lbls),
                                       lst=", ".join(f"{a}{q}" for a, q in lbls)).rstrip("."),
                    "entities": [f"{a}{q}" for a, q in lbls], "numbers": [len(lbls)],
                    "epistemic": "predicted_structure", "source": "Boltz-1x model_0"})
            json.dump(cj, open(f"{V2}/cot_{idx}/claims.v2.json", "w"))
    except Exception as e:
        stats["err"] += 1
        if len(ex) < 8:
            ex.append((idx, -1, type(e).__name__, str(e)[:90]))

print(("DRY RUN (pass --write to write)" if DRY else "WROTE") + f"   {stats}")
for x in ex:
    print("   ", x)
assert stats["mismatch"] == 0, "rendered labels and parsed labels disagree -- do not build on this"
