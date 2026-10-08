#!/usr/bin/env python3
"""The two NEW position layers that give stage 4 and stage 5 a discrete channel.

Today step4.md and step5.md name no residue at all, so build_pairs.py extracts exactly 0 pairs for
those stages and GroundingInjector is a no-op there: the discrete channel exists only at stages 2
and 3.  This adds one layer per stage, computed from the SAME cif and the SAME heavy-atom minimum
distance claims_v2.py already uses, so no second epistemic frame enters the record.

  stage 4  shell2  the <=24 residues nearest the substrate in the (5, 8] A band, outside core u
                   pocket.  Step 4's job is the backbone, and this is the layer the backbone has to
                   place in order to hold the first shell in register.
  stage 5  anchor  the 16 most buried positions outside every earlier layer, burial = number of
                   residue centroids within 10 A.  Step 5's job is every side chain outside the
                   site, and these are the ones whose identity the fold actually determines.

core and pocket are READ BACK from claims.json rather than recomputed, so the new layers cannot
overlap the residues the existing step2/step3 text already names.

1-BASED: claims and the .md text number residues from 1 (gemmi seqid.num).  build_pairs.py subtracts
1 to get the tensor index, so a record whose seqids are not exactly 1..L would put every label at
the wrong index.  That is asserted per record and such records are dropped, not silently emitted.
"""
import os, json, sys
import numpy as np, gemmi

V2  = os.environ.get("RUNS_V2", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs_v2")
S   = os.environ.get("SPLIT_DIR", "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2")
DS  = os.environ.get("DS_DIR", "/data/share/enzyme_design/deepseek_polished_per_protein")
CIF = os.environ.get("CIF_ROOT", "/data/share/enzyme_design/enzyme_data/boltz-1x/output") + "/{i}/boltz_results_{i}/predictions/{i}/{i}_model_0.cif"
OUT = os.environ.get("SHELLS_OUT", f"{S}/shells_v1.json")
AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
          "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
AA1 = "ACDEFGHIKLMNPQRSTVWY"
N_SHELL2, N_ANCHOR = 24, 16
BAND_LO, BAND_HI, BURIAL_CUT = 5.0, 8.0, 10.0

split = json.load(open(f"{S}/split_seqcluster.json"))
allids = {i for v in split.values() for i in v}
# DS_DIR="" means every record; otherwise restrict to the deepseek-polished intersection.
# The join key is asserted either way: an empty intersection from a format mismatch already cost a
# round once (our idx IS the full "10016_Q3ZFI4" form, the deepseek dirs are not a bare prefix).
if DS and os.path.isdir(DS):
    dsd = set(os.listdir(DS))
    inter = sorted(allids & dsd)
    print(f"rows {len(allids)}  deepseek dirs {len(dsd)}  intersection {len(inter)}", flush=True)
    assert len(inter) > 20000, f"join collapsed: {len(inter)}"
else:
    inter = sorted(allids)
    print(f"rows {len(allids)}  no DS_DIR: using every record", flush=True)
print(f"records {len(inter)}", flush=True)

def ents_of(cj, stage):
    out = set()
    for c in cj["claims"]:
        if c["stage"] == stage and c.get("part") == "OBS":
            for e in c.get("entities", []):
                if len(e) > 1 and e[0] in AA1 and e[1:].isdigit():
                    out.add(int(e[1:]))
    return out

bad = {"no_cif": 0, "no_claims": 0, "seqid_gap": 0, "no_ligand": 0, "err": 0}
res = {}
for n, idx in enumerate(inter):
    try:
        p = CIF.format(i=idx)
        if not os.path.exists(p): bad["no_cif"] += 1; continue
        cj = json.load(open(f"{V2}/cot_{idx}/claims.json"))
        stx = gemmi.read_structure(p); stx.setup_entities()
        prot, hetc = {}, {}
        for ch in stx[0]:
            for r in ch:
                t = gemmi.find_tabulated_residue(r.name)
                xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
                if len(xyz) == 0: continue
                if t and t.is_amino_acid(): prot[r.seqid.num] = (r.name, xyz)
                else: hetc.setdefault(ch.name, []).append(xyz)
        keys = sorted(prot); L = len(keys)
        if keys != list(range(1, L + 1)):        # 1-BASED assertion, per record
            bad["seqid_gap"] += 1; continue
        hetc = {c: np.vstack(v) for c, v in hetc.items()}
        subch = [g["chain"] for g in cj["ligands"] if g.get("kind") == "substrate"]
        ref = [hetc[c] for c in subch if c in hetc] or list(hetc.values())
        if not ref: bad["no_ligand"] += 1; continue
        ref = np.vstack(ref)
        hc = [prot[k][1] for k in keys]
        dmin = np.array([np.sqrt(((h[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min() for h in hc])
        inner = ents_of(cj, "step2") | ents_of(cj, "step3")
        band = sorted(((dmin[i], keys[i]) for i in range(L)
                       if BAND_LO < dmin[i] <= BAND_HI and keys[i] not in inner))
        shell2 = sorted(k for _, k in band[:N_SHELL2])
        cen = np.array([h.mean(0) for h in hc])
        D = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
        nb = (D <= BURIAL_CUT).sum(1) - 1
        taken = inner | set(shell2)
        cand = sorted(((-int(nb[i]), keys[i]) for i in range(L) if keys[i] not in taken))
        anchor = sorted(k for _, k in cand[:N_ANCHOR])
        assert not (set(shell2) & inner) and not (set(anchor) & taken), idx
        res[idx] = {"L": L,
                    "shell2": [[AA3to1.get(prot[k][0], "X"), k] for k in shell2],
                    "anchor": [[AA3to1.get(prot[k][0], "X"), k] for k in anchor],
                    "n_core": len(ents_of(cj, "step2")), "n_pocket": len(ents_of(cj, "step3"))}
    except Exception as e:
        bad["err"] += 1
        if bad["err"] <= 3: print("  ERR", idx, type(e).__name__, e, flush=True)
    if (n + 1) % 2000 == 0: print(f"  {n+1}/{len(inter)}  ok={len(res)}", flush=True)

print(f"\nusable {len(res)}/{len(inter)}   dropped {bad}")
import statistics as sta
for f in ("shell2", "anchor"):
    v = sorted(len(r[f]) for r in res.values())
    print(f"  {f:8s} med {v[len(v)//2]:3d}  p10 {v[len(v)//10]:3d}  min {v[0]:3d}  zero {sum(1 for x in v if x==0)}")
# any label with an unknown residue letter would be dropped by build_pairs' alphabet, silently
nx = sum(1 for r in res.values() for f in ("shell2", "anchor") for a, _ in r[f] if a not in AA1)
print(f"  labels with a non-standard residue letter: {nx}  (these would vanish from the parse)")
for k, v in split.items():
    print(f"  split {k}: {len(set(v) & set(res))}")
json.dump(res, open(OUT, "w"))
print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
