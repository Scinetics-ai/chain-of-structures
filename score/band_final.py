"""Everything against length, on one table: writer (gate 1), sequence (repetition), structure
(ESMFold self-consistency), function (CLEAN EC3), each with its natural-sequence reference."""
import json, math, os, csv, torch
import sys
sys.path.insert(0, "/data/workspace/caiqingran/cc_scripts")
from score_lenscan import metrics, load

S = "/data/workspace/caiqingran/protein_design/DISCO_repo/runs/cot_split_v2"
B = "/data/workspace/caiqingran/protein_design/DISCO_repo/runs"
R = "/data/workspace/caiqingran/protein_design/CLEAN/app/results/inputs"
ref = torch.load(f"{S}/site_pairs.pt", map_location="cpu")["pairs"]
gen = torch.load(f"{S}/site_pairs_gen_band.pt", map_location="cpu")["pairs"]
rows_all = {r["idx"]: r for r in json.load(open(f"{S}/test_rows_cotfull.json"))}
lab = {i: {".".join(e.strip().split(".")[:3]) for e in str(r["ec_list"]).split(";")
           if len(e.strip().split(".")) >= 3} for i, r in rows_all.items()}

def ec3(tag):
    p = f"{R}/{tag}_maxsep.csv"
    if not os.path.exists(p): return None
    ok = n = 0
    for ln in open(p):
        ln = ln.strip()
        if not ln: continue
        head, *preds = ln.split(",")
        _, _, idx = head.partition("|")
        idx = idx or head
        want = lab.get(idx) or set()
        if not want: continue
        got = {".".join(x.split("/")[0].replace("EC:", "").split(".")[:3]) for x in preds if x}
        n += 1; ok += bool(want & got)
    return (100.0 * ok / n, ok, n) if n else None

def fold(tag):
    p = f"{B}/cotband_{tag}/esmfold_sc.csv"
    if not os.path.exists(p): return None
    sc, pl, rm, pt = [], [], [], []
    for r in csv.DictReader(open(p)):
        if str(r.get("ok", "")).lower() != "true": continue
        for k, acc in (("sc_tm", sc), ("plddt", pl), ("sc_rmsd", rm), ("ptm", pt)):
            v = r.get(k, "")
            if v not in ("", None):
                try: acc.append(float(v))
                except ValueError: pass
    f = lambda a: sum(a)/len(a) if a else float("nan")
    return {"scTM": f(sc), "pLDDT": f(pl), "rmsd": f(rm), "pTM": f(pt),
            "scTM>0.5": (sum(1 for x in sc if x > 0.5)/len(sc) if sc else float("nan")),
            "rmsd<2": (sum(1 for x in rm if x < 2)/len(rm) if rm else float("nan")), "n": len(sc)}

BANDS = [("b1","103-318"),("b2","321-499"),("b3","513-637"),("b4","641-985")]
print(f"{'band':5}{'长度':>10}{'g1_exact':>10}{'g1_pos':>9}{'重复覆盖':>10}{'scTM':>8}"
      f"{'scTM>0.5':>10}{'rmsd<2A':>9}{'pLDDT':>8}{'EC3生成':>9}{'EC3天然':>9}")
for tag, rng in BANDS:
    idxs = [r["idx"] for r in json.load(open(f"{S}/lenband_{tag}_rows.json"))]
    ex = pos = tot = 0
    for i in idxs:
        for st in (2, 3):
            t = (ref.get(i) or {}).get(st) or []
            gd = {p: a for a, p in ((gen.get(i) or {}).get(st) or [])}
            for a, p in t:
                tot += 1; ex += (gd.get(p) == a); pos += (p in gd)
    f = f"{B}/cotband_{tag}/shard0/generated_test/aatype.fasta"
    m = metrics(load(f)) if os.path.exists(f) else {}
    fo = fold(tag) or {}
    cg, cn = ec3(f"cotband_{tag}"), ec3(f"natband_{tag}")
    g = lambda d, k: d.get(k, float("nan"))
    print(f"{tag:5}{rng:>10}{ex/max(tot,1):10.4f}{pos/max(tot,1):9.4f}{m.get('repcov',float('nan')):10.4f}"
          f"{g(fo,'scTM'):8.3f}{g(fo,'scTM>0.5'):10.3f}{g(fo,'rmsd<2'):9.3f}{g(fo,'pLDDT'):8.3f}"
          f"{(cg[0] if cg else float('nan')):9.1f}{(cn[0] if cn else float('nan')):9.1f}")
