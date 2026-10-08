#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks)
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        records[name] = "".join(chunks)
    return records


def read_ca(path: Path) -> tuple[np.ndarray, str]:
    coords: list[list[float]] = []
    seq: list[str] = []
    seen: set[tuple[str, str]] = set()
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            key = (line[21:22].strip() or "A", line[22:27])
            if key in seen:
                continue
            seen.add(key)
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return np.asarray(coords, dtype=np.float64), "".join(seq)


def kabsch_rmsd(pred: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(pred), len(ref))
    pred = pred[:n]
    ref = ref[:n]
    pred_c = pred - pred.mean(axis=0)
    ref_c = ref - ref.mean(axis=0)
    cov = pred_c.T @ ref_c
    u, _, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    aligned = pred_c @ rot.T
    return float(np.sqrt(((aligned - ref_c) ** 2).sum(axis=-1).mean()))


def tm_score(pred_ca: np.ndarray, ref_ca: np.ndarray, pred_seq: str, ref_seq: str) -> float:
    from tmtools import tm_align

    n_pred, n_ref = len(pred_ca), len(ref_ca)
    if len(pred_seq) != n_pred:
        pred_seq = (pred_seq + "A" * n_pred)[:n_pred]
    if len(ref_seq) != n_ref:
        ref_seq = (ref_seq + "A" * n_ref)[:n_ref]
    result = tm_align(pred_ca, ref_ca, pred_seq, ref_seq)
    return float(result.tm_norm_chain2)


class ESMFoldRunner:
    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", local_files_only=True)
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval().to(self.device)

        try:
            import transformers.models.esm.modeling_esmfold as modeling_esmfold
            import transformers.models.esm.openfold_utils.loss as ofloss

            original_compute_tm = ofloss.compute_tm

            def safe_compute_tm(*args, **kwargs):
                try:
                    return original_compute_tm(*args, **kwargs)
                except Exception:
                    return torch.tensor(0.0, device=args[0].device if args else self.device)

            ofloss.compute_tm = safe_compute_tm
            modeling_esmfold.compute_tm = safe_compute_tm
        except Exception:
            pass

    def fold_ca(self, sequence: str) -> tuple[np.ndarray, float]:
        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        pos = out.positions.detach().cpu()
        if pos.ndim == 5:
            pos = pos[-1, 0]
        elif pos.ndim == 4:
            pos = pos[0]
        ca = pos.numpy().astype(np.float64)[:, 1, :]
        plddt = out.plddt.detach().cpu()
        if plddt.ndim >= 2:
            plddt = plddt[0]
        if plddt.ndim == 2:
            plddt = plddt.mean(dim=-1)
        return ca, float(plddt.numpy().mean())


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {row["idx"] for row in csv.DictReader(handle) if row.get("idx")}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [r for r in rows if r.get("ok")]
    rmsd = np.asarray([float(r["sc_rmsd"]) for r in ok], dtype=float)
    tm = np.asarray([float(r["sc_tm"]) for r in ok], dtype=float)
    plddt = np.asarray([float(r["plddt"]) for r in ok], dtype=float)
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "rmsd_lt_2": int((rmsd < 2.0).sum()) if len(rmsd) else 0,
        "rmsd_lt_2_rate": float((rmsd < 2.0).mean()) if len(rmsd) else None,
        "rmsd_lt_5": int((rmsd < 5.0).sum()) if len(rmsd) else 0,
        "mean_rmsd": float(rmsd.mean()) if len(rmsd) else None,
        "median_rmsd": float(np.median(rmsd)) if len(rmsd) else None,
        "mean_sc_tm": float(tm.mean()) if len(tm) else None,
        "median_sc_tm": float(np.median(tm)) if len(tm) else None,
        "sc_tm_gt_0p8": int((tm > 0.8).sum()) if len(tm) else 0,
        "sc_tm_gt_0p5": int((tm > 0.5).sum()) if len(tm) else 0,
        "mean_plddt": float(plddt.mean()) if len(plddt) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--out_csv")
    parser.add_argument("--summary_json")
    args = parser.parse_args()

    run = Path(args.run)
    fasta = run / f"generated_{args.split}" / "aatype.fasta"
    pdb_dir = run / f"generated_{args.split}" / "pdb"
    out_csv = Path(args.out_csv) if args.out_csv else run / f"esmfold_self_consistency_{args.split}.csv"
    summary_json = Path(args.summary_json) if args.summary_json else run / f"esmfold_self_consistency_{args.split}_summary.json"

    records = list(read_fasta(fasta).items())
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        records = records[: args.limit]

    done = existing_ids(out_csv)
    runner = ESMFoldRunner()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx", "ok", "seq_len", "pdb_len", "sc_rmsd", "sc_tm", "plddt", "error",
    ]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, (idx, seq) in enumerate(records, 1):
            if idx in done:
                continue
            pdb = pdb_dir / f"{idx}.pdb"
            row: dict[str, object] = {
                "idx": idx,
                "ok": False,
                "seq_len": len(seq),
                "pdb_len": 0,
                "sc_rmsd": float("nan"),
                "sc_tm": float("nan"),
                "plddt": float("nan"),
                "error": "",
            }
            try:
                ref_ca, ref_seq = read_ca(pdb)
                row["pdb_len"] = len(ref_ca)
                if len(ref_ca) < 10:
                    raise ValueError("too few CA atoms")
                fold_ca, plddt = runner.fold_ca(seq)
                row["plddt"] = plddt
                row["sc_rmsd"] = kabsch_rmsd(fold_ca, ref_ca)
                row["sc_tm"] = tm_score(fold_ca, ref_ca, seq, ref_seq or seq)
                row["ok"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}:{str(exc)[:160]}"
            writer.writerow(row)
            handle.flush()
            print(
                f"[{i}/{len(records)}] {idx} ok={row['ok']} "
                f"rmsd={float(row['sc_rmsd']):.2f} scTM={float(row['sc_tm']):.3f} "
                f"pLDDT={float(row['plddt']):.1f} {row['error']}",
                flush=True,
            )

    with out_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ok"] = str(row.get("ok")).lower() == "true"
    summary = summarize(rows)
    summary["csv"] = str(out_csv)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
