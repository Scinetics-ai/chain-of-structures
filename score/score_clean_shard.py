#!/usr/bin/env python3
"""Run existing CLEAN for one arm/shard, preserving the full expected denominator.

--output is a directory. Every invocation uses unique dataset AND sequence identifiers:
CLEAN's shared ESM cache is keyed by FASTA header, not by dataset filename.
No generation, reference sequences, or site annotations are sent to CLEAN.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid


AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def dump_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_fasta(path):
    """Return unambiguous IDs only; never silently overwrite duplicate records."""
    seqs, duplicates = {}, set()
    key, parts = None, []

    def commit():
        if key is None:
            return
        if key in seqs or key in duplicates:
            duplicates.add(key)
            seqs.pop(key, None)
        else:
            seqs[key] = "".join(parts).upper()

    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:].strip()
                if not header:
                    raise ValueError("Empty FASTA header")
                key = header.split()[0].split("|")[0]
                parts = []
            elif key is None:
                raise ValueError("Sequence precedes the first FASTA header")
            else:
                parts.append("".join(line.split()))
    commit()
    return seqs, sorted(duplicates)


def parse_ec_labels(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        source = [part for item in value for part in parse_ec_labels(item)]
    else:
        source = [re.sub(r"^EC\s*:?\s*", "", v.strip(), flags=re.I)
                  for v in re.split(r"[;,]", str(value)) if v.strip()]
    return list(dict.fromkeys(v for v in source if v))


def ec_prefix(ec, level):
    parts = ec.split(".")
    if len(parts) < level or not all(p.isdigit() for p in parts[:level]):
        return None
    return ".".join(parts[:level])


def read_predictions(path, reverse):
    preds, errors, unexpected, duplicates = {}, [], [], set()
    with Path(path).open(newline="") as f:
        for line_no, fields in enumerate(csv.reader(f), 1):
            if not fields:
                continue
            alias = fields[0].strip()
            if alias not in reverse:
                unexpected.append(alias)
                continue
            idx = reverse[alias]
            if idx in preds or idx in duplicates:
                duplicates.add(idx)
                preds.pop(idx, None)
                continue
            parsed = []
            for field in fields[1:]:
                try:
                    label, score = field.strip().rsplit("/", 1)
                    score = float(score)
                    if not math.isfinite(score):
                        raise ValueError("non-finite score")
                    label = re.sub(r"^EC:", "", label)
                    if ec_prefix(label, 1) is None:
                        raise ValueError("invalid EC")
                    parsed.append({"ec": label, "score": score})
                except (ValueError, TypeError):
                    errors.append({"line": line_no, "idx": idx, "field": field})
            preds[idx] = parsed
    return preds, errors, unexpected, sorted(duplicates)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--rows", required=True, help="Expected shard rows JSON, not full test rows")
    ap.add_argument("--output", required=True, help="Directory for summary.json and per-sample files")
    ap.add_argument("--name-prefix", required=True, help="Arm/shard label; invocation UUID added automatically")
    ap.add_argument("--clean-root", default="/data/caiqingran/CLEAN/app")
    ap.add_argument("--python", default="/data/caiqingran/env/py/bin/python")
    ap.add_argument("--timeout", type=float, default=21600)
    args = ap.parse_args(argv)
    output, app = Path(args.output).resolve(), Path(args.clean_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".score_clean.lock"
    # Refuse concurrent writers to one output directory. Arm/shard directories must differ.
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    try:
        return run(args, output, app)
    finally:
        lock.unlink(missing_ok=True)


def run(args, output, app):
    started = time.time()
    rows = json.loads(Path(args.rows).read_text())
    if not isinstance(rows, list) or not all(isinstance(r, dict) and "idx" in r for r in rows):
        raise ValueError("--rows must contain a list of records with idx")
    expected = [str(r["idx"]) for r in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate expected row IDs: refusing ambiguous evaluation")
    truth = {str(r["idx"]): r for r in rows}
    fasta_error = None
    try:
        seqs, duplicate_fasta = read_fasta(args.fasta)
    except (OSError, ValueError) as exc:
        seqs, duplicate_fasta, fasta_error = {}, [], str(exc)
    extra_fasta = sorted((set(seqs) | set(duplicate_fasta)) - set(expected))
    missing_fasta = [i for i in expected if i not in seqs and i not in duplicate_fasta]
    invalid = {i: "empty_sequence" if not seqs[i] else "invalid_sequence_characters"
               for i in expected if i in seqs and (not seqs[i] or set(seqs[i]) - AA)}
    valid = [i for i in expected if i in seqs and i not in invalid]
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", args.name_prefix).strip("_")[:72] or "clean"
    name = safe_prefix + "_" + uuid.uuid4().hex[:16]
    aliases = {i: f"{name}__{n:06d}" for n, i in enumerate(valid)}
    reverse = {alias: i for i, alias in aliases.items()}
    dump_json(output / "id_mapping.json", {"dataset_name": name, "alias_to_idx": reverse})
    command = [str(Path(args.python).absolute()), str(app / "CLEAN_infer_fasta.py"), "-d", name]
    run_meta = {
        "name_prefix": args.name_prefix, "dataset_name": name, "fasta": str(Path(args.fasta).resolve()),
        "rows": str(Path(args.rows).resolve()), "fasta_sha256": sha256(args.fasta),
        "rows_sha256": sha256(args.rows), "clean_root": str(app), "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_unix": started,
        "input_expected": len(expected), "input_valid": len(valid),
        "clean_infer_sha256": sha256(app / "CLEAN_infer_fasta.py"),
    }
    dump_json(output / "run_manifest.json", run_meta)
    raw_result = app / "results" / "inputs" / (name + "_maxsep.csv")
    returncode, clean_error = None, None
    if valid:
        try:
            input_dir = app / "data" / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            raw_result.parent.mkdir(parents=True, exist_ok=True)
            with (input_dir / (name + ".fasta")).open("x") as f:
                for idx in valid:
                    f.write(f">{aliases[idx]}\n{seqs[idx]}\n")
            env = dict(os.environ)
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # Trusted, installed ESM/CLEAN checkpoint files.
            # CLEAN spawns a literal `python` command for ESM embedding extraction.
            env["PATH"] = str(Path(args.python).absolute().parent) + os.pathsep + env.get("PATH", "")
            env["PYTHONPATH"] = str(app / "src") + os.pathsep + env.get("PYTHONPATH", "")
            with (output / "clean.log").open("w") as log:
                proc = subprocess.run(command, cwd=app, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            returncode = proc.returncode
            if returncode:
                clean_error = f"CLEAN exited with code {returncode}"
            if not raw_result.is_file():
                clean_error = (clean_error + "; " if clean_error else "") + "CLEAN result file absent"
        except (OSError, subprocess.TimeoutExpired) as exc:
            clean_error = str(exc)
    else:
        clean_error = "No valid generated sequences to score"
    predictions, parse_errors, unexpected_predictions, duplicate_predictions = {}, [], [], []
    if raw_result.is_file():
        shutil.copy2(raw_result, output / "clean_maxsep.csv")
        predictions, parse_errors, unexpected_predictions, duplicate_predictions = read_predictions(raw_result, reverse)
    samples = []
    for idx in expected:
        row = truth[idx]
        seq = seqs.get(idx)
        ecs = parse_ec_labels(row.get("ec_list", ""))
        pred = predictions.get(idx, [])
        top1 = pred[0] if pred else None
        if idx in duplicate_fasta:
            status = "duplicate_fasta_id"
        elif idx in missing_fasta:
            status = "missing_fasta"
        elif idx in invalid:
            status = invalid[idx]
        elif idx in duplicate_predictions:
            status = "duplicate_clean_prediction"
        elif idx not in predictions:
            status = "clean_failed" if clean_error else "missing_clean_prediction"
        elif not pred:
            status = "empty_clean_prediction"
        else:
            status = "scored"
        target_length = int(row.get("length") or len(row.get("sequence") or ""))
        sample = {
            "idx": idx, "status": status, "truth_ec": ecs,
            "top1_ec": None if top1 is None else top1["ec"],
            "top1_score": None if top1 is None else top1["score"], "predictions": pred,
            "generated_length": None if seq is None else len(seq), "expected_length": target_length,
            "length_matches": seq is not None and len(seq) == target_length,
            "identical_to_reference": bool(seq and row.get("sequence") and seq == row["sequence"]),
            "sequence_sha256": None if seq is None else hashlib.sha256(seq.encode()).hexdigest(),
        }
        for k in range(1, 5):
            targets = {p for ec in ecs if (p := ec_prefix(ec, k)) is not None}
            prefix = None if top1 is None else ec_prefix(top1["ec"], k)
            sample[f"ec{k}_annotated"] = bool(targets)
            sample[f"ec{k}_top1_match"] = bool(status == "scored" and prefix is not None and prefix in targets)
        samples.append(sample)
    dump_json(output / "per_sample.json", samples)
    fields = ["idx", "status", "truth_ec", "top1_ec", "top1_score", "generated_length", "expected_length",
              "length_matches", "identical_to_reference", "sequence_sha256"]
    fields += [f"ec{k}_{suffix}" for k in range(1, 5) for suffix in ("annotated", "top1_match")]
    with (output / "per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            writer.writerow({**sample, "truth_ec": ";".join(sample["truth_ec"])})
    metrics = {}
    for k in range(1, 5):
        hits = sum(s[f"ec{k}_top1_match"] for s in samples)
        annotated = sum(s[f"ec{k}_annotated"] for s in samples)
        metrics[f"EC{k}"] = {"top1_hits": hits, "expected_denominator": len(expected),
                            "top1_rate_all_expected": hits / len(expected) if expected else None,
                            "annotated_denominator": annotated,
                            "top1_rate_annotated_expected": hits / annotated if annotated else None}
    complete = bool(expected) and all(s["status"] == "scored" for s in samples)
    complete = complete and not (extra_fasta or unexpected_predictions or parse_errors or clean_error)
    summary = {
        **run_meta, "completed_unix": time.time(), "elapsed_seconds": time.time() - started,
        "coverage_complete": complete, "clean_returncode": returncode, "clean_error": clean_error,
        "fasta_error": fasta_error, "metrics": metrics,
        "status_counts": dict(collections.Counter(s["status"] for s in samples)),
        "missing_fasta_ids": missing_fasta, "duplicate_fasta_ids": duplicate_fasta,
        "extra_fasta_ids": extra_fasta, "unexpected_prediction_ids": unexpected_predictions,
        "duplicate_prediction_ids": duplicate_predictions, "prediction_parse_errors": parse_errors,
        "length_mismatch_count": sum(s["generated_length"] is not None and not s["length_matches"] for s in samples),
        "identical_to_reference_count": sum(s["identical_to_reference"] for s in samples),
        "denominator_policy": "Every expected row retained; missing/failed predictions are non-hits. No site-pair filtering.",
        "metric_scope": "CLEAN target EC top-1 prefix matching; not folding, enzymatic activity, or proof of leakage.",
    }
    dump_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "coverage_complete": complete,
                      "expected": len(expected), "status_counts": summary["status_counts"],
                      "metrics": metrics}, ensure_ascii=False), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
