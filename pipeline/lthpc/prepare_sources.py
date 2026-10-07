#!/usr/bin/env python3
"""Build private natural reconstruction sources from local, trusted archives.

Only cotstep1.comment is permitted as public text. Sequence and structure bits
are private reconstruction targets, never assistant SFT answers. Pickle/PT input
must be trusted local archives; no downloaded objects are deserialized here.
Fallback PT tokenizations are audited separately, not silently mixed into the
primary corpus without establishing their codec provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re

AA = set("ACDEFGHIKLMNPQRSTVWY")
REVIEWED_PUBLIC_TEXT = {}
ADDRESS_PATTERNS = {
    "numbered_amino_acid": re.compile(r"\b(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\s*[-:]?\s*\d+\b", re.I),
    "one_letter_residue_address": re.compile(r"\b[ACDEFGHIKLMNPQRSTVWY][1-9]\d{1,3}\b"),
    "explicit_residue_position": re.compile(r"\b(?:residue|position|site)\s*(?:number|index|at|#|:)??\s*\d+\b", re.I),
    "later_cot_section": re.compile(r"\b(?:Catalytic core|Active site|Backbone|Full structure)\s*:", re.I),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate(source: dict, public: dict, bit_dim: int):
    reasons = []
    seq = source.get("sequence", "")
    codes = source.get("struct_ids")
    if codes is None:
        raw = source.get("struct_seq", "")
        try:
            codes = [int(x) for x in raw.split(",")]
        except (ValueError, AttributeError):
            codes = []
    if not isinstance(seq, str) or not seq:
        reasons.append("missing_sequence")
        seq = ""
    elif set(seq) - AA:
        reasons.append("noncanonical_sequence")
    if not isinstance(codes, list) or len(codes) != len(seq):
        reasons.append("sequence_structure_length_mismatch")
    if not codes or any(type(x) is not int or not 0 <= x < (1 << bit_dim) for x in codes):
        reasons.append("invalid_structure_code")
    if len(seq) != public.get("length"):
        reasons.append("public_length_target_mismatch")
    text = public.get("comment", "")
    if not isinstance(text, str) or not text.strip():
        reasons.append("missing_public_text")
    else:
        flags = ["public_text_flag:" + name for name, pattern in ADDRESS_PATTERNS.items() if pattern.search(text)]
        review = REVIEWED_PUBLIC_TEXT.get(str(public["idx"]), {})
        if review.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
            reasons += flags
        if len(seq) >= 20 and seq in text:
            reasons.append("public_text_contains_target_sequence")
    return seq, codes, reasons


def make_record(source, public, split, bit_dim, public_path, private_path, hashes):
    seq, codes, reasons = validate(source, public, bit_dim)
    if reasons:
        return None, reasons
    bits = [[(code >> i) & 1 for i in range(bit_dim - 1, -1, -1)] for code in codes]
    return {
        "schema": "natural_reconstruction_source_v1",
        "parent_id": str(public["idx"]), "split": split,
        "function_text": public["comment"].strip(),
        "sequence": seq, "bits": bits,
        "ec_list": public.get("ec_list", ""),
        "provenance": {
            "public_text_source": str(public_path), "public_text_sha256": hashes[str(public_path)],
            "private_target_source": str(private_path), "private_target_sha256": hashes[str(private_path)],
            "source_row_idx": str(public["idx"]),
            "target_sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
            "bit_dim": bit_dim, "bit_order": "descending_binary_LFQ_code_index",
            "contains_expert_agent_trace": False,
            "scope": "sequence_and_bits_are_private_targets_never_public_agent_input",
        },
    }, []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step1-dir", type=Path, required=True)
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bit-dim", type=int, required=True)
    p.add_argument("--audit-pt-fallback", action="store_true")
    p.add_argument("--reviewed-public-text", type=Path)
    args = p.parse_args()
    if args.reviewed_public_text is not None:
        REVIEWED_PUBLIC_TEXT.update(json.loads(args.reviewed_public_text.read_text()))
    if not 1 <= args.bit_dim <= 30:
        p.error("bit-dim must be 1..30 and must match the loaded generator codec")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "source_audit_v1", "source_hashes": {}, "splits": {},
              "public_text_policy": "cotstep1.comment only; no cotfull/comment from token archives",
              "length_policy": "no length cutoff; mismatched declared lengths quarantined without truncation",
              "sft_policy": "no fabricated actions, addresses, chemical roles or reasoning labels",
              "codec_caveat": "13-bit range and descending binary order verified; original tokenization checkpoint hash not present in source pickle",
              "provenance_scope": "trusted historical dataset, not independently verified chemical truth"}
    report["reviewed_public_text_false_positive_ids"] = sorted(REVIEWED_PUBLIC_TEXT)
    if args.reviewed_public_text is not None:
        report["reviewed_public_text_file"] = {"path": str(args.reviewed_public_text),
                                               "sha256": sha256(args.reviewed_public_text)}
    hashes, allseq, allids, missing_by_split, public_maps = report["source_hashes"], {}, {}, {}, {}
    for split, textsplit, targetsplit in (("train", "train", "train"), ("validation", "val", "valid"), ("test", "test", "test")):
        public_path = args.step1_dir / (textsplit + "_rows_cotstep1.json")
        target_path = args.target_dir / (targetsplit + ".pkl")
        for path in (public_path, target_path):
            hashes[str(path)] = sha256(path)
        public = json.loads(public_path.read_text())
        with target_path.open("rb") as fh:
            targets = pickle.load(fh)
        by_id = {str(x["idx"]): x for x in targets}
        pby_id = {str(x["idx"]): x for x in public}
        if len(by_id) != len(targets) or len(pby_id) != len(public):
            raise ValueError("Duplicate source ID; resolve before constructing training sources")
        public_maps[split] = (pby_id, public_path)
        rows, rejected, counts = [], [], Counter()
        missing = []
        for req in public:
            idx = str(req["idx"])
            target = by_id.get(idx)
            if target is None:
                reasons = ["missing_private_target"]
                missing.append(idx)
            else:
                row, reasons = make_record(target, req, split, args.bit_dim, public_path, target_path, hashes)
                if not reasons:
                    rows.append(row)
                    continue
            counts.update(reasons)
            rejected.append({"parent_id": idx, "reasons": reasons, "declared_length": req.get("length")})
        write_jsonl(args.out_dir / f"native_sources_{split}.jsonl", rows)
        write_jsonl(args.out_dir / f"quarantine_{split}.jsonl", rejected)
        # Smoke records are a deterministic first subset, not new dataset splits.
        smoke_count = 12 if split == "train" else 4
        if split != "test":
            write_jsonl(args.out_dir / f"smoke_{split}.jsonl", rows[:smoke_count])
        allseq[split] = {r["provenance"]["target_sequence_sha256"] for r in rows}
        allids[split] = {r["parent_id"] for r in rows}
        missing_by_split[split] = missing
        lens = sorted(len(r["sequence"]) for r in rows)
        report["splits"][split] = {
            "public_rows": len(public), "private_rows": len(targets), "accepted_rows": len(rows),
            "quarantined_rows": len(rejected), "reasons": dict(counts),
            "target_rows_not_in_public_split": len(set(by_id) - set(pby_id)),
            "length_min": min(lens), "length_max": max(lens), "length_median": lens[len(lens)//2],
            "gt512": sum(n > 512 for n in lens), "unique_sequences": len(allseq[split]),
        }
        print(split, report["splits"][split], flush=True)
        del rows, targets, by_id
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[a + "_" + b] = {
            "identical_sequence_hash_count": len(allseq[a] & allseq[b]),
            "parent_id_count": len(allids[a] & allids[b]),
        }
    report["cross_split_overlap"] = overlaps
    report["split_policy"] = "original split preserved; overlap reported, never silently moved"
    report["exact_sequence_and_id_disjoint_certified"] = not any(v["identical_sequence_hash_count"] or v["parent_id_count"] for v in overlaps.values())
    report["homology_disjoint_certified"] = False
    report["reaction_disjoint_certified"] = False
    if args.audit_pt_fallback:
        import torch
        fallback_report = {}
        for split, names in (("train", ("train_dplm2_tokens.pt", "len512_train_dplm2_tokens.pt")),
                             ("validation", ("val_dplm2_tokens.pt",)), ("test", ("test_dplm2_tokens.pt",))):
            missing = set(missing_by_split[split])
            pby_id, public_path = public_maps[split]
            candidates = []
            for name in names:
                path = args.step1_dir / name
                if not path.exists():
                    continue
                hashes[str(path)] = sha256(path)
                archive = torch.load(path, map_location="cpu", weights_only=False)
                count, rejected = 0, Counter()
                for item in archive:
                    idx = str(item["idx"])
                    if idx not in missing:
                        continue
                    row, reasons = make_record(item, pby_id[idx], split, args.bit_dim, public_path, path, hashes)
                    if reasons:
                        rejected.update(reasons)
                    else:
                        row["provenance"]["quarantine_reason"] = "fallback_tokenization_checkpoint_provenance_requires_review"
                        candidates.append(row)
                        count += 1
                fallback_report[name] = {"rows": len(archive), "full_length_valid_candidates": count,
                                         "rejection_reasons": dict(rejected)}
            write_jsonl(args.out_dir / f"fallback_candidates_{split}.jsonl", candidates)
        report["fallback_audit_not_merged"] = fallback_report
    report["outputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                         for p in sorted(args.out_dir.glob("*.jsonl"))}
    (args.out_dir / "source_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("audit", args.out_dir / "source_audit.json", flush=True)


if __name__ == "__main__":
    main()
