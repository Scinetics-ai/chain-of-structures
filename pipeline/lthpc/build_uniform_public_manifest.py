#!/usr/bin/env python3
"""Audit global field intersection and export uniform, non-ready public goals.

This standalone CPU tool does not change any trainer or existing source file.
Accepted rows, splits and order are retained. No natural sequence/structure/site
answer is copied into the public goal; provenance is a separate sidecar.
Presence/SMILES string parsing is NOT chemical consistency certification.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

SPLITS = ("train", "validation", "test")
PUBLIC_FIELDS = frozenset({"function_text", "final_length", "reaction_smiles", "substrate_smiles",
                           "ec_numbers", "rhea_id", "master_rhea_id"})


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nonempty(value):
    return isinstance(value, str) and value.strip().lower() not in {"", "[]", "{}", "nan", "none", "null", "-"}


def string_list(value):
    """Parse storage syntax only; do not invent values or claim chemical validity."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, list) or not value or any(not nonempty(x) for x in value):
        return None
    return [x.strip() for x in value]


def ec_list(value):
    if not isinstance(value, str):
        return None
    values = re.findall(r"[1-7]\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)", value)
    return values or None


def validate_policy(policy):
    if policy.get("schema") != "agent_dplmbit.public_field_policy.v1" or tuple(policy.get("splits", [])) != SPLITS:
        raise ValueError("Unsupported policy or split order")
    candidates = policy.get("candidate_public_fields", [])
    if not candidates or len(set(candidates)) != len(candidates) or set(candidates) - PUBLIC_FIELDS:
        raise ValueError("Public policy contains unsupported or private fields")
    if set(policy.get("required_goal_fields", [])) - set(candidates):
        raise ValueError("Required fields must be audited public candidates")
    if policy.get("chemistry_gate", {}).get("default_ready") is not False:
        raise ValueError("Uncertified goals must not default to ready")


def read_accepted(source_paths, expected_counts):
    """Keep minimal approved fields, source pointers and private hashes in memory."""
    result, hashes, seen = {}, {}, set()
    for split in SPLITS:
        path, rows, digest = Path(source_paths[split]), [], hashlib.sha256()
        with path.open("rb") as f:
            for line_no, raw in enumerate(f, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("schema") != "natural_reconstruction_source_v1" or row.get("split") != split:
                    raise ValueError(f"{path}:{line_no}: unexpected accepted-source schema/split")
                idx = row.get("parent_id")
                if not isinstance(idx, str) or not idx or idx in seen:
                    raise ValueError(f"Duplicate or missing accepted ID: {idx!r}")
                seen.add(idx)
                seq = row.get("sequence")
                if not isinstance(seq, str) or not seq:
                    raise ValueError(f"Accepted source lacks its existing private target: {idx}")
                rows.append({"parent_id": idx, "split": split,
                             "function_text": "Design an enzyme for the supplied reaction SMILES. Produce one polypeptide chain of the specified final length.",
                             "private_original_function_text": row.get("function_text"),
                             "final_length": len(seq), "ec_numbers": ec_list(row.get("ec_list")),
                             "source_path": str(path), "source_line": line_no,
                             "source_row_sha256": hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest(),
                             "private_sequence_sha256": hashlib.sha256(seq.encode()).hexdigest()})
        if len(rows) != expected_counts[split]:
            raise ValueError(f"Accepted count changed for {split}: {len(rows)} != {expected_counts[split]}")
        result[split], hashes[str(path)] = rows, digest.hexdigest()
    return result, hashes


def read_metadata(path, accepted_ids):
    csv.field_size_limit(10000000)
    result = {}
    # Reference sequence is only hashed for private join verification and then discarded.
    with Path(path).open() as f:
        for line_no, row in enumerate(csv.DictReader(f), 2):
            idx = row.get("IDX")
            if idx not in accepted_ids:
                continue
            if idx in result:
                raise ValueError(f"Duplicate chemistry metadata ID: {idx}")
            result[idx] = {"reaction_smiles": row.get("smarts"), "substrate_smiles": string_list(row.get("SUBSTRATE")),
                           "rhea_id": row.get("RHEA_ID"), "master_rhea_id": row.get("MASTER_ID"),
                           "cofactor_smiles_present": bool(string_list(row.get("COFACTOR_SMILES"))),
                           "cofactor_annotation_present": nonempty(row.get("Cofactor")),
                           "metadata_sequence_sha256": hashlib.sha256(row.get("Sequence", "").encode()).hexdigest(),
                           "metadata_line": line_no}
    return result


def field_values(row, meta):
    values = {k: row.get(k) for k in ("function_text", "final_length", "ec_numbers")}
    values.update({k: meta.get(k) for k in ("reaction_smiles", "substrate_smiles", "rhea_id", "master_rhea_id")})
    for name, value in list(values.items()):
        if name == "final_length":
            valid = type(value) is int and value > 0
        elif name in {"substrate_smiles", "ec_numbers"}:
            valid = isinstance(value, list) and bool(value) and all(nonempty(x) for x in value)
        elif name in {"rhea_id", "master_rhea_id"}:
            valid = isinstance(value, str) and value.strip().isdigit()
        elif name == "reaction_smiles":
            valid = nonempty(value) and value.count(">>") == 1 and all(nonempty(x) for x in value.split(">>"))
        else:
            valid = nonempty(value)
        if not valid:
            values[name] = None
        elif isinstance(value, str):
            values[name] = value.strip()
    return values


def field_intersection(rows_by_split, metadata, policy):
    candidates = policy["candidate_public_fields"]
    coverage, values = {}, {}
    for split in SPLITS:
        missing = {field: [] for field in candidates}
        values[split] = []
        for row in rows_by_split[split]:
            value = field_values(row, metadata.get(row["parent_id"], {}))
            values[split].append(value)
            for field in candidates:
                if value[field] is None:
                    missing[field].append(row["parent_id"])
        count = len(rows_by_split[split])
        coverage[split] = {field: {"present_valid_count": count - len(missing[field]),
                                   "total": count, "missing_or_invalid_ids": missing[field]}
                           for field in candidates}
    included = [field for field in candidates if all(not coverage[s][field]["missing_or_invalid_ids"] for s in SPLITS)]
    return included, coverage, values


def read_certificates(path):
    if path is None:
        return {}
    result = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("split"), row.get("parent_id"))
            if key in result:
                raise ValueError(f"Duplicate chemistry certificate: {key}")
            result[key] = row
    return result


def certificate_status(cert, goal_hash):
    if not cert:
        return "missing"
    if (cert.get("schema") != "agent_dplmbit.chemistry_certificate.v1"
            or cert.get("public_goal_sha256") != goal_hash or cert.get("status") != "pass"
            or not nonempty(cert.get("reviewer")) or not cert.get("evidence_sources")
            or cert.get("checks", {}).get("source_text_reaction_substrate_consistency") is not True
            or cert.get("checks", {}).get("no_private_answer_leakage") is not True
            or cert.get("checks", {}).get("inference_available_information") is not True):
        return "invalid_or_not_passed"
    return "hash_bound_certificate_present"


def known_issues(path):
    if path is None:
        return {}
    result = {}
    data = json.loads(Path(path).read_text())
    for example in data.get("sanitized_initial_goal_examples", []):
        if example.get("source_text_chemistry_consistency") == "example_conflict_detected":
            idx = example["private_provenance_sidecar"]["source_record_idx"]
            result[idx] = example["issue"]
    return result


def make_manifest_record(row, values, included, required, cert=None, issue=None):
    goal = {field: values[field] for field in included}
    if set(goal) - PUBLIC_FIELDS or any(value is None for value in goal.values()):
        raise ValueError("Public goal violates global intersection")
    digest = canonical_hash(goal)
    status = certificate_status(cert, digest)
    reasons = []
    if set(required) - set(included):
        reasons.append("required_goal_field_not_globally_complete")
    if status != "hash_bound_certificate_present":
        reasons.append("chemistry_consistency_not_certified")
    if issue:
        reasons.append("known_source_text_chemistry_conflict")
    return {"schema": "agent_dplmbit.uniform_public_goal.v1", "parent_id": row["parent_id"],
            "split": row["split"], "public_goal": goal, "public_goal_sha256": digest,
            "readiness": {"ready": not reasons, "reasons": reasons, "chemistry_certificate_status": status}}


def as_public_request(record):
    """Explicit opt-in adapter; unverified offline candidates cannot become inputs."""
    if not record["readiness"]["ready"]:
        raise ValueError("Public chemistry goal is not certified ready")
    goal = record["public_goal"]
    if set(goal) - PUBLIC_FIELDS:
        raise ValueError("Unknown public fields")
    # Parent ID, split, hashes and provenance are never rendered into the prompt.
    return {"function_text": json.dumps({k: v for k, v in goal.items() if k != "final_length"},
                                         sort_keys=True, ensure_ascii=False),
            "final_length": goal["final_length"]}


def build(source_paths, metadata_path, policy, out_dir, certificates_path=None, known_issues_path=None):
    validate_policy(policy)
    out_dir = Path(out_dir)
    # Avoid replacing any prior dataset, including one referenced by running jobs.
    out_dir.mkdir(parents=True, exist_ok=False)
    rows, source_hashes = read_accepted(source_paths, policy["expected_accepted_counts"])
    ids = {r["parent_id"] for split in rows.values() for r in split}
    metadata = read_metadata(metadata_path, ids)
    included, coverage, values = field_intersection(rows, metadata, policy)
    certs, issues = read_certificates(certificates_path), known_issues(known_issues_path)
    source_hashes[str(metadata_path)] = file_hash(metadata_path)
    for path in (certificates_path, known_issues_path):
        if path is not None:
            source_hashes[str(path)] = file_hash(path)
    report = {"schema": "agent_dplmbit.uniform_public_audit.v1", "policy_sha256": canonical_hash(policy),
              "source_hashes": source_hashes, "included_public_fields": included,
              "omitted_candidate_fields": [x for x in policy["candidate_public_fields"] if x not in included],
              "excluded_public_fields": policy["excluded_public_fields"], "coverage": coverage, "splits": {},
              "record_filtering": "none; all previously accepted records retained in existing split/order",
              "public_text_handling": "Historical step1 prose is private-only; public function_text is a fixed neutral template without cofactors, mechanism, residue identities/positions or unverified reaction names.",
              "private_target_handling": "Existing private sequence/bit supervision is unchanged; the global intersection policy applies only to public initial-goal fields.",
              "task_input_change": policy.get("task_input_change"),
              "scope": "initial chemical request cards only; not teacher requests, current protein states, tool reports or complete agent SFT",
              "dataset_ready": False, "running_A0_changed": False,
              "certificate_limit": "A supplied certificate is an attributable review assertion, not an independent proof generated by this script.",
              "semantic_limit": "No verified mechanism, CoT action labels, natural positions or reference structures are manufactured."}
    output_hashes = {}
    for split in SPLITS:
        public_path, private_path = out_dir / f"public_{split}.jsonl", out_dir / f"private_provenance_{split}.jsonl"
        readiness, joined, exact_sequence, cofactor, conf_ids = Counter(), 0, 0, 0, []
        with public_path.open("w") as public_file, private_path.open("w") as private_file:
            for row, value in zip(rows[split], values[split]):
                idx, meta = row["parent_id"], metadata.get(row["parent_id"])
                issue = issues.get(idx)
                if meta:
                    joined += 1
                    exact_sequence += meta["metadata_sequence_sha256"] == row["private_sequence_sha256"]
                    cofactor += meta["cofactor_smiles_present"]
                if meta and meta["metadata_sequence_sha256"] != row["private_sequence_sha256"]:
                    issue = {"type": "metadata_reference_sequence_mismatch", "existing_issue": issue}
                result = make_manifest_record(row, value, included, policy["required_goal_fields"],
                                              certs.get((split, idx)), issue)
                readiness[result["readiness"]["chemistry_certificate_status"]] += 1
                readiness["ready" if result["readiness"]["ready"] else "not_ready"] += 1
                if issue:
                    conf_ids.append(idx)
                public_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                sidecar = {"parent_id": idx, "split": split, "private_source_path": row["source_path"],
                           "private_source_line": row["source_line"], "source_row_sha256": row["source_row_sha256"],
                           "private_sequence_sha256": row["private_sequence_sha256"], "chemistry_source": str(metadata_path),
                           "private_original_function_text": row["private_original_function_text"],
                           "chemistry_source_line": meta["metadata_line"] if meta else None,
                           "public_goal_sha256": result["public_goal_sha256"], "known_issue": issue,
                           "policy_sha256": report["policy_sha256"], "model_input": False}
                private_file.write(json.dumps(sidecar, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["splits"][split] = {"accepted_input_records": len(rows[split]), "output_records": len(rows[split]),
                                   "metadata_matched_records": joined, "metadata_sequence_exact_matches": exact_sequence,
                                   "omitted_cofactor_SMILES_present_records": cofactor,
                                   "readiness_counts": dict(readiness), "known_conflict_ids": conf_ids,
                                   "id_order_sha256": canonical_hash([r["parent_id"] for r in rows[split]]),
                                   "length_min": min(r["final_length"] for r in rows[split]),
                                   "length_max": max(r["final_length"] for r in rows[split])}
        output_hashes[str(public_path)] = file_hash(public_path)
        output_hashes[str(private_path)] = file_hash(private_path)
    overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            a = {r["private_sequence_sha256"] for r in rows[left]}
            b = {r["private_sequence_sha256"] for r in rows[right]}
            overlaps[f"{left}__{right}"] = len(a & b)
    report["exact_sequence_cross_split_overlap_counts"] = overlaps
    report["dataset_ready"] = all(s["readiness_counts"].get("not_ready", 0) == 0 for s in report["splits"].values()) and not any(overlaps.values())
    report["output_hashes"] = output_hashes
    (out_dir / "field_policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chemistry-certificates", type=Path)
    p.add_argument("--known-issues-audit", type=Path)
    args = p.parse_args()
    report = build({s: args.source_dir / f"native_sources_{s}.jsonl" for s in SPLITS}, args.metadata_csv,
                   json.loads(args.policy.read_text()), args.out_dir, args.chemistry_certificates, args.known_issues_audit)
    print(json.dumps({k: report[k] for k in ("included_public_fields", "omitted_candidate_fields", "splits", "dataset_ready")}, indent=2))


if __name__ == "__main__":
    main()
