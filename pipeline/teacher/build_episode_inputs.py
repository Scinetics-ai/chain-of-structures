#!/usr/bin/env python3
"""Prepare hash-bound offline agent inputs; no teacher calls and no new labels.

All accepted parents are retained, including blocked inputs. The official Rhea
bulk names are request facts, never catalytic-role evidence. One existing primary
literature card is an optional family hypothesis, never a target-specific label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

from agent_dplmbit.schema import AA_ALPHABET, ContractError, PublicExample, PublicRequest, PublicState, ToolRequest
from agent_dplmbit.tools import ToolRegistry

EXPECTED = {"train": 49711, "validation": 2736, "test": 2724}
FIELDS = {"function_text", "final_length", "reaction_smiles", "substrate_smiles", "ec_numbers", "rhea_id", "master_rhea_id"}
REQUEST_SCHEMA = "agent_dplmbit.teacher_request.v1"
REVIEW_SCHEMA = "agent_dplmbit.teacher_episode_input_review.v1"
VERSION = "episode_input_preparation_v1"
SCOPE = "Input and source consistency only; no chemical candidates, future edits, teacher answers or SFT labels approved."


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_rhea(path):
    """Parse official flat-file fields, preserving named reaction direction."""
    records, current, previous = {}, {}, None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "///":
                if current:
                    rid = current.get("ENTRY", "").removeprefix("RHEA:")
                    if not rid.isdigit() or rid in records:
                        raise ValueError("Invalid or duplicate Rhea entry")
                    records[rid] = current
                current, previous = {}, None
                continue
            key, value = line[:12].strip(), line[12:].strip()
            if key:
                if key in current:
                    current[key] += " " + value
                else:
                    current[key] = value
                previous = key
            elif value and previous:
                current[previous] += " " + value
    if current:
        raise ValueError("Unterminated Rhea record")
    return records


def named_fact(goal, rhea, directions, source_sha):
    master, rid = goal["master_rhea_id"], goal["rhea_id"]
    record, directed = rhea.get(master, {}), rhea.get(rid, {})
    mapping = directions.get(master, {})
    form = next((x for x in ("RHEA_ID_LR", "RHEA_ID_RL") if mapping.get(x) == rid), None)
    errors = []
    if form is None:
        errors.append("named_fact_direction_unresolved")
    if not record.get("DEFINITION") or not directed.get("DEFINITION"):
        errors.append("official_named_reaction_unavailable")
    ecs = sorted(set(re.findall(r"[1-7]\.\d+\.\d+\.\d+", record.get("ENZYME", ""))))
    if not set(goal["ec_numbers"]) <= set(ecs):
        errors.append("bulk_names_EC_link_not_fully_supported")
    if record.get("DEFINITION") and directed.get("DEFINITION") and form:
        sides = record["DEFINITION"].split(" = ")
        wanted = " => ".join(sides if form == "RHEA_ID_LR" else list(reversed(sides)))
        if len(sides) != 2 or wanted != directed["DEFINITION"]:
            errors.append("named_equation_direction_not_verified")
    if not record or not directed:
        return None, errors
    fact = {
        "schema": "agent_dplmbit.reaction_request_facts.v1",
        "source": "https://www.rhea-db.org/rhea/" + master,
        "status": "observed", "tier": "request_facts",
        "master_rhea_id": master, "directional_rhea_id": rid,
        "master_equation": record.get("DEFINITION"),
        "requested_direction_equation": directed.get("DEFINITION"),
        "chebi_equation": directed.get("EQUATION"),
        "official_master_ec_numbers": ecs, "direction_form": form,
        "source_artifact": {"url": "https://ftp.expasy.org/databases/rhea/txt/rhea-reactions.txt.gz",
                            "sha256": source_sha},
        "scope": "Curated names and reaction participants only. No catalytic residue, position, mechanism or effective edit is implied.",
        "approved_as_catalytic_role_label": False,
    }
    fact["fact_sha256"] = digest(fact)
    return fact, errors


def public_request(goal, state=None):
    if set(goal) != FIELDS:
        raise ContractError("Public goal must retain exactly the seven uniform fields")
    text = canonical({k: v for k, v in goal.items() if k != "final_length"})
    n = goal["final_length"]
    state = state or PublicState(n, "initial", (None,) * n, (None,) * n, (True,) * n)
    if state.length != n:
        raise ContractError("Partial-state length does not match reviewed request")
    return PublicExample(PublicRequest(text, n), state)


def partial_state(source, goal, visibility, seed):
    """Observed subset, NOT generated intermediate state or expert edit trace."""
    seq, bits = source["sequence"], source["bits"]
    n = goal["final_length"]
    if len(seq) != n or len(bits) != n or any(a not in AA_ALPHABET for a in seq):
        raise ContractError("Private paired source disagrees with uniform length or AA vocabulary")
    if not 0 < visibility < 1:
        raise ContractError("Partial visibility must be between zero and one")
    rng = random.Random(int(digest([seed, source["parent_id"], visibility]), 16))
    count = min(n - 1, max(1, int(n * visibility)))
    aa_seen = set(rng.sample(range(n), count))
    bits_seen = set(rng.sample(range(n), count))
    state = PublicState(n, "denoising",
        tuple(AA_ALPHABET.index(a) if i in aa_seen else None for i, a in enumerate(seq)),
        tuple(tuple(b) if i in bits_seen else None for i, b in enumerate(bits)),
        (True,) * n, candidate_source="observed")
    return state


def check_record(row, audit, split, raw_row_sha256=None):
    if row.get("schema") != "agent_dplmbit.uniform_public_goal.v1" or set(row.get("public_goal", {})) != FIELDS:
        raise ContractError("Wrong uniform schema or public fields")
    if row["split"] != split or audit["split"] != split or row["parent_id"] != audit["parent_id"]:
        raise ContractError("Source/audit split or parent mismatch")
    if digest(row["public_goal"]) != row["public_goal_sha256"] or audit["public_goal_sha256"] != row["public_goal_sha256"]:
        raise ContractError("Stale public goal hash")
    if audit["source_row_sha256"] != (raw_row_sha256 or digest(row)):
        raise ContractError("Stale input audit row binding")


def make_record(row, audit, fact, fact_errors, *, view="initial", state=None, extra_provenance=None):
    public = public_request(row["public_goal"], state)
    rid = "episode_" + digest([row["split"], row["parent_id"], view, public.to_dict()])[:24]
    request = {"schema": REQUEST_SCHEMA, "request_id": rid, "parent_id": row["parent_id"],
               "split": row["split"], "public": public.to_dict()}
    reasons = list(audit.get("failed_checks", [])) + list(fact_errors)
    if not audit.get("input_consistency_passed") and not reasons:
        reasons.append("source_input_audit_not_passed")
    if fact is None and "official_named_reaction_unavailable" not in reasons:
        reasons.append("official_named_reaction_unavailable")
    binding = {"source_row_sha256": audit["source_row_sha256"], "source_canonical_row_sha256": digest(row), "source_audit_sha256": digest(audit),
               "fact_sha256": digest(fact), "public_sha256": digest(public.to_dict()),
               "scope": SCOPE, "view": view, "extra_provenance": extra_provenance or {}}
    review = {"request_id": rid, "public_sha256": binding["public_sha256"],
              "ready": not reasons, "reviewer": VERSION, "review_sha256": digest(binding),
              "allow_bound_positions": public.state.phase != "initial",
              "reason": "; ".join(reasons) if reasons else "Passed source-consistency and official named-reaction checks; input-only, not output approval."}
    return request, review, binding


def load_verified_sources(args):
    summary = json.loads((args.audit_dir / "summary.json").read_text())
    contract = json.loads((args.audit_dir / "run_contract.json").read_text())
    source_hashes = {}
    for split in EXPECTED:
        source = args.uniform_dir / f"public_{split}.jsonl"
        sh = file_hash(source)
        recorded = next(v for k, v in contract["source_hashes"].items() if Path(k).name == source.name)
        if sh != recorded:
            raise ContractError("Uniform source changed since full audit: " + split)
        audit_path = args.audit_dir / f"record_audit_{split}.jsonl"
        if file_hash(audit_path) != summary["output_files"][audit_path.name]["sha256"]:
            raise ContractError("Source record audit hash mismatch: " + split)
        source_hashes[split] = {"uniform_sha256": sh, "audit_sha256": file_hash(audit_path)}
    with args.directions.open() as f:
        directions = {r["RHEA_ID_MASTER"]: r for r in csv.DictReader(f, delimiter="\t")}
    ds = next(r["sha256"] for r in contract["official_sources"] if r["file"] == args.directions.name)
    if file_hash(args.directions) != ds:
        raise ContractError("Direction file differs from original official input audit")
    receipt = json.loads(args.rhea_receipt.read_text())
    if file_hash(args.rhea_names) != receipt["sha256"] or receipt["url"] != "https://ftp.expasy.org/databases/rhea/txt/rhea-reactions.txt.gz":
        raise ContractError("Named-reaction source receipt mismatch")
    return read_rhea(args.rhea_names), directions, source_hashes, receipt


def paired_rows(args, split):
    seen = set()
    with (args.uniform_dir / f"public_{split}.jsonl").open() as source:
        for line, audit in itertools.zip_longest(source, rows(args.audit_dir / f"record_audit_{split}.jsonl")):
            if line is None or audit is None:
                raise ContractError("Audit/source row count mismatch")
            row = json.loads(line)
            check_record(row, audit, split, hashlib.sha256(line.rstrip("\n").encode()).hexdigest())
            if row["parent_id"] in seen:
                raise ContractError("Duplicate source parent")
            seen.add(row["parent_id"])
            yield row, audit
    if len(seen) != EXPECTED[split]:
        raise ContractError("Original accepted split count changed")


def write_pilot(args, rhea, directions, source_hashes, receipt):
    chosen_ids = [r["parent_id"] for r in rows(args.pilot_requests)]
    if len(chosen_ids) != 8 or len(set(chosen_ids)) != 8:
        raise ContractError("Pilot must preserve the original eight unique parents")
    selected = {}
    for row, audit in paired_rows(args, "train"):
        if row["parent_id"] in chosen_ids:
            selected[row["parent_id"]] = (row, audit)
    if set(selected) != set(chosen_ids):
        raise ContractError("Original pilot parent missing")
    source_rows = {}
    if args.native_train:
        wanted = set(chosen_ids[:2])
        for src in rows(args.native_train):
            if src["parent_id"] in wanted:
                if src["split"] != "train" or src["parent_id"] in source_rows:
                    raise ContractError("Paired-source split or uniqueness error")
                source_rows[src["parent_id"]] = src
        if set(source_rows) != wanted:
            raise ContractError("Partial-state sources missing; no synthetic replacement")
    requests, reviews, bindings, mapping = [], [], [], {}
    for parent in chosen_ids:
        row, audit = selected[parent]
        fact, errors = named_fact(row["public_goal"], rhea, directions, receipt["sha256"])
        req, rev, binding = make_record(row, audit, fact, errors)
        if not rev["ready"]:
            raise ContractError("Previously selected pilot is blocked: " + parent + " " + rev["reason"])
        requests.append(req); reviews.append(rev); bindings.append(binding)
        mapping[req["public"]["request"]["function_text"]] = [fact]
    for index, parent in enumerate(chosen_ids[:2] if source_rows else []):
        row, audit = selected[parent]
        visibility = (.25, .60)[index]
        src = source_rows[parent]
        state = partial_state(src, row["public_goal"], visibility, args.seed)
        fact, errors = named_fact(row["public_goal"], rhea, directions, receipt["sha256"])
        provenance = {"trajectory_type": "fixed_offline_natural_observed_subset_not_generated_trajectory_not_expert_edit",
                      "paired_source_sha256": file_hash(args.native_train), "private_row_sha256": digest(src),
                      "seed": args.seed, "visibility": visibility, "coordinates_exported": False,
                      "hidden_targets_exported": False, "source_legacy_function_text_used": False}
        req, rev, binding = make_record(row, audit, fact, errors, view=f"partial_{visibility}",
                                       state=state, extra_provenance=provenance)
        requests.append(req); reviews.append(rev); bindings.append(binding)
    names = args.output_dir / "pilot_names_only"
    export_batch(names, requests, reviews, bindings, mapping)
    role = json.loads(args.role_evidence.read_text())
    if role.get("status") != "hypothesized" or role.get("approved_as_target_specific_expert_action") is not False:
        raise ContractError("Literature card must remain a non-target-specific hypothesis")
    master = role["reaction_scope"]["master_rhea_id"]
    selected_indices = [i for i, r in enumerate(requests[:8]) if json.loads(r["public"]["request"]["function_text"])["master_rhea_id"] == master]
    if len(selected_indices) != 1:
        raise ContractError("Literature contrast requires one initial parent")
    i = selected_indices[0]
    key = requests[i]["public"]["request"]["function_text"]
    role["original_evidence_sha256"] = role.pop("evidence_sha256", None)
    role["tier"] = "family_hypothesis"
    role["evidence_sha256"] = digest(role)
    export_batch(args.output_dir / "pilot_literature", [requests[i]], [reviews[i]], [bindings[i]],
                 {key: mapping[key] + [role]})
    return {"names_only_requests": len(requests), "names_only_unique_parents": 8,
            "partial_observed_views": len(source_rows), "literature_contrast_requests": 1,
            "literature_vs_names_public_sha256_identical": True,
            "no_new_expert_actions": True, "native_source_read_scope": "two fixed observed subsets only"}


def export_batch(path, requests, reviews, bindings, evidence):
    path.mkdir(parents=True, exist_ok=True)
    for filename, records in (("requests.jsonl", requests), ("input_bindings.jsonl", bindings)):
        (path / filename).write_text("".join(canonical(r) + "\n" for r in records))
    write_json(path / "reviews.json", {"schema": REVIEW_SCHEMA, "requests": reviews})
    write_json(path / "evidence_map.json", evidence)
    traces = []
    registry = ToolRegistry(reaction_evidence=evidence)
    for req in requests:
        public = PublicExample.from_dict(req["public"])
        lookup = ToolRequest("lookup_reaction_evidence", {}, public.state.state_hash)
        obs = registry.execute(lookup, public)
        traces.append({"request_id": req["request_id"], "tool_request": lookup.to_dict(), "observation": obs.to_dict(),
                       "scope": "CPU backend smoke only; not an agent-chosen trajectory"})
    (path / "backend_smoke.jsonl").write_text("".join(canonical(r) + "\n" for r in traces))


def write_full(args, rhea, directions, source_hashes, receipt):
    outputs, mapping, chemical_keys, named_ids, facts = {}, {}, set(), set(), {}
    for split in EXPECTED:
        counts, failures = Counter(), Counter()
        request_file = args.output_dir / f"requests_{split}.jsonl"
        review_file = args.output_dir / f"reviews_{split}.jsonl"
        bind_file = args.output_dir / f"input_bindings_{split}.jsonl"
        with request_file.with_suffix(".jsonl.tmp").open("w") as rf, review_file.with_suffix(".jsonl.tmp").open("w") as vf, bind_file.with_suffix(".jsonl.tmp").open("w") as bf:
            for row, audit in paired_rows(args, split):
                goal = row["public_goal"]
                fact, errors = named_fact(goal, rhea, directions, receipt["sha256"])
                req, rev, binding = make_record(row, audit, fact, errors)
                counts["records"] += 1
                counts["input_ready" if rev["ready"] else "blocked_preserved"] += 1
                counts["named_fact_present" if fact else "named_fact_unavailable"] += 1
                for reason in audit.get("failed_checks", []) + errors:
                    failures[reason] += 1
                chemical_keys.add(goal["reaction_smiles"])
                if fact:
                    named_ids.add(goal["rhea_id"])
                    facts[digest(fact)] = fact
                    text = req["public"]["request"]["function_text"]
                    if rev["ready"]:
                        mapping[text] = [fact]
                rf.write(canonical(req) + "\n"); vf.write(canonical(rev) + "\n"); bf.write(canonical(binding) + "\n")
                if counts["records"] % 1000 == 0:
                    write_json(args.output_dir / "progress.json", {"status": "preparing", "split": split, "records": counts["records"], "resume": "Completed split files are immutable per run; restart into a new output directory if interrupted."})
        for path in (request_file, review_file, bind_file):
            path.with_suffix(".jsonl.tmp").replace(path)
        outputs[split] = {**counts, "failure_classification_counts": dict(failures),
                          "output_sha256": {p.name: file_hash(p) for p in (request_file, review_file, bind_file)}}
    write_json(args.output_dir / "evidence_map_names_only.json", mapping)
    (args.output_dir / "reaction_fact_cache.jsonl").write_text("".join(canonical(facts[k]) + "\n" for k in sorted(facts)))
    return {"splits": outputs, "unique_reaction_strings": len(chemical_keys),
            "unique_directional_reactions_named": len(named_ids), "eligible_exact_request_evidence_keys": len(mapping),
            "all_parents_preserved": True, "no_length_filter": True,
            "role_evidence_in_full_map": False, "native_source_files_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    for name in ("uniform-dir", "audit-dir", "rhea-names", "rhea-receipt", "directions", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pilot-requests", type=Path)
    parser.add_argument("--role-evidence", type=Path)
    parser.add_argument("--native-train", type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ContractError("Use a new output directory; never overwrite a completed audit or frozen pilot")
    rhea, directions, source_hashes, receipt = load_verified_sources(args)
    args.output_dir.mkdir(parents=True)
    result = write_pilot(args, rhea, directions, source_hashes, receipt) if args.mode == "pilot" else write_full(args, rhea, directions, source_hashes, receipt)
    summary = {"schema": "agent_dplmbit.episode_input_preparation.v1", "version": VERSION,
               "script_sha256": file_hash(__file__), "mode": args.mode, "scope": SCOPE,
               "source_hashes": source_hashes, "official_named_reaction_source": receipt,
               "teacher_calls": 0, "GPU_jobs": 0, "source_manifests_modified": False,
               "teacher_outputs_approved_for_training": False, "created_at_utc": datetime.now(timezone.utc).isoformat(), **result}
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "progress.json", {"status": "completed"})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
