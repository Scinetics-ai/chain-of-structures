#!/usr/bin/env python3
"""CPU-only train teacher-input plan; no HTTP, model inference or SFT approval.

Stream all parents, use disk indexes, and retain every blocked row. Deduplication
is exact public snapshot + supplied evidence, within train only. Shard readiness
certifies inputs; runtime token budgets and the stage quality gate remain open.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_drafts as g
from agent_dplmbit.schema import AA_ALPHABET, ContractError, PublicExample, ToolRequest
from agent_dplmbit.tools import ToolRegistry

VERSION = "train_role_evidence_queue_v1_20261003"
REVIEW_SCHEMA = "agent_dplmbit.teacher_episode_input_review.v1"
MATCHES = {"exact_ChEBI_stoichiometry_same_direction", "exact_ChEBI_stoichiometry_reverse_direction",
           "EC_only_family_hypothesis"}


def require(ok, message):
    if not ok:
        raise ContractError(message)


def rows(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def line(stream, value):
    stream.write(g.canonical(value) + "\n")


def association(matches):
    values = set(matches)
    if "exact_ChEBI_stoichiometry_same_direction" in values:
        return "exact_same_direction"
    if "exact_ChEBI_stoichiometry_reverse_direction" in values:
        return "exact_reverse_direction"
    return "EC_only" if "EC_only_family_hypothesis" in values else "none"


def qualified_card(card, goal):
    """Recheck exported qualification; upstream accession exclusions stay audited upstream."""
    reasons = []
    if not (isinstance(card, dict) and card.get("schema") == "agent_dplmbit.mcsa_role_evidence.v1"
            and card.get("status") == "hypothesized" and card.get("tier") == "family_hypothesis"
            and type(card.get("mcsa_entry_id")) is int and card["mcsa_entry_id"] > 0
            and card.get("approved_as_target_label") is False and card.get("approved_for_SFT") is False):
        return ["role_card_schema_or_scope"]
    if card.get("source_reaction_ec") not in goal["ec_numbers"]:
        reasons.append("role_EC_does_not_match_public_request")
    source = card.get("source", "")
    if source != f"https://www.ebi.ac.uk/thornton-srv/m-csa/entry/{card['mcsa_entry_id']}/":
        reasons.append("role_source_not_bound_to_MCSA_entry")
    a = card.get("reaction_association", {})
    if a.get("requested_rhea_id") != goal["rhea_id"] or a.get("match") not in MATCHES:
        reasons.append("role_reaction_association_mismatch")
    mechanisms = card.get("mechanism_provenance", [])
    if not mechanisms or any(type(m.get("rating")) is not int or m["rating"] not in {2, 3} for m in mechanisms):
        reasons.append("role_mechanism_rating_not_qualified")
    if not any(r.get("doi") or r.get("pubmed_id") for m in mechanisms for r in m.get("references", [])):
        reasons.append("role_without_traceable_DOI_or_PMID")
    hypotheses = card.get("aa_role_hypotheses", [])
    if not hypotheses or any(h.get("aa") not in AA_ALPHABET or len(h.get("aa", "")) != 1
                            or not h.get("roles") or any(not re.fullmatch(r"EMO_\d+", r.get("emo", ""))
                                                        for r in h["roles"]) for h in hypotheses):
        reasons.append("role_missing_canonical_AA_or_EMO")
    if not g.valid_sha(card.get("source_snapshot", {}).get("sha256")):
        reasons.append("role_source_snapshot_hash_missing")
    # These fields cannot be present in address-free external family cards.
    prohibited = {"position", "position_1based", "residue_id", "residue_number", "resnum",
                  "sequence", "coordinates", "accession", "uniprot_id", "chain", "chain_id"}
    def forbidden(value):
        if isinstance(value, dict):
            return any(k in prohibited or forbidden(v) for k, v in value.items())
        return isinstance(value, list) and any(forbidden(v) for v in value)
    if forbidden(card):
        reasons.append("role_contains_target_or_source_residue_addresses")
    return reasons


def pilot_registry(paths):
    result, sources = {}, {}
    for directory in paths:
        directory = Path(directory)
        rp, ep = directory / "requests.jsonl", directory / "evidence_map.json"
        evidence = json.loads(ep.read_text()); g._private_guard(evidence)
        sources[str(rp)] = g.sha_file(rp); sources[str(ep)] = g.sha_file(ep)
        for request in g.read_requests(rp):
            require(request.split == "train", "Pilot registration must not merge validation/test into train")
            key = request.public.request.function_text
            require(key in evidence, "Pilot request lacks exact evidence key")
            evidence_sha = g.digest(evidence[key])
            queue_key = g.digest({"split": "train", "public_sha256": request.public_sha256,
                                  "evidence_sha256": evidence_sha})
            result.setdefault(queue_key, []).append({"directory": str(directory), "request_id": request.request_id,
                "parent_id": request.parent_id, "public_sha256": request.public_sha256, "evidence_sha256": evidence_sha,
                "status": "exact_input_reuse_candidate_only", "reuse_validated": False,
                "teacher_result_status": "not_checked", "approved_for_training": False})
    return result, sources


def prepare_coverage(path, db):
    count = 0
    for number, row in rows(path):
        require(row.get("split") == "train", "Coverage includes validation/test; train planner refuses mixing")
        require(isinstance(row.get("parent_id"), str) and row["parent_id"]
                and g.valid_sha(row.get("public_goal_sha256")) and type(row.get("input_ready")) is bool
                and row.get("records_preserved") is True and row.get("approved_as_target_label") is False,
                "Invalid source coverage record")
        require(isinstance(row.get("conservative_matches"), dict)
                and set(row["conservative_matches"].values()) <= MATCHES, "Unsupported coverage association")
        try:
            db.execute("INSERT INTO coverage VALUES(?,?)", (row["parent_id"], g.canonical(row)))
        except sqlite3.IntegrityError as exc:
            raise ContractError("Duplicate coverage parent") from exc
        count += 1
    db.commit()
    return count


def plan(input_dir, coverage_path, role_map_path, names_map_path, output_dir, *,
         pilot_dirs=(), shard_size=32, expected_parents=49711, role_audit_paths=()):
    require(type(shard_size) is int and 1 <= shard_size <= 32, "Each shard must contain 1..32 requests")
    require(type(expected_parents) is int and expected_parents > 0, "Expected train parent count is required")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    require(not output_dir.exists(), "New plan needs a new output directory; never overwrite sealed inputs")
    source_paths = [input_dir / f"{kind}_train.jsonl" for kind in ("requests", "reviews", "input_bindings")]
    req_path, rev_path, binding_path = source_paths
    sources = {str(p): g.sha_file(p) for p in [*source_paths, Path(coverage_path), Path(role_map_path),
                                             Path(names_map_path), *map(Path, role_audit_paths)]}
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        source_summary = json.loads(summary_path.read_text())
        recorded = source_summary["splits"]["train"]["output_sha256"]
        require(source_summary["splits"]["train"]["records"] == expected_parents, "Expected parent count differs from source receipt")
        for p in source_paths:
            require(recorded.get(p.name) == sources[str(p)], "Source file differs from immutable full_v1 receipt")
        sources[str(summary_path)] = g.sha_file(summary_path)
    roles = json.loads(Path(role_map_path).read_text())
    names = json.loads(Path(names_map_path).read_text())
    require(isinstance(roles, dict) and isinstance(names, dict), "Evidence maps must be exact-public-text objects")
    g._private_guard(roles); g._private_guard(names)
    pilot, pilot_sources = pilot_registry(pilot_dirs)
    sources.update(pilot_sources)
    # Original names come first, matching the reviewed pilot interface.
    merged = {key: names.get(key, []) + values for key, values in roles.items()}
    g._private_guard(merged)
    registry = ToolRegistry(reaction_evidence=merged, max_items=64)
    output_dir.mkdir(parents=True)
    db = sqlite3.connect(output_dir / "planning_index.sqlite")
    db.executescript("CREATE TABLE coverage(parent TEXT PRIMARY KEY, value TEXT);"
                     "CREATE TABLE seen(parent TEXT PRIMARY KEY, request_id TEXT UNIQUE);"
                     "CREATE TABLE queue(key TEXT PRIMARY KEY, representative TEXT, value TEXT, review TEXT, binding TEXT,"
                     "class TEXT, parents INTEGER, first_line INTEGER);"
                     "CREATE TABLE ledger(row_number INTEGER PRIMARY KEY, value TEXT);")
    counts, input_reasons, blockers, eligible_classes, all_evidence_classes = Counter(), Counter(), Counter(), Counter(), Counter()
    observations = output_dir / "backend_smoke.jsonl"
    try:
        require(prepare_coverage(coverage_path, db) == expected_parents, "Coverage does not preserve all expected train parents")
        with observations.open("w") as smoke:
            for indexed in itertools.zip_longest(rows(req_path), rows(rev_path), rows(binding_path)):
                require(all(x is not None for x in indexed), "Train requests/reviews/bindings row counts disagree")
                (n, request), (_, source_review), (_, binding) = indexed
                require(set(request) == {"schema", "request_id", "parent_id", "split", "public"}
                        and request["schema"] == g.REQUEST_SCHEMA and request["split"] == "train", "Train request schema/split mismatch")
                g._audit_public_fields(request["public"]); g._private_guard(request["public"])
                public = PublicExample.from_dict(request["public"])
                require(public.state.phase == "initial" and not public.history and not public.observations
                        and all(a is None for a in public.state.aa) and all(b is None for b in public.state.bits)
                        and (public.state.ca is None or all(x is None for x in public.state.ca)),
                        "Production queue accepts initial all-mask public inputs only; no natural positions")
                public_sha = g.digest(public.to_dict())
                require(source_review.get("request_id") == request["request_id"]
                        and source_review.get("public_sha256") == binding.get("public_sha256") == public_sha
                        and source_review.get("review_sha256") == g.digest(binding)
                        and type(source_review.get("ready")) is bool and source_review.get("allow_bound_positions") is False,
                        "Source review/binding does not certify this exact first-round input")
                try:
                    db.execute("INSERT INTO seen VALUES(?,?)", (request["parent_id"], request["request_id"]))
                except sqlite3.IntegrityError as exc:
                    raise ContractError("Duplicate train parent/request; parent ledger must be one-to-one") from exc
                match = db.execute("SELECT value FROM coverage WHERE parent=?", (request["parent_id"],)).fetchone()
                require(match is not None, "Train parent missing from role coverage")
                coverage = json.loads(match[0])
                goal = json.loads(public.request.function_text)
                require(isinstance(goal, dict) and "final_length" not in goal, "Public function text is not the uniform goal")
                goal["final_length"] = public.request.final_length
                require(g.digest(goal) == coverage["public_goal_sha256"]
                        and coverage["input_ready"] == source_review["ready"], "Role coverage belongs to different goal/readiness")
                text = public.request.function_text
                cards = roles.get(text, [])
                qualification_errors = []
                for card in cards:
                    qualification_errors += qualified_card(card, goal)
                    card_id = str(card.get("mcsa_entry_id"))
                    if coverage["conservative_matches"].get(card_id) != card.get("reaction_association", {}).get("match"):
                        qualification_errors.append("projected_role_not_in_parent_conservative_coverage")
                role_ok = bool(cards) and not qualification_errors
                fact_list = names.get(text, [])
                named_ok = bool(fact_list) and all(f.get("schema") == "agent_dplmbit.reaction_request_facts.v1"
                    and f.get("status") == "observed" and f.get("tier") == "request_facts"
                    and f.get("directional_rhea_id") == goal["rhea_id"]
                    and f.get("master_rhea_id") == goal["master_rhea_id"]
                    and isinstance(f.get("requested_direction_equation"), str) and f["requested_direction_equation"]
                    and f.get("approved_as_catalytic_role_label") is False for f in fact_list)
                if source_review["ready"] and named_ok:
                    require(len(fact_list) == 1 and g.digest(fact_list[0]) == binding["fact_sha256"],
                            "Official names map changed from the certified input fact")
                reasons = []
                if not source_review["ready"]:
                    reasons.append("input_blocked")
                    input_reasons[source_review.get("reason", "unspecified")] += 1
                if not coverage["conservative_matches"]:
                    reasons.append("no_qualified_family_role_evidence")
                elif not cards:
                    reasons.append("qualified_coverage_not_projected_to_public_input")
                if qualification_errors:
                    reasons.append("projected_role_qualification_failed")
                if source_review["ready"] and not named_ok:
                    reasons.append("official_named_reaction_missing_or_invalid")
                eligible = source_review["ready"] and role_ok and named_ok
                evidence_sha = g.digest(merged[text]) if text in merged else None
                cls = association(card["reaction_association"]["match"] for card in cards) if role_ok else "none"
                coverage_cls = association(coverage["conservative_matches"].values())
                all_evidence_classes[coverage_cls] += 1
                queue_key, representative = None, None
                if eligible:
                    queue_key = g.digest({"split": "train", "public_sha256": public_sha, "evidence_sha256": evidence_sha})
                    existing = db.execute("SELECT representative FROM queue WHERE key=?", (queue_key,)).fetchone()
                    if existing:
                        representative = existing[0]
                        db.execute("UPDATE queue SET parents=parents+1 WHERE key=?", (queue_key,))
                    else:
                        representative = request["request_id"]
                        state_before = public.state.state_hash
                        observation = registry.execute(ToolRequest("lookup_reaction_evidence", {}, state_before), public)
                        g._private_guard(observation.to_dict())
                        require(public.state.state_hash == state_before and observation.result.get("status") == "available"
                                and observation.result.get("records") == merged[text], "Actual lookup failed or changed evidence/state")
                        queue_binding = {"schema": "agent_dplmbit.teacher_queue_input_binding.v1", "queue_key": queue_key,
                            "request_id": representative, "split": "train", "public_sha256": public_sha,
                            "evidence_sha256": evidence_sha, "source_binding_sha256": g.digest(binding),
                            "source_review_sha256": g.digest(source_review), "source_coverage_sha256": g.digest(coverage),
                            "tool_observation_sha256": g.digest(observation.to_dict()), "scope": "input/evidence consistency only; no SFT approval"}
                        ready_review = {"request_id": representative, "public_sha256": public_sha, "ready": True,
                            "reviewer": VERSION, "review_sha256": g.digest(queue_binding), "allow_bound_positions": False,
                            "reason": "Source input checks and address-free family evidence passed; teacher output not approved."}
                        db.execute("INSERT INTO queue VALUES(?,?,?,?,?,?,1,?)", (queue_key, representative,
                            g.canonical(request), g.canonical(ready_review), g.canonical(queue_binding), cls, n))
                        line(smoke, {"queue_key": queue_key, "request_id": representative,
                            "public_sha256": public_sha, "evidence_sha256": evidence_sha,
                            "state_hash_before": state_before, "state_hash_after": public.state.state_hash,
                            "observation": observation.to_dict(), "private_guard_passed": True,
                            "teacher_called": False, "approved_for_training": False})
                    eligible_classes[cls] += 1
                counts["parents"] += 1
                counts["input_ready" if source_review["ready"] else "input_blocked"] += 1
                counts["eligible_parents" if eligible else "blocked_parents"] += 1
                counts["coverage_with_qualified_roles"] += bool(coverage["conservative_matches"])
                if source_review["ready"] and not coverage["conservative_matches"]:
                    counts["input_ready_but_no_qualified_roles"] += 1
                if source_review["ready"] and coverage["conservative_matches"] and not eligible:
                    counts["input_ready_role_coverage_but_runtime_evidence_blocked"] += 1
                blockers.update(reasons)
                entry = {"schema": "agent_dplmbit.teacher_queue_parent.v1", "row_number": n,
                    "parent_id": request["parent_id"], "request_id": request["request_id"], "split": "train",
                    "public_sha256": public_sha, "source_request_sha256": g.digest(request),
                    "source_review_sha256": g.digest(source_review), "source_input_binding_sha256": g.digest(binding),
                    "coverage_sha256": g.digest(coverage), "input_ready": source_review["ready"],
                    "input_block_reason": None if source_review["ready"] else source_review.get("reason"),
                    "coverage_role_association": coverage_cls, "projected_role_association": cls,
                    "projected_role_count": len(cards), "role_qualification_errors": sorted(set(qualification_errors)),
                    "evidence_sha256": evidence_sha, "eligible_role_evidence_queue": eligible,
                    "blockers": reasons, "queue_key": queue_key, "representative_request_id": representative,
                    "pilot_reuse_candidates": pilot.get(queue_key, []), "approved_for_training": False}
                db.execute("INSERT INTO ledger VALUES(?,?)", (n, g.canonical(entry)))
                if n % 1000 == 0:
                    db.commit()
                    write_json(output_dir / "progress.json", {"status": "planning_only", **counts})
        db.commit()
        require(counts["parents"] == expected_parents, "Not all expected train parents were retained")
        with (output_dir / "parent_ledger.jsonl").open("w") as stream:
            for (value,) in db.execute("SELECT value FROM ledger ORDER BY row_number"):
                stream.write(value + "\n")
        queue_counts, shard_receipts = Counter(), []
        cursor = db.execute("SELECT key,representative,value,review,binding,class,parents FROM queue ORDER BY first_line")
        with (output_dir / "queue_entries.jsonl").open("w") as queue_stream:
            for shard_index in itertools.count():
                batch = cursor.fetchmany(shard_size)
                if not batch:
                    break
                shard = output_dir / "shards" / f"{shard_index:05d}"
                shard.mkdir(parents=True)
                shard_map, reviews, bindings = {}, [], []
                with (shard / "requests.jsonl").open("w") as stream:
                    for key, representative, value, ready_review, queue_binding, cls, parents in batch:
                        request = json.loads(value); line(stream, request)
                        text = request["public"]["request"]["function_text"]
                        shard_map[text] = merged[text]
                        reviews.append(json.loads(ready_review)); bindings.append(json.loads(queue_binding))
                        queue_counts[cls] += 1
                        line(queue_stream, {"queue_key": key, "representative_request_id": representative,
                            "split": "train", "parent_count": parents, "association": cls,
                            "shard": f"shards/{shard_index:05d}", "pilot_reuse_candidates": pilot.get(key, []),
                            "dispatch_ready": False, "validation_gate_passed": False,
                            "blocking_reason": "missing_stage_validation", "requires_runtime_tokenizer_context_gate": True,
                            "approved_for_training": False})
                write_json(shard / "reviews.json", {"schema": REVIEW_SCHEMA, "requests": reviews})
                write_json(shard / "evidence_map.json", shard_map)
                with (shard / "input_bindings.jsonl").open("w") as stream:
                    for binding in bindings:
                        line(stream, binding)
                # Real client guards on bounded shard only; do not load the whole corpus.
                require(len(g.read_requests(shard / "requests.jsonl")) == len(batch), "Collector request contract mismatch")
                files = {p.name: g.sha_file(p) for p in sorted(shard.iterdir()) if p.is_file()}
                manifest = {"schema": "agent_dplmbit.teacher_queue_shard.v1", "split": "train", "requests": len(batch),
                    "parent_count": sum(r[-1] for r in batch), "files_sha256": files,
                    "source_hashes": sources, "dispatch_ready": False, "validation_gate_passed": False,
                    "blocking_reason": "missing_stage_validation", "requires_runtime_tokenizer_context_gate": True,
                    "teacher_calls": 0, "approved_for_training": False}
                manifest["manifest_sha256"] = g.digest(manifest)
                write_json(shard / "manifest.json", manifest)
                shard_receipts.append({"path": str(shard.relative_to(output_dir)), "requests": len(batch),
                                       "manifest_sha256": manifest["manifest_sha256"]})
        pilot_keys = {r[0] for r in db.execute("SELECT key FROM queue")}.intersection(pilot)
        summary = {"schema": "agent_dplmbit.teacher_queue_plan.v1", "version": VERSION, **counts,
            "expected_train_parents": expected_parents, "all_parents_preserved": True, "length_filter_applied": False,
            "eligible_parent_associations": dict(eligible_classes), "unique_teacher_requests": sum(queue_counts.values()),
            "unique_request_associations": dict(queue_counts), "coverage_associations_all_parents": dict(all_evidence_classes),
            "deduplicated_parent_requests": counts["eligible_parents"] - sum(queue_counts.values()),
            "blocked_reasons_may_overlap": dict(blockers), "input_failure_reasons": dict(input_reasons),
            "shards": shard_receipts, "max_requests_per_shard": shard_size,
            "pilot_exact_input_reuse_candidate_keys": len(pilot_keys), "pilot_results_checked_or_promoted": False,
            "dispatch_ready": False, "validation_gate_passed": False, "blocking_reason": "missing_stage_validation",
            "runtime_context_gate": "Exact tokenizer check before each real call; no truncation or length-based deletion",
            "teacher_calls": 0, "GPU_jobs": 0, "approved_for_training": False, "source_hashes": sources,
            "limitations": ["Exact ChEBI association is not unique mechanism or target-specific catalytic truth.",
                "EC-only matches remain separately counted family hypotheses.",
                "Qualification depends on audited upstream family-overlap/location exclusions; this planner rechecks exported cards.",
                "Pilot input equality does not certify output validity, identical prompt/runtime, or reusable SFT.",
                "Parent mappings preserve corpus provenance; duplicated inputs are not independent teacher evidence."]}
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "pilot_registry.json", {"entries": pilot, "results_checked": False, "approved_for_training": False})
        write_json(output_dir / "progress.json", {"status": "sealed_planning_only", **counts})
    finally:
        db.close()
    (output_dir / "planning_index.sqlite").unlink()
    manifest = {"schema": "agent_dplmbit.teacher_queue_seal.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": sources, "planner_sha256": g.sha_file(__file__), "client_guard_sha256": g.sha_file(g.__file__),
        "tool_registry_sha256": g.sha_file(Path(__file__).resolve().parents[2] / "src/agent_dplmbit/tools.py"),
        "files_sha256": {str(p.relative_to(output_dir)): g.sha_file(p) for p in sorted(output_dir.rglob("*")) if p.is_file()},
        "teacher_calls": 0, "dispatch_ready": False, "validation_gate_passed": False, "approved_for_training": False}
    manifest["manifest_sha256"] = g.digest(manifest)
    write_json(output_dir / "plan_manifest.json", manifest)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True); p.add_argument("--coverage", required=True)
    p.add_argument("--roles-map", required=True); p.add_argument("--names-map", required=True)
    p.add_argument("--output", required=True); p.add_argument("--pilot-dir", action="append", default=[])
    p.add_argument("--role-audit", action="append", default=[])
    p.add_argument("--shard-size", type=int, default=32); p.add_argument("--expected-parents", type=int, default=49711)
    a = p.parse_args()
    result = plan(a.input_dir, a.coverage, a.roles_map, a.names_map, a.output, pilot_dirs=a.pilot_dir,
                  role_audit_paths=a.role_audit, shard_size=a.shard_size, expected_parents=a.expected_parents)
    print(json.dumps({k: result[k] for k in ("parents", "eligible_parents", "unique_teacher_requests",
                     "input_blocked", "input_ready_but_no_qualified_roles", "dispatch_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
