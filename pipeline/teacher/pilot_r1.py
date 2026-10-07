#!/usr/bin/env python3
"""Prepare and run 8–16 reviewed train-only R1 teacher drafts, never approved SFT.

No GPU/model service is launched. `prepare`, `inspect`, and `summarize` do not
send inference requests. `smoke` sends one reviewed request; `run` gates the
remaining selected requests on that same first response and reuses its result.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_drafts as teacher
from agent_dplmbit.schema import ContractError
from agent_dplmbit.reasoned_response import parse_reasoned_response

PLAN_SCHEMA = "agent_dplmbit.teacher_r1_pilot.v1"
SUMMARY_SCHEMA = "agent_dplmbit.teacher_r1_pilot_summary.v1"
SCOPE = "train_R1_public_evidence_drafts_only_not_expert_labels_not_full_tool_agent"


def _write_same(path: Path, content: str):
    """Make preparation resumable while refusing to replace a different pilot."""
    if path.exists():
        if path.read_text() != content:
            raise ContractError(f"Refusing to overwrite a different pilot artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_report(path: Path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(teacher.canonical(value) + "\n")
    os.replace(temporary, path)


def _initial_only(request):
    state = request.public.state
    if state.phase != "initial" or any(a is not None for a in state.aa) or any(b is not None for b in state.bits):
        return False
    if state.ca is not None and any(x is not None for x in state.ca):
        return False
    # This first pilot has no generated protein/history to interpret. Reaction
    # evidence may be preassembled, but a protein report must not be fabricated.
    if request.public.history:
        return False
    for observation in request.public.observations:
        if observation.tool == "lookup_reaction_evidence":
            continue
        if (observation.tool == "current_prediction_status"
                and observation.result.get("status") == "unavailable"):
            continue
        return False
    return True


def prepare(config_path, requests_path, reviews_path, output_dir, *, count=8, seed=20261003):
    if type(count) is not int or not 8 <= count <= 16:
        raise ValueError("The real pilot requires 8–16 distinct train parents")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = teacher.read_config(config_path)
    if not config["check_models"]:
        raise ContractError("Pilot requires advertised server-model validation")
    requests = teacher.read_requests(requests_path)
    reviews = teacher.read_reviews(reviews_path)
    eligible, excluded = [], Counter()
    for request in requests:
        if request.split != "train":
            excluded["not_train"] += 1
            continue
        review = reviews.get(request.request_id)
        if review is None or review["ready"] is not True:
            excluded["not_reviewed_ready"] += 1
            continue
        if review["public_sha256"] != request.public_sha256:
            raise ContractError("Ready review public hash differs for " + request.request_id)
        if not _initial_only(request):
            excluded["not_initial_public_only"] += 1
            continue
        if review["allow_bound_positions"] is not False:
            excluded["review_does_not_explicitly_forbid_addresses"] += 1
            continue
        eligible.append(request)
    eligible.sort(key=lambda r: teacher.digest([seed, r.parent_id, r.request_id]))
    distinct, seen = [], set()
    for request in eligible:
        if request.parent_id in seen:
            excluded["duplicate_parent"] += 1
            continue
        seen.add(request.parent_id)
        distinct.append(request)
    if len(distinct) < count:
        raise ContractError(f"Only {len(distinct)} eligible distinct train/R1 parents; need {count}. "
                            "No review was fabricated and no validation/test input was substituted.")
    selected = distinct[:count]
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_rows = [{"schema": teacher.REQUEST_SCHEMA, "request_id": r.request_id,
                     "parent_id": r.parent_id, "split": r.split, "public": r.public.to_dict()}
                    for r in selected]
    review_rows = [reviews[r.request_id] for r in selected]
    artifacts = {
        "config.json": teacher.canonical(config) + "\n",
        "requests.jsonl": "".join(teacher.canonical(r) + "\n" for r in request_rows),
        "reviews.json": teacher.canonical({"schema": teacher.REVIEW_SCHEMA, "requests": review_rows}) + "\n",
        "smoke_requests.jsonl": teacher.canonical(request_rows[0]) + "\n",
        "smoke_reviews.json": teacher.canonical({"schema": teacher.REVIEW_SCHEMA, "requests": review_rows[:1]}) + "\n",
    }
    for name, text in artifacts.items():
        _write_same(directory / name, text)
    plan = {"schema": PLAN_SCHEMA, "scope": SCOPE, "count": count, "seed": seed,
            "request_ids": [r.request_id for r in selected],
            "parent_ids": [r.parent_id for r in selected], "smoke_request_id": selected[0].request_id,
            "selection": "hash_rank_of_ready_train_initial_requests_one_per_parent",
            "eligible_distinct_parents": len(distinct), "excluded_counts": dict(excluded),
            "teacher_prompt_version": teacher.PROMPT_VERSION,
            "teacher_client_sha256": teacher.sha_file(teacher.__file__),
            "pilot_client_sha256": teacher.sha_file(__file__),
            "source_sha256": {"config": teacher.sha_file(config_path), "requests": teacher.sha_file(requests_path),
                              "reviews": teacher.sha_file(reviews_path)},
            "artifact_sha256": {name: teacher.sha_file(directory / name) for name in artifacts},
            "approved_for_training": False, "semantic_claims_verified": False}
    _write_same(directory / "plan.json", teacher.canonical(plan) + "\n")
    return plan


def load_plan(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("approved_for_training") is not False
            or plan.get("semantic_claims_verified") is not False):
        raise ContractError("Not an unapproved R1 pilot plan")
    if (plan["teacher_prompt_version"] != teacher.PROMPT_VERSION
            or plan["teacher_client_sha256"] != teacher.sha_file(teacher.__file__)
            or plan["pilot_client_sha256"] != teacher.sha_file(__file__)):
        raise ContractError("Teacher/pilot code changed; prepare a new audited pilot directory")
    for name, expected in plan["artifact_sha256"].items():
        if Path(name).name != name or teacher.sha_file(directory / name) != expected:
            raise ContractError("Prepared pilot artifact changed: " + name)
    return directory, plan


def _expected_requests(directory):
    config = teacher.read_config(directory / "config.json")
    reviews = teacher.read_reviews(directory / "reviews.json")
    result = {}
    for request in teacher.read_requests(directory / "requests.jsonl"):
        review = reviews[request.request_id]
        payload = {"model": teacher.MODEL, "messages": teacher.messages_for(request, review), "stream": False, "n": 1,
                   "max_tokens": config["max_tokens"], "temperature": config["temperature"], "top_p": config["top_p"],
                   "seed": config["seed"], "chat_template_kwargs": teacher.chat_template_kwargs(config)}
        key = teacher.digest({"request_id": request.request_id, "public_sha256": request.public_sha256,
                              "review": review, "payload": payload, "prompt_version": teacher.PROMPT_VERSION})
        result[key] = request
    return result


def summarize(directory, *, request_ids=None):
    directory, plan = load_plan(directory)
    expected = _expected_requests(directory)
    wanted = set(plan["request_ids"] if request_ids is None else request_ids)
    if not wanted <= set(plan["request_ids"]):
        raise ContractError("Summary includes an unselected request")
    latest, records, usage = {}, 0, Counter()
    path = directory / "drafts.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if (row.get("schema") != teacher.DRAFT_SCHEMA or row.get("approved_for_training") is not False
                or row.get("semantic_claims_verified") is not False):
            raise ContractError("Output contains non-draft or promoted training data")
        key = row.get("request_hash")
        request = expected.get(key)
        if request is None or (row.get("request_id"), row.get("parent_id"), row.get("split"), row.get("public_sha256")) != (
                request.request_id, request.parent_id, "train", request.public_sha256):
            raise ContractError("Output is not bound to this exact selected public request")
        if request.request_id not in wanted:
            continue
        records += 1
        latest[request.request_id] = row
        reported = row.get("diagnostics", {}).get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = reported.get(field)
            if type(value) is int and value >= 0:
                usage[field] += value
    counts, aa, errors, summaries = Counter(), Counter(), [], set()
    by_id = {request.request_id: request for request in expected.values()}
    for request_id, row in latest.items():
        status = row.get("status")
        counts[str(status)] += 1
        if status != "draft_schema_valid":
            errors.append({"request_id": request_id, "reason": row.get("validation_error", row.get("error", status))})
            continue
        try:
            response = parse_reasoned_response(row["visible_content"], by_id[request_id].public)
            teacher.validate_reference_menu(by_id[request_id].public, response.reasoning["evidence_refs"])
            if response.final["type"] != "condition":
                raise ContractError("R1 pilot final must be a condition")
            if response.final != row.get("validated_response", {}).get("final"):
                raise ContractError("Stored final differs from actual visible response")
            hypotheses = response.final["hypotheses"]
            if len(hypotheses) > 3:
                raise ContractError("R1 pilot candidate budget exceeded")
            if any(h["address_binding"] != "pending" or h["position_1based"] is not None
                   or h["residue_id"] is not None for h in hypotheses):
                raise ContractError("R1 pilot cannot bind natural or proposed protein positions")
            if any(h["action"] not in {"NONE", "PREFER"} for h in hypotheses):
                raise ContractError("R1 pilot does not implement KEEP/REOPEN actions")
            counts["revalidated_r1_contract"] += 1
            counts["empty_hypotheses"] += not hypotheses
            counts["hypotheses"] += len(hypotheses)
            counts["public_evidence_refs"] += len(response.resolved_evidence)
            summaries.add(response.reasoning["summary"])
            aa.update(h["aa_preference"] for h in hypotheses)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            counts["summary_contract_rejected"] += 1
            errors.append({"request_id": request_id, "reason": str(exc)})
    complete = set(latest) == wanted
    return {"schema": SUMMARY_SCHEMA, "scope": SCOPE, "model": teacher.MODEL,
            "expected_requests": len(wanted), "completed_requests": len(latest),
            "missing_request_ids": sorted(wanted - set(latest)), "output_records_including_retries": records,
            "counts": dict(counts), "aa_candidate_counts": dict(aa), "distinct_summary_texts": len(summaries),
            "reported_usage_sum": dict(usage),
            "usage_scope": "reported responses only; failed transport/retries may have unreported server cost",
            "errors": errors, "technical_contract_passed": complete and counts["revalidated_r1_contract"] == len(wanted),
            "approved_for_training": False, "semantic_claims_verified": False,
            "protein_generation_tested": False, "agent_tool_loop_tested": False,
            "model_weight_revision_verified_by_client": False}


async def execute(directory, *, mode, transport=None, token_counter=None):
    if mode not in {"inspect", "smoke", "run"}:
        raise ValueError("Unsupported pilot mode")
    directory, plan = load_plan(directory)
    paths = (directory / "config.json", directory / "requests.jsonl", directory / "reviews.json",
             directory / "drafts.jsonl")
    audit = await teacher.run(*paths, dry_run=True, transport=transport, token_counter=token_counter)
    _write_report(directory / "context_audit.json", audit)
    if mode == "inspect":
        return audit
    if audit["blocked"] or audit["ready_with_verified_context"] != plan["count"]:
        raise ContractError("The whole selected pilot must pass exact context/readiness checks before HTTP")
    start = time.monotonic()
    smoke_run = await teacher.run(paths[0], directory / "smoke_requests.jsonl", directory / "smoke_reviews.json",
                                 paths[3], transport=transport, token_counter=token_counter)
    smoke = summarize(directory, request_ids=[plan["smoke_request_id"]])
    smoke["elapsed_seconds_this_invocation"] = time.monotonic() - start
    smoke["client_run"] = smoke_run
    _write_report(directory / "smoke_summary.json", smoke)
    if not smoke["technical_contract_passed"]:
        return {"mode": mode, "remaining_pilot_dispatched": False, "smoke": smoke}
    if mode == "smoke":
        return smoke
    remainder = await teacher.run(*paths, transport=transport, token_counter=token_counter)
    summary = summarize(directory)
    summary["elapsed_seconds_this_invocation"] = time.monotonic() - start
    summary["client_run"] = remainder
    summary["smoke_request_reused"] = True
    _write_report(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare", help="Select reviewed R1 train inputs; no HTTP")
    for flag in ("config", "requests", "review-manifest", "output-dir"):
        prep.add_argument("--" + flag, required=True)
    prep.add_argument("--count", type=int, default=8)
    prep.add_argument("--seed", type=int, default=20261003)
    for mode in ("inspect", "smoke", "run", "summarize"):
        command = sub.add_parser(mode)
        command.add_argument("--pilot-dir", required=True)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        result = prepare(args.config, args.requests, args.review_manifest, args.output_dir, count=args.count, seed=args.seed)
    elif args.mode == "summarize":
        result = summarize(args.pilot_dir)
        _write_report(Path(args.pilot_dir) / "summary.json", result)
    else:
        result = asyncio.run(execute(args.pilot_dir, mode=args.mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("technical_contract_passed") is False or result.get("remaining_pilot_dispatched") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
