#!/usr/bin/env python3
"""Replay public teacher episodes and export only externally reviewed fragments.

Mechanical validity is not semantic approval. These are interchange records,
NOT an input format already accepted by a trainer. No GPU, HTTP, target protein
or private endpoint is used. Empty/rejected reviews export zero records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_drafts as teacher
from agent_dplmbit.reasoned_response import parse_reasoned_response
from agent_dplmbit.schema import ContractError, PublicExample, ToolObservation
from agent_dplmbit.tools import ToolRegistry

EPISODE_SCHEMA = "agent_dplmbit.teacher_episode.v1"
REVIEW_SCHEMA = "agent_dplmbit.episode_semantic_reviews.v1"
AUDIT_SCHEMA = "agent_dplmbit.episode_mechanical_audit.v1"
LANGUAGE_SCHEMA = "agent_dplmbit.a1_language_fragment.v1"
DECISION_SCHEMA = "agent_dplmbit.a1_decision_weak_labels.v1"
REASONING_FIELDS = {"/reasoning/summary", "/reasoning/uncertainty", "/reasoning/evidence_refs"}
ADDRESS_FIELDS = {"address_binding", "residue_id", "position_1based"}
WEAK_FIELDS = ADDRESS_FIELDS | {"aa_preference", "action"}


def require(condition, message):
    if not condition:
        raise ContractError(message)


def episode_hash(episode):
    return teacher.digest({key: value for key, value in episode.items() if key != "episode_hash"})


def request_row(episode):
    return {"schema": teacher.REQUEST_SCHEMA, "request_id": episode["request_id"],
            "parent_id": episode["parent_id"], "split": episode["split"],
            "public": episode["initial_public"]}


def trajectory_hash(episode):
    return teacher.digest([{"public": turn.get("public_before"), "visible": turn.get("visible_content"),
                            "tool_observation": turn.get("tool_observation")}
                           for turn in episode.get("turns", [])])


def advance_public(public, visible, observation):
    """The sole v1 causal transition: readonly report and two history messages."""
    history = (*public.history,
               {"role": "assistant", "content": visible, "state_hash": public.state.state_hash},
               {"role": "tool", "content": teacher.canonical(observation.to_dict()),
                "state_hash": public.state.state_hash})
    return replace(public, observations=(*public.observations, observation), history=history)


def audit_episode(episode, evidence_map, *, evidence_map_sha256=None, availability_summary=False,
                  unbound_prior_task=False):
    """Return replay receipts; malformed episodes raise and are preserved by run()."""
    require(isinstance(episode, dict) and episode.get("schema") == EPISODE_SCHEMA, "Unsupported episode schema")
    require(episode.get("episode_hash") == episode_hash(episode), "Episode content hash mismatch")
    require(episode.get("approved_for_training") is False and episode.get("semantic_claims_verified") is False,
            "Collector output must remain unapproved, semantically unverified drafts")
    for key in ("parent_id", "request_id"):
        require(isinstance(episode.get(key), str) and bool(episode[key].strip()), "Missing episode " + key)
    require(episode.get("split") in {"train", "validation", "test"}, "Unknown split")
    teacher._private_guard(episode["initial_public"])
    public = PublicExample.from_dict(episode["initial_public"])
    require(public.to_dict() == episode["initial_public"], "Initial public state is not losslessly canonical")
    require(teacher.digest(public.to_dict()) == episode.get("initial_public_sha256"), "Initial public hash mismatch")
    require(not public.history and not public.observations,
            "v1 requires empty initial history/observations; prefilled reports need a separately replayable trace")
    if public.state.phase == "initial":
        require(all(a is None for a in public.state.aa) and all(b is None for b in public.state.bits)
                and (public.state.ca is None or all(x is None for x in public.state.ca)),
                "Initial state cannot contain protein identities, structure bits or coordinates")
    provenance = episode.get("provenance", {})
    require(isinstance(provenance, dict), "Missing episode provenance")
    # Reconstruct the exact collector prompt locally, without HTTP/tokenization.
    # Old episodes need their pinned collector/parser source for replay.
    import collect_episodes as collector
    config = provenance.get("config")
    require(isinstance(config, dict), "Missing exact teacher configuration")
    max_policy, max_tools = provenance.get("max_policy_calls"), provenance.get("max_tool_calls")
    require(type(max_policy) is int and 1 <= max_policy <= 6
            and type(max_tools) is int and 0 <= max_tools <= 4, "Episode budgets differ from collector contract")
    structured_output = provenance.get("structured_output", False)
    require(type(structured_output) is bool, "Invalid constrained decoding declaration")
    declared_availability = provenance.get("availability_summary", False)
    require(type(availability_summary) is bool and type(declared_availability) is bool
            and availability_summary == declared_availability,
            "Availability protocol differs; replay v4 with --availability-summary, v3 without it")
    require(not availability_summary or structured_output, "Availability v4 requires constrained decoding")
    declared_unbound = provenance.get("unbound_prior_task", False)
    require(type(unbound_prior_task) is bool and type(declared_unbound) is bool
            and unbound_prior_task == declared_unbound,
            "Unbound-prior protocol differs; replay v5 with --unbound-prior-task, earlier versions without it")
    require(not unbound_prior_task or (availability_summary and structured_output),
            "Unbound-prior v5 requires availability_summary and structured_output")
    version, system_prompt = collector.prompt_settings(structured_output, availability_summary, unbound_prior_task)
    require(provenance.get("prompt_version") == version
            and provenance.get("system_prompt_sha256") == teacher.digest(system_prompt),
            "Collector prompt provenance differs; replay with the pinned source")
    if structured_output:
        require(provenance.get("constraints_sha256") == teacher.sha_file(Path(__file__).with_name(
                    "episode_constraints_v4.py" if availability_summary else "episode_constraints.py")),
                "Structured grammar source differs; replay with pinned source")
        require(provenance.get("initial_lookup_policy") == "protocol_required_not_autonomous_tool_selection",
                "Mandatory lookup must not be reported as autonomous policy evidence")
    if availability_summary:
        require(provenance.get("constraints_base_sha256") == teacher.sha_file(Path(__file__).with_name("episode_constraints.py")),
                "Base grammar differs; replay with pinned source")
        require(provenance.get("availability_module_sha256") == teacher.sha_file(
                    Path(__file__).resolve().parents[2]/"src/agent_dplmbit/availability.py"),
                "Availability implementation differs; replay with pinned source")
        require(provenance.get("repeated_lookup_policy") == "forbidden_after_actual_same_state_lookup",
                "Missing versioned no-repeat lookup policy")
    require(provenance.get("input_request_sha256") == teacher.digest(request_row(episode)),
            "Input request hash (including parent/split) mismatch")
    require(provenance.get("model") == teacher.MODEL, "Unexpected teacher model")
    expected_evidence_hash = evidence_map_sha256 or teacher.digest(evidence_map)
    require(provenance.get("evidence_map_sha256") == expected_evidence_hash, "Evidence map hash mismatch")
    input_review = provenance.get("input_review")
    require(isinstance(input_review, dict) and input_review.get("ready") is True
            and input_review.get("request_id") == episode["request_id"]
            and input_review.get("public_sha256") == episode["initial_public_sha256"],
            "Missing hash-bound ready input review")
    require(type(input_review.get("allow_bound_positions")) is bool, "Missing reviewed address permission")
    require(teacher.valid_sha(input_review.get("review_sha256")) and bool(input_review.get("reviewer")),
            "Input readiness provenance is incomplete")
    teacher._private_guard(evidence_map)
    registry = ToolRegistry(reaction_evidence=evidence_map, max_items=64)
    turns = episode.get("turns")
    require(isinstance(turns, list) and bool(turns), "Empty episode")
    receipts, final_seen, tool_calls = [], False, 0
    require(len(turns) <= max_policy, "Episode exceeded policy-call budget")
    fixed_state, fixed_request = public.state.to_dict(), public.request.to_dict()
    for index, turn in enumerate(turns):
        require(not final_seen, "Turns appear after a final condition")
        require(isinstance(turn, dict) and turn.get("index") == index, "Turn index sequence differs")
        teacher._private_guard(turn.get("public_before"))
        require(turn.get("public_before") == public.to_dict(), "Noncausal public observations/history or state mutation")
        require(turn.get("public_sha256") == teacher.digest(public.to_dict()), "Per-turn public hash mismatch")
        require(public.state.to_dict() == fixed_state and public.request.to_dict() == fixed_request,
                "Readonly episode mutated state or request")
        require(turn.get("state_hash_before") == public.state.state_hash
                and turn.get("state_hash_after") == public.state.state_hash, "Readonly state hash changed")
        request_sha = teacher.digest(public.request.to_dict())
        require(turn.get("request_sha256_before") == request_sha and turn.get("request_sha256_after") == request_sha,
                "Readonly request hash changed")
        if turn.get("error") is not None:
            raise ContractError("Turn has a recorded error; retain for diagnostics only: " + str(turn["error"])[:500])
        raw = turn.get("raw_response")
        require(isinstance(raw, dict) and raw.get("model") == teacher.MODEL, "Wrong/missing raw response model")
        choices = raw.get("choices")
        require(isinstance(choices, list) and len(choices) == 1, "Require exactly one raw completion")
        choice = choices[0]
        require(choice.get("finish_reason") == "stop", "Truncated/abnormal response is not accepted")
        message = choice.get("message", {})
        visible = turn.get("visible_content")
        require(isinstance(visible, str) and visible == message.get("content"), "Visible text differs from raw content")
        response = parse_reasoned_response(visible, public, validate_tool_arguments=False)
        if structured_output:
            _, checked_wire = collector.checked_response(raw, public, input_review, structured_output=True,
                                                          availability_summary=availability_summary)
            require(checked_wire.final == response.final, "Constrained wire validation differs")
        teacher.validate_reference_menu(public, response.reasoning["evidence_refs"])
        require(turn.get("validated_response") == {"reasoning": response.reasoning, "final": response.final},
                "Stored validation differs from actual parsed response")
        require(turn.get("allowed_evidence_refs") == teacher.allowed_evidence_refs(public), "Evidence menu mismatch")
        allowed_tools = turn.get("allowed_tool_names")
        require(allowed_tools == list(collector.allowed_tool_names(public, availability_summary=availability_summary)),
                "Tool menu invalid")
        if availability_summary:
            from agent_dplmbit.availability import state_availability_summary
            require(turn.get("state_availability_summary") == state_availability_summary(public.state),
                    "Availability summary differs from the current PublicState; bits do not provide CA")
        req = teacher.Request(episode["request_id"], episode["parent_id"], episode["split"],
                              public, teacher.digest(public.to_dict()))
        messages = collector.messages_for(req, public, input_review, max_policy-index, max_tools-tool_calls,
                                         structured_output=structured_output, availability_summary=availability_summary,
                                         unbound_prior_task=unbound_prior_task)
        prompt_sha = teacher.context_prompt_hash(messages, config)
        require(turn.get("prompt_sha256") == prompt_sha, "Prompt differs from replayed public evidence/menu/budget")
        context = turn.get("context_audit", {})
        require(isinstance(context, dict) and context.get("verified") is True and context.get("fits") is True
                and context.get("prompt_sha256") == prompt_sha and teacher.valid_sha(context.get("tokenizer_sha256")),
                "Teacher context was not tokenizer-audited")
        for key in ("input_tokens", "max_output_tokens", "context_length", "safety_margin"):
            require(type(context.get(key)) is int and context[key] >= 0, "Malformed context token budget")
        require(context["input_tokens"] + context["max_output_tokens"] + context["safety_margin"] <= context["context_length"],
                "Teacher context budget exceeded")
        require(context["max_output_tokens"] == config.get("max_tokens")
                and context["context_length"] == config.get("context_length")
                and context["safety_margin"] == config.get("context_safety_margin")
                and context.get("chat_template_kwargs") == teacher.chat_template_kwargs(config),
                "Context audit differs from teacher configuration")
        payload = collector.request_payload(messages, public, input_review, config,
                                            structured_output=structured_output, availability_summary=availability_summary)
        require(turn.get("request_payload_sha256") == teacher.digest(payload), "HTTP payload hash differs from replay")
        first = public.state.phase == "initial" or (all(a is None for a in public.state.aa)
                                                   and all(b is None for b in public.state.bits))
        receipt = {"index": index, "public_sha256": turn["public_sha256"],
                   "visible_sha256": teacher.digest(visible), "response_type": response.final["type"],
                   "first_round_without_protein": first, "tool_replay_matched": None, "tool_error_observed": False}
        if response.tool_request is not None:
            call = response.tool_request
            tool_calls += 1
            require(tool_calls <= max_tools and turn.get("tool_executed") is True, "Unexecuted/overbudget tool call")
            require(call.name in allowed_tools, "Call outside allowed tool menu")
            require(turn.get("tool_request") == call.to_dict(), "Tool request differs from final payload")
            try:
                observation = registry.execute(call, public)
            except ContractError as exc:
                observation = ToolObservation(call.name, public.state.state_hash,
                    {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                    "actual_tool_validation_error")
                receipt["tool_error_observed"] = True
            require(turn.get("tool_observation") == observation.to_dict(), "Forged/altered tool observation differs from replay")
            teacher._private_guard(observation.to_dict())
            receipt["tool_replay_matched"] = True
            public = advance_public(public, visible, observation)
        else:
            require(turn.get("tool_request") is None and turn.get("tool_observation") is None,
                    "Final condition must not fabricate a tool report")
            require(turn.get("tool_executed") is False, "Final condition cannot execute a tool")
            hypotheses = response.final["hypotheses"]
            require(len(hypotheses) <= 3 and all(h["action"] in {"NONE", "PREFER"} for h in hypotheses),
                    "Final soft-action budget/vocabulary differs")
            if first or not input_review["allow_bound_positions"]:
                require(all(h["address_binding"] == "pending" for h in hypotheses), "First-round/unreviewed address binding")
            final_seen = True
        receipts.append(receipt)
    require(episode.get("status") == "completed" and final_seen, "Incomplete/error episode remains diagnostic only")
    return {"episode_hash": episode["episode_hash"], "request_id": episode["request_id"],
            "parent_id": episode["parent_id"], "split": episode["split"],
            "mechanically_valid": True, "semantic_claims_verified": False,
            "turns": receipts, "scope": "hash, causality, readonly replay and protocol only; no biological truth attestation"}


def read_semantic_reviews(path):
    if path is None:
        return {}, None
    document = json.loads(Path(path).read_text())
    require(isinstance(document, dict) and set(document) == {"schema", "reviews"}
            and document["schema"] == REVIEW_SCHEMA and isinstance(document["reviews"], list),
            "Invalid external semantic review manifest")
    result = {}
    for review in document["reviews"]:
        require(isinstance(review, dict) and teacher.valid_sha(review.get("episode_hash")), "Semantic review lacks episode hash")
        require(review["episode_hash"] not in result, "Duplicate semantic review")
        result[review["episode_hash"]] = review
    return result, teacher.sha_file(path)


def reviewed_fragments(episode, audit, review):
    """Explicit field-level supervision, with no numeric-strength pseudo labels."""
    if review is None or review.get("decision") != "approve":
        return [], []
    for key in ("episode_hash", "request_id", "parent_id", "split", "initial_public_sha256"):
        require(review.get(key) == episode[key], "Semantic approval has mismatched " + key)
    require(review.get("approved_for_training") is True and review.get("semantic_claims_verified") is True,
            "External approval must explicitly approve training fragments and semantic claims")
    for key in ("reviewer", "reviewed_at", "notes"):
        require(isinstance(review.get(key), str) and bool(review[key].strip()), "External approval lacks " + key)
    require(isinstance(review.get("turns"), list), "Review needs explicit per-turn approvals")
    language, decisions, seen = [], [], set()
    for turn_review in review["turns"]:
        i = turn_review.get("index")
        require(type(i) is int and 0 <= i < len(episode["turns"]) and i not in seen, "Invalid/duplicate reviewed turn")
        seen.add(i)
        turn, checked = episode["turns"][i], audit["turns"][i]
        require(turn_review.get("public_sha256") == checked["public_sha256"]
                and turn_review.get("visible_sha256") == checked["visible_sha256"], "Stale per-turn semantic approval")
        if turn_review.get("decision") != "approve":
            continue
        fields = turn_review.get("approved_fields")
        require(isinstance(fields, list) and len(fields) == len(set(fields))
                and all(isinstance(field, str) for field in fields), "Invalid approved field paths")
        approved = set(fields)
        response = turn["validated_response"]
        final = response["final"]
        allowed = set(REASONING_FIELDS)
        if final["type"] == "tool_call" and not checked["tool_error_observed"]:
            allowed.add("/final/tool_call")
        elif final["type"] == "condition":
            allowed |= {f"/final/hypotheses/{j}/{name}" for j in range(len(final["hypotheses"])) for name in WEAK_FIELDS}
        require(approved <= allowed, "Uncalibrated strength/distribution or unknown field cannot be supervised")
        base = {"episode_hash": episode["episode_hash"], "request_id": episode["request_id"],
                "parent_id": episode["parent_id"], "split": episode["split"], "turn_index": i,
                "public_sha256": checked["public_sha256"], "public_before": turn["public_before"],
                "review_sha256": teacher.digest(review), "reviewer": review["reviewer"],
                "generator_prefix": {"source": "public_before_only", "current_teacher_final_included": False,
                                     "current_teacher_reasoning_included": False},
                "format_status": "new_interchange_schema_not_existing_trainer_input"}
        fragments = {path: response["reasoning"][path.rsplit("/", 1)[1]]
                     for path in sorted(approved & REASONING_FIELDS)}
        if "/final/tool_call" in approved:
            fragments["/final/tool_call"] = final
        if fragments:
            language.append({**base, "schema": LANGUAGE_SCHEMA, "approved_fragments": fragments,
                             "loss_scope": "only listed field values/protocol fragment; no tool result or unreviewed response loss"})
        weak = []
        for j, hypothesis in enumerate(final.get("hypotheses", [])):
            prefix = f"/final/hypotheses/{j}/"
            selected = {name for name in WEAK_FIELDS if prefix + name in approved}
            if selected & ADDRESS_FIELDS:
                require(ADDRESS_FIELDS <= selected, "Address approval must bind all address fields together")
                require(not checked["first_round_without_protein"], "Never supervise first-round address fields, including nulls")
            if selected:
                weak.append({"hypothesis_index": j, "role_id": hypothesis["role_id"],
                             "approved_fields": {name: hypothesis[name] for name in sorted(selected)}})
        if weak:
            decisions.append({**base, "schema": DECISION_SCHEMA, "weak_labels": weak,
                              "label_status": "reviewed_teacher_proposals_not_expert_or_outcome_verified",
                              "strength_supervision": False, "biological_efficacy_verified": False})
    return language, decisions


def write_once(path, text):
    if path.exists():
        require(path.read_text() == text, "Refusing to overwrite different prior review/export: " + str(path))
    else:
        with path.open("x") as stream:
            stream.write(text)


def run(episodes_path, evidence_path, output_dir, semantic_reviews_path=None, *, availability_summary=False,
        unbound_prior_task=False):
    evidence = json.loads(Path(evidence_path).read_text())
    evidence_sha = teacher.sha_file(evidence_path)
    reviews, review_file_sha = read_semantic_reviews(semantic_reviews_path)
    episodes = [json.loads(line) for line in Path(episodes_path).read_text().splitlines() if line.strip()]
    require(bool(episodes), "No episodes to audit")
    unknown_reviews = sorted(set(reviews) - {episode.get("episode_hash") for episode in episodes})
    audit_rows, language, decisions = [], [], []
    parent_splits, request_splits = defaultdict(set), defaultdict(set)
    for episode in episodes:
        parent_splits[episode.get("parent_id")].add(episode.get("split"))
        request_splits[episode.get("request_id")].add(episode.get("split"))
    seen, exported_trajectories = set(), set()
    for line, episode in enumerate(episodes, 1):
        report = {"line": line, "episode_hash": episode.get("episode_hash"),
                  "request_id": episode.get("request_id"), "parent_id": episode.get("parent_id"),
                  "split": episode.get("split"), "mechanically_valid": False,
                  "semantic_claims_verified": False, "errors": [], "language_exported": 0, "decision_exported": 0}
        try:
            require(len(parent_splits[episode.get("parent_id")]) == 1
                    and len(request_splits[episode.get("request_id")]) == 1, "Parent/request appears in multiple splits")
            require(episode.get("episode_hash") not in seen, "Duplicate exact episode; no duplicate supervision export")
            seen.add(episode.get("episode_hash"))
            audit = audit_episode(episode, evidence, evidence_map_sha256=evidence_sha,
                                  availability_summary=availability_summary, unbound_prior_task=unbound_prior_task)
            report.update(audit)
            review = reviews.get(episode["episode_hash"])
            # An unrelated review file is a configuration error, never a silent
            # partial approval. Still audit every source record and preserve it.
            lrows, drows = ([], []) if unknown_reviews else reviewed_fragments(episode, audit, review)
            signature = trajectory_hash(episode)
            if (lrows or drows) and signature in exported_trajectories:
                lrows, drows = [], []
                report["duplicate_trajectory_export_skipped"] = True
            elif lrows or drows:
                exported_trajectories.add(signature)
            language.extend(lrows)
            decisions.extend(drows)
            report.update(language_exported=len(lrows), decision_exported=len(drows),
                          semantic_review_status="absent" if review is None else review.get("decision"),
                          approved_fragment_semantics_reviewed=bool(lrows or drows))
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        audit_rows.append(report)
    stats = {"schema": AUDIT_SCHEMA, "source_episodes_sha256": teacher.sha_file(episodes_path),
             "evidence_map_sha256": evidence_sha, "semantic_reviews_sha256": review_file_sha,
             "reviewer_code_sha256": teacher.sha_file(__file__), "episode_rows": len(episodes),
             "unique_episode_hashes": len({episode.get("episode_hash") for episode in episodes}),
             "unique_parent_ids": len(parent_splits), "unique_request_ids": len(request_splits),
             "unique_requests": len({teacher.digest(e.get("initial_public", {}).get("request")) for e in episodes}),
             "unique_trajectories": len({trajectory_hash(episode) for episode in episodes}),
             "split_counts": dict(Counter(e.get("split") for e in episodes)),
             "mechanically_valid_episodes": sum(r["mechanically_valid"] and not r["errors"] for r in audit_rows),
             "episodes_with_errors": sum(bool(r["errors"]) for r in audit_rows),
             "a1_language_fragments": len(language), "a1_decision_records": len(decisions),
             "unknown_semantic_review_hashes": unknown_reviews,
             "status": "blocked_unknown_semantic_reviews" if unknown_reviews else "reviewed_fragments_exported" if language or decisions else (
                 "blocked_no_semantic_reviews" if semantic_reviews_path is None else "blocked_no_approved_fragments"),
             "automatic_semantic_approval": False, "training_launched": False,
             "limits": ["Hashes establish consistency, not the authenticity of an external reviewer or source.",
                        "Public prose can conceal leakage; semantic source review remains mandatory.",
                        "Reviewed weak proposals are not expert action labels or demonstrated generation gains.",
                        "Raw response/recorded errors stay in source and audit; never become supervised content."]}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("audit.jsonl", audit_rows), ("a1_language.jsonl", language), ("a1_decision_weak.jsonl", decisions)):
        write_once(out / name, "".join(teacher.canonical(row) + "\n" for row in rows))
    write_once(out / "summary.json", json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--semantic-reviews")
    parser.add_argument("--availability-summary", action="store_true",
                        help="Require and replay opt-in v4 availability/no-repeat-lookup episodes")
    parser.add_argument("--unbound-prior-task", action="store_true",
                        help="Require and replay opt-in v5 source-scoped unbound-prior episodes")
    args = parser.parse_args()
    result = run(args.episodes, args.evidence_map, args.output_dir, args.semantic_reviews,
                 availability_summary=args.availability_summary, unbound_prior_task=args.unbound_prior_task)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "reviewed_fragments_exported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
