#!/usr/bin/env python3
"""Offline teacher/tool episodes on one immutable public protein snapshot.

No generator rollout, training, automatic semantic approval, or fabricated tool
observations. Completed and invalid episodes are never silently resampled.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("episode_teacher_client", Path(__file__).with_name("generate_drafts.py"))
g = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)
from agent_dplmbit.reasoned_response import parse_reasoned_response
from agent_dplmbit.schema import ContractError, PublicExample, ToolObservation
from agent_dplmbit.tools import ToolRegistry

SCHEMA = "agent_dplmbit.teacher_episode.v1"
REVIEW_SCHEMA = "agent_dplmbit.teacher_episode_input_review.v1"
PROMPT_VERSION = "offline_fixed_state_tool_episodes_v1_20261003"
ALLOWED_TOOLS = ("lookup_reaction_evidence", "get_current_state", "inspect_current_geometry",
                 "get_dplm_proposals", "inspect_candidate_evidence")
TERMINAL = {"completed", "invalid", "budget_exhausted", "request_failed"}
SYSTEM_PROMPT = g.SYSTEM_PROMPT.replace(
    "strength in [0,1] is optional.", "Omit strength: no calibrated strength label is available."
).replace(
    "For this offline drafting task, do not invoke tools or simulate tool results; reason only from supplied observations.",
    """This is an offline tool episode on a fixed protein snapshot. Each final is either one tool_call or one condition.
A tool call has exactly: {"type":"tool_call","name":"ALLOWED_TOOL","arguments":{},"state_hash":"CURRENT_HASH"}.
Use tools only from available_tools with their documented arguments. Tools execute externally; never simulate their output.
Before a chemistry lookup, do not turn raw SMILES/EC into an asserted reaction name, mechanism or catalytic identity.
If chemical facts are insufficient, briefly say what is missing and request lookup_reaction_evidence.
After lookup, use only the returned, source-scoped names/equations/roles. No role evidence means no justified AA claim.
Missing predictions or geometry must remain unavailable. A valid editable slot is not evidence of catalytic suitability.
Observe remaining budgets; when no useful allowed query remains, end with a condition, possibly empty.
Only write a concise checkable summary, not an extended internal thought process. Conditions are unverified proposals."""
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_reviews(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {"schema", "requests"} or data["schema"] != REVIEW_SCHEMA:
        raise ContractError("A separate episode input review manifest is required")
    if not isinstance(data["requests"], list):
        raise ContractError("Episode review requests must be a list")
    fields = {"request_id", "public_sha256", "ready", "reviewer", "review_sha256", "allow_bound_positions", "reason"}
    reviews = {}
    for row in data["requests"]:
        if (not isinstance(row, dict) or set(row)-fields or not {"request_id", "ready"} <= set(row)
                or not isinstance(row["request_id"], str) or not row["request_id"] or row["request_id"] in reviews
                or type(row["ready"]) is not bool):
            raise ContractError("Malformed or duplicate episode input review")
        if row["ready"] and (not g.valid_sha(row.get("public_sha256")) or not g.valid_sha(row.get("review_sha256"))
                or not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip()
                or type(row.get("allow_bound_positions")) is not bool):
            raise ContractError("Ready episode inputs need bound hashes, reviewer and address policy")
        reviews[row["request_id"]] = row
    return reviews


def prompt_settings(structured_output=False, availability_summary=False, unbound_prior_task=False):
    if type(unbound_prior_task) is not bool:
        raise ContractError("unbound_prior_task must be boolean")
    if unbound_prior_task:
        if not (structured_output and availability_summary):
            raise ContractError("Unbound-prior v5 requires structured_output and availability_summary")
        _, previous = prompt_settings(True, True)
        return "offline_fixed_state_tool_episodes_v5_unbound_prior_20261003", previous + """
This task distinguishes source-supported, unbound AA chemical priors from
verified positions, mechanisms and generation benefits. After the actual
lookup, assess relevant role evidence for weak pending candidates. No observed
protein or validated position is not, by itself, a reason to reject an unbound
candidate. Keep residue_id and position_1based null whenever binding is forbidden.
EC_only evidence supports only a weak family hypothesis, not the requested
reaction's established mechanism. A family hypothesis need not be certified to
be proposed with its scope and uncertainty. Source mechanism star_rating is
not certification of an individual AA role. Select candidates from this actual
source and its applicability; do not copy a fixed AA list across requests.
Selecting fewer candidates does not mean the source contains only that many.
The required pre-lookup check does not mean SMILES lack chemical information
or reactant identities; do not describe a supplied reaction as only an EC label.
Use reviewed request facts for the fixed requested Rhea direction. A source's
reverse-direction role limitations do not make the requested direction unknown.
State actual conflicts separately. Cite supporting observations for each proposal
and distinguish a chemical prior from any claim of positional or functional gain.
Empty hypotheses remain allowed for specific missing, conflicting or inapplicable
role evidence; explain that reason. Do not force nonempty output or upgrade a
weak candidate into a verified action, calibrated probability or strength label.
"""
    if availability_summary:
        if not structured_output:
            raise ContractError("Availability-summary v4 requires structured_output")
        _, previous = prompt_settings(True)
        return "offline_fixed_state_tool_episodes_v4_availability_20261003", previous + """
Use state_availability_summary, computed only from this current PublicState, as
the authoritative count of available AA identities, structure-bit rows and CA
coordinates. Structure bits are not decoded coordinates: visible bits do not
establish any CA position, distance, contact or side-chain geometry. A phase
name does not establish geometric availability. Do not describe partial bits
as partial CA geometry when the CA count is zero.
The reaction_evidence_lookup_completed flag records an actual lookup observation
for this exact state. Once true, the lookup result is already in observations;
do not ask for the same lookup again or claim it has not been obtained. Its
result may be unavailable/error; that still counts as queried, not as evidence.
After lookup, choose only the remaining tools or finish with supported soft
hypotheses. Empty hypotheses remain valid; no AA preference is forced.
"""
    if not structured_output:
        return PROMPT_VERSION, SYSTEM_PROMPT
    return "offline_fixed_state_tool_episodes_v3_structural_tag_20261003", SYSTEM_PROMPT + """
The decoding grammar requires a reaction-evidence lookup before a final condition.
This is a prescribed grounding check, not evidence of autonomous tool selection.
Before the lookup, summarize only the need to retrieve evidence and the observed
state availability. Do not interpret or name reactants, products, functional
groups or mechanisms from the SMILES. A length and empty slots are a design
canvas, not an already existing protein scaffold. After lookup, distinguish
available reaction facts from unavailable role evidence, and quote their scope.
Use the actual observed counts; a partially visible state is not an all-null state.
Candidate role_id values are protocol slots candidate_1, candidate_2, candidate_3
in array order; they do not assert a chemical role. Explain chemical scope only
in the evidence-grounded summary. Do not emit strength or probability fields.
"""


def allowed_tool_names(public, *, availability_summary=False):
    if availability_summary:
        from episode_constraints_v4 import allowed_tool_names as names
        return names(public)
    return ALLOWED_TOOLS


def messages_for(request, public, review, remaining_policy_calls, remaining_tool_calls, *, structured_output=False,
                 availability_summary=False, unbound_prior_task=False):
    current = g.Request(request.request_id, request.parent_id, request.split, public, g.digest(public.to_dict()))
    messages = g.messages_for(current, review)
    payload = json.loads(messages[1]["content"])
    payload["constraints"].update(allowed_final_types=["tool_call", "condition"],
                                  remaining_policy_calls=remaining_policy_calls,
                                  remaining_tool_calls=remaining_tool_calls, protein_state_is_fixed=True)
    payload["available_tools"] = [
        {"name": "lookup_reaction_evidence", "arguments": {}, "scope": "Read approved source-scoped reaction evidence; may be unavailable."},
        {"name": "get_current_state", "arguments": {"slots": "Optional list of <=64 one-based positions or stable IDs; defaults first <=64."},
         "scope": "Read observed identities and availability, not catalytic suitability."},
        {"name": "inspect_current_geometry", "arguments": {"pairs": "Required list of <=64 [slot,slot] pairs."},
         "scope": "Compute available current CA distances only, never side-chain catalysis."},
        {"name": "get_dplm_proposals", "arguments": {}, "scope": "Returns unavailable unless a current prediction report is supplied; never runs a generator."},
        {"name": "inspect_candidate_evidence", "arguments": {"slots": "Optional list of <=64 one-based positions or stable IDs."},
         "scope": "Read current slot evidence only, not an action verifier."},
    ]
    if availability_summary:
        from agent_dplmbit.availability import state_availability_summary
        from episode_constraints_v4 import lookup_completed
        payload["state_availability_summary"] = state_availability_summary(public.state)
        payload["constraints"]["reaction_evidence_lookup_completed"] = lookup_completed(public)
        allowed = set(allowed_tool_names(public, availability_summary=True))
        payload["available_tools"] = [tool for tool in payload["available_tools"] if tool["name"] in allowed]
    g._private_guard(payload)
    return [{"role": "system", "content": prompt_settings(structured_output, availability_summary, unbound_prior_task)[1]},
            {"role": "user", "content": g.canonical(payload)}]


def request_payload(messages, public, review, config, *, structured_output=False, availability_summary=False):
    if availability_summary and not structured_output:
        raise ContractError("Availability-summary v4 requires structured_output")
    payload = {"model": g.MODEL, "messages": messages, "stream": False, "n": 1,
               "max_tokens": config["max_tokens"], "temperature": config["temperature"],
               "top_p": config["top_p"], "seed": config["seed"],
               "chat_template_kwargs": g.chat_template_kwargs(config)}
    if structured_output:
        if availability_summary:
            from episode_constraints_v4 import make_response_format
        else:
            from episode_constraints import make_response_format
        payload["response_format"] = make_response_format(public, review)
    return payload


def checked_response(raw, public, review, *, structured_output=False, availability_summary=False):
    if availability_summary and not structured_output:
        raise ContractError("Availability-summary v4 requires structured_output")
    if not isinstance(raw, dict) or raw.get("model") != g.MODEL:
        raise ContractError("Response does not identify the required teacher model")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ContractError("Expected exactly one completion")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ContractError("Response did not finish normally; no semantic resampling")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ContractError("Visible content is required; internal reasoning is not an answer")
    checked = parse_reasoned_response(message["content"], public, validate_tool_arguments=False)
    g.validate_reference_menu(public, checked.reasoning["evidence_refs"])
    final = checked.final
    if structured_output:
        has_lookup = any(o.tool == "lookup_reaction_evidence" for o in public.observations)
        if availability_summary:
            from episode_constraints_v4 import lookup_completed
            has_lookup = lookup_completed(public)
            if has_lookup and final["type"] == "tool_call" and final.get("name") == "lookup_reaction_evidence":
                raise ContractError("Availability-summary v4 forbids repeated reaction lookup on the same fixed state")
        if not has_lookup and (final["type"] != "tool_call"
                or final.get("name") != "lookup_reaction_evidence" or final.get("arguments") != {}):
            raise ContractError("Constrained protocol requires the actual grounding lookup before termination")
        for index, hypothesis in enumerate(final.get("hypotheses", []), 1):
            if set(hypothesis) & {"strength", "aa_probabilities", "action_probabilities", "candidate_probabilities"}:
                raise ContractError("Constrained protocol forbids uncalibrated numeric/distribution labels")
            if hypothesis["role_id"] != f"candidate_{index}":
                raise ContractError("Constrained protocol requires canonical unique candidate slot IDs")
    if final["type"] == "tool_call":
        if final["name"] not in allowed_tool_names(public, availability_summary=availability_summary):
            raise ContractError("Tool is not allowed in fixed-state teacher episodes")
        g._private_guard(final["arguments"])
    else:
        hypotheses = final["hypotheses"]
        if len(hypotheses) > 3 or any(h["action"] not in {"NONE", "PREFER"} for h in hypotheses):
            raise ContractError("Only <=3 NONE/PREFER soft hypotheses are executable")
        if not review["allow_bound_positions"] and any(h["address_binding"] != "pending" for h in hypotheses):
            raise ContractError("Input review forbids bound positions")
    return message["content"], checked


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(g.canonical(value)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+".tmp")
    with temp.open("w") as stream:
        stream.write(g.canonical(value)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_hash(row, key):
    if row.get(key) != g.digest({k: v for k, v in row.items() if k != key}):
        raise ContractError(f"Corrupt {key}; refusing ambiguous resume")


class LoggedClient:
    """Reuse the old retry implementation while persisting every raw attempt."""
    def __init__(self, client, path, episode_key, index):
        self.client, self.path, self.episode_key, self.index = client, path, episode_key, index
        self.attempts, self.last_raw = 0, None

    async def post(self, url, *, json):
        self.attempts += 1
        identity = {"episode_key": self.episode_key, "turn_index": self.index,
                    "attempt": self.attempts, "time_utc": now()}
        append_json(self.path, {**identity, "event": "request", "url": url, "payload": json})
        try:
            response = await self.client.post(url, json=json)
        except Exception as exc:
            append_json(self.path, {**identity, "event": "transport_error", "error_type": type(exc).__name__,
                                   "error": str(exc), "recorded_utc": now()})
            raise
        try:
            self.last_raw = response.json()
        except ValueError:
            self.last_raw = None
        append_json(self.path, {**identity, "event": "response", "recorded_utc": now(),
                               "status_code": response.status_code, "body": response.text,
                               "content_type": response.headers.get("content-type")})
        return response


def frozen_hashes(public):
    return g.digest(public.request.to_dict()), g.digest(public.state.to_dict())


def advance(public, visible, observation):
    h = public.state.state_hash
    history = public.history + ({"role": "assistant", "content": visible, "state_hash": h},
                                {"role": "tool", "content": g.canonical(observation.to_dict()), "state_hash": h})
    result = PublicExample(public.request, public.state, public.observations+(observation,), history)
    g._private_guard(result.to_dict())
    return result


async def collect_one(request, review, registry, client, config, counter, limiter, output, workdir,
                      binding, provenance, max_policy_calls, max_tool_calls, structured_output=False,
                      availability_summary=False, unbound_prior_task=False):
    key = g.digest({"binding": binding, "request_id": request.request_id})
    checkpoint = workdir/(key+".json")
    public = request.public
    invariant = frozen_hashes(public)
    record = {"schema": SCHEMA, "episode_key": key, "request_id": request.request_id,
              "parent_id": request.parent_id, "split": request.split,
              "initial_public_sha256": request.public_sha256, "initial_public": public.to_dict(),
              "turns": [], "status": "running", "approved_for_training": False,
              "semantic_claims_verified": False, "created_utc": now(),
              "provenance": {**provenance, "binding_sha256": binding, "reviewer": review["reviewer"],
                             "review_sha256": review["review_sha256"], "input_review": review,
                             "input_request_sha256": g.digest({"schema": g.REQUEST_SCHEMA,
                                 "request_id": request.request_id, "parent_id": request.parent_id,
                                 "split": request.split, "public": request.public.to_dict()})}}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text()); verify_hash(saved, "checkpoint_hash")
        if saved["binding_sha256"] != binding or saved["record"]["episode_key"] != key:
            raise ContractError("Checkpoint belongs to different inputs/configuration/evidence")
        record, public = saved["record"], PublicExample.from_dict(saved["current_public"])
        if frozen_hashes(public) != invariant:
            raise ContractError("Checkpoint changed immutable state/request")
        if saved["inflight"]:
            record.update(status="request_failed", error="Interrupted inflight request/tool; no automatic resampling or re-execution")

    def save(inflight=False):
        saved = {"binding_sha256": binding, "record": record, "current_public": public.to_dict(), "inflight": inflight}
        saved["checkpoint_hash"] = g.digest(saved)
        atomic_json(checkpoint, saved)

    tool_calls = sum(bool(t.get("tool_executed")) for t in record["turns"])
    while record["status"] == "running" and len(record["turns"]) < max_policy_calls:
        index = len(record["turns"])
        if frozen_hashes(public) != invariant:
            record.update(status="invalid", error="Immutable request/state changed before policy call")
            break
        messages = messages_for(request, public, review, max_policy_calls-index, max_tool_calls-tool_calls,
                                structured_output=structured_output, availability_summary=availability_summary,
                                unbound_prior_task=unbound_prior_task)
        context = g.audit_context(messages, {}, config, counter)
        turn = {"index": index, "public_before": public.to_dict(), "public_sha256": g.digest(public.to_dict()),
                "state_hash_before": public.state.state_hash, "state_hash_after": public.state.state_hash,
                "request_sha256_before": invariant[0], "request_sha256_after": invariant[0],
                "prompt_sha256": context["prompt_sha256"], "context_audit": context,
                "allowed_tool_names": list(allowed_tool_names(public, availability_summary=availability_summary)), "allowed_evidence_refs": g.allowed_evidence_refs(public),
                "visible_content": None, "raw_response": None, "tool_executed": False}
        if availability_summary:
            from agent_dplmbit.availability import state_availability_summary
            turn["state_availability_summary"] = state_availability_summary(public.state)
        record["turns"].append(turn)
        if not context.get("verified") or not context.get("fits"):
            turn["error"] = "Exact per-turn context unavailable or exceeds budget; no truncation"
            record["status"] = "budget_exhausted"
            break
        payload = request_payload(messages, public, review, config, structured_output=structured_output,
                                  availability_summary=availability_summary)
        turn["request_payload_sha256"] = g.digest(payload)
        save(inflight=True)
        logged = LoggedClient(client, output.with_suffix(output.suffix+".http.jsonl"), key, index)
        try:
            raw, attempts = await g.completion(logged, payload, config, limiter)
            turn.update(raw_response=raw, attempts=attempts)
        except Exception as exc:
            turn.update(raw_response=logged.last_raw, attempts=logged.attempts,
                        error=f"{type(exc).__name__}: {exc}")
            record["status"] = "request_failed"
            save()
            break
        try:
            # Preserve visible content even if evidence/schema validation fails.
            choices = raw.get("choices", []) if isinstance(raw, dict) else []
            message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
            turn["visible_content"] = message.get("content") if isinstance(message, dict) else None
            visible, checked = checked_response(raw, public, review, structured_output=structured_output,
                                                availability_summary=availability_summary)
            turn["validated_response"] = {"reasoning": checked.reasoning, "final": checked.final}
            if checked.final["type"] == "condition":
                record["status"] = "completed"
            else:
                call = checked.tool_request
                turn["tool_request"] = call.to_dict()
                if tool_calls >= max_tool_calls:
                    turn["error"] = "Global tool-call budget exhausted; tool was not executed"
                    record["status"] = "budget_exhausted"
                else:
                    turn["tool_executed"] = True
                    tool_calls += 1
                    try:
                        observation = registry.execute(call, public)
                    except ContractError as exc:
                        observation = ToolObservation(call.name, public.state.state_hash,
                            {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                            "actual_tool_validation_error")
                    turn["state_hash_after"] = public.state.state_hash
                    turn["request_sha256_after"] = g.digest(public.request.to_dict())
                    if frozen_hashes(public) != invariant:
                        raise ContractError("Tool mutated immutable protein state/request")
                    if not isinstance(observation, ToolObservation) or observation.tool != call.name:
                        raise ContractError("Registry did not return the requested typed observation")
                    g._private_guard(observation.to_dict())
                    turn["tool_observation"] = observation.to_dict()
                    public = advance(public, visible, observation)
        except Exception as exc:
            turn["error"] = f"{type(exc).__name__}: {exc}"
            record["status"] = "invalid"
        save()
    if record["status"] == "running":
        record["status"] = "budget_exhausted"
    record.update(finished_utc=now(), policy_calls=sum("request_payload_sha256" in t for t in record["turns"]),
                  recorded_turns=len(record["turns"]), tool_calls=tool_calls,
                  http_attempts=sum(t.get("attempts", 0) for t in record["turns"]))
    record["episode_hash"] = g.digest({k: v for k, v in record.items() if k != "episode_hash"})
    save()
    append_json(output, record)
    return record


async def run(config_path, requests_path, reviews_path, evidence_path, output_path, *,
              max_policy_calls=6, max_tool_calls=4, dry_run=False, transport=None, token_counter=None,
              registry_factory=ToolRegistry, structured_output=False, availability_summary=False,
              unbound_prior_task=False):
    import httpx
    if type(max_policy_calls) is not int or not 1 <= max_policy_calls <= 6 or type(max_tool_calls) is not int or not 0 <= max_tool_calls <= 4:
        raise ContractError("Global budget must be 1..6 policy calls and 0..4 tool calls")
    config, requests, reviews = g.read_config(config_path), g.read_requests(requests_path), read_reviews(reviews_path)
    evidence = json.loads(Path(evidence_path).read_text()); g._private_guard(evidence)
    registry = registry_factory(reaction_evidence=evidence, max_items=64)
    if type(structured_output) is not bool:
        raise ContractError("structured_output must be boolean")
    if type(availability_summary) is not bool or (availability_summary and not structured_output):
        raise ContractError("availability_summary must be boolean and requires structured_output when enabled")
    if structured_output and (max_policy_calls < 2 or max_tool_calls < 1):
        raise ContractError("Constrained protocol needs at least one lookup and a subsequent policy call")
    version, system_prompt = prompt_settings(structured_output, availability_summary, unbound_prior_task)
    provenance = {"scope": "offline_teacher_tools_fixed_public_state_unapproved_not_student_on_policy_not_expert_labels",
                  "model": g.MODEL, "prompt_version": version, "system_prompt_sha256": g.digest(system_prompt),
                  "config": config, "requests_sha256": g.sha_file(requests_path),
                  "review_manifest_sha256": g.sha_file(reviews_path), "evidence_map_sha256": g.sha_file(evidence_path),
                  "collector_sha256": g.sha_file(__file__), "client_sha256": g.sha_file(g.__file__),
                  "parser_sha256": g.sha_file(Path(__file__).resolve().parents[2]/"src/agent_dplmbit/reasoned_response.py"),
                  "tools_sha256": g.sha_file(Path(__file__).resolve().parents[2]/"src/agent_dplmbit/tools.py"),
                  "max_policy_calls": max_policy_calls, "max_tool_calls": max_tool_calls}
    if structured_output:
        provenance.update(structured_output=True,
            constraints_sha256=g.sha_file(Path(__file__).with_name("episode_constraints_v4.py" if availability_summary else "episode_constraints.py")),
            initial_lookup_policy="protocol_required_not_autonomous_tool_selection")
    if availability_summary:
        provenance.update(availability_summary=True,
            constraints_base_sha256=g.sha_file(Path(__file__).with_name("episode_constraints.py")),
            availability_module_sha256=g.sha_file(Path(__file__).resolve().parents[2]/"src/agent_dplmbit/availability.py"),
            repeated_lookup_policy="forbidden_after_actual_same_state_lookup")
    if unbound_prior_task:
        provenance["unbound_prior_task"] = True
    binding = g.digest(provenance)
    jobs, blocked = [], []
    for req in requests:
        review = reviews.get(req.request_id)
        if review is None or not review["ready"]:
            blocked.append({"request_id": req.request_id, "reason": "episode_input_not_ready"})
            continue
        if review["public_sha256"] != req.public_sha256:
            raise ContractError("Episode review public-input hash differs")
        if req.public.observations or req.public.history:
            raise ContractError("Episode v1 needs empty initial history/observations; prefilled reports require an audited execution trace")
        jobs.append((req, review))
    if jobs and token_counter is None and config["tokenizer_path"]:
        if not dry_run or Path(config["tokenizer_path"]).is_dir():
            token_counter = g.TokenCounter(config["tokenizer_path"], config["enable_thinking"], config["reasoning_effort"])
    accepted = []
    for req, review in jobs:
        context = g.audit_context(messages_for(req, req.public, review, max_policy_calls, max_tool_calls,
                                  structured_output=structured_output, availability_summary=availability_summary,
                                  unbound_prior_task=unbound_prior_task), {}, config, token_counter)
        if not context.get("verified") or not context.get("fits"):
            blocked.append({"request_id": req.request_id, "reason": "initial_context_not_verified_or_over_budget", "context_audit": context})
        else:
            accepted.append((req, review))
    summary = {"schema": "agent_dplmbit.teacher_episode_run.v1", "total_inputs": len(requests),
               "ready_inputs": len(accepted), "blocked": blocked, "dry_run": dry_run,
               "binding_sha256": binding, "approved_for_training": False, "semantic_claims_verified": False}
    if dry_run or not accepted:
        return summary
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    workdir = output.with_suffix(output.suffix+".work")
    manifest = output.with_suffix(output.suffix+".manifest.json")
    with output.with_suffix(output.suffix+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = {"binding_sha256": binding, "provenance": provenance}
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ContractError("Resume configuration/input/evidence/code hash differs; use a new output")
        elif output.exists() or workdir.exists():
            raise ContractError("Unbound prior output/checkpoint exists")
        else:
            atomic_json(manifest, expected)
        workdir.mkdir(exist_ok=True)
        completed = set()
        for line in output.read_text().splitlines() if output.exists() else []:
            row = json.loads(line); verify_hash(row, "episode_hash")
            if row.get("schema") != SCHEMA or row.get("approved_for_training") is not False or row.get("semantic_claims_verified") is not False or row.get("status") not in TERMINAL:
                raise ContractError("Invalid or promoted prior episode")
            if row["provenance"]["binding_sha256"] != binding or row["episode_key"] in completed:
                raise ContractError("Duplicate or unbound prior episode")
            completed.add(row["episode_key"])
        pending = [(r, v) for r, v in accepted if g.digest({"binding": binding, "request_id": r.request_id}) not in completed]
        summary.update(resumed_skipped=len(accepted)-len(pending), new_episodes=0)
        if not pending:
            return summary
        headers = {}
        if config["api_key_env"]:
            secret = os.environ.get(config["api_key_env"])
            if not secret:
                raise ContractError("Configured API key environment variable is missing")
            headers["Authorization"] = "Bearer "+secret
        limiter = g.RateLimiter(config["requests_per_second"])
        semaphore = asyncio.Semaphore(config["concurrency"])
        async with httpx.AsyncClient(timeout=config["timeout_seconds"], trust_env=False, headers=headers, transport=transport,
                                    limits=httpx.Limits(max_connections=config["concurrency"])) as client:
            if config["check_models"]:
                models = await client.get(config["base_url"].rstrip("/")+"/models"); models.raise_for_status()
                if g.MODEL not in {r.get("id") for r in models.json().get("data", [])}:
                    raise ContractError("Service does not advertise the required model")
            async def work(req, review):
                async with semaphore:
                    return await collect_one(req, review, registry, client, config, token_counter, limiter,
                        output, workdir, binding, provenance, max_policy_calls, max_tool_calls, structured_output,
                        availability_summary, unbound_prior_task)
            records = await asyncio.gather(*(work(r, v) for r, v in pending))
        summary.update(new_episodes=len(records), statuses=dict(Counter(r["status"] for r in records)),
                       policy_calls=sum(r["policy_calls"] for r in records), tool_calls=sum(r["tool_calls"] for r in records),
                       http_attempts=sum(r["http_attempts"] for r in records),
                       completed_conditions=sum(r["status"] == "completed" for r in records),
                       nonempty_condition_proposals=sum(r["status"] == "completed" and bool(r["turns"][-1]["validated_response"]["final"]["hypotheses"]) for r in records))
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "requests", "review-manifest", "evidence-map", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--max-policy-calls", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--availability-summary", action="store_true",
                        help="Opt-in v4 current-state availability and no-repeat lookup; requires --structured-output")
    parser.add_argument("--unbound-prior-task", action="store_true",
                        help="Opt-in v5 source-scoped pending AA priors; requires --availability-summary and --structured-output")
    args = parser.parse_args(argv)
    result = asyncio.run(run(args.config, args.requests, args.review_manifest, args.evidence_map, args.output,
                             max_policy_calls=args.max_policy_calls, max_tool_calls=args.max_tool_calls, dry_run=args.dry_run,
                             structured_output=args.structured_output, availability_summary=args.availability_summary,
                             unbound_prior_task=args.unbound_prior_task))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["blocked"] and not any(k != "completed" and v for k, v in result.get("statuses", {}).items()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
