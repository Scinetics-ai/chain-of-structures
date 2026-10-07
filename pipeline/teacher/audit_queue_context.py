#!/usr/bin/env python3
"""CPU tokenizer probe against a locked release; never sends teacher requests."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.release / "src"), str(a.release / "ops/teacher")]
    spec = importlib.util.spec_from_file_location("queue_context_collector", a.release / "ops/teacher/collect_episodes.py")
    c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)
    g = c.g
    from agent_dplmbit.schema import ToolRequest
    from agent_dplmbit.tools import ToolRegistry
    config = g.read_config(a.config)
    requests = g.read_requests(a.shard / "requests.jsonl")
    reviews = c.read_reviews(a.shard / "reviews.json")
    evidence = json.loads((a.shard / "evidence_map.json").read_text()); g._private_guard(evidence)
    registry = ToolRegistry(reaction_evidence=evidence)
    counter = g.TokenCounter(config["tokenizer_path"], config["enable_thinking"], config["reasoning_effort"])
    audits = []
    for req in requests:
        public = req.public
        review = reviews[req.request_id]
        if req.split != "train" or not review["ready"] or review["public_sha256"] != req.public_sha256:
            raise ValueError("Shard is not a certified train input")
        kwargs = {"structured_output": True, "availability_summary": True}
        initial = c.messages_for(req, public, review, 6, 4, **kwargs)
        before = g.audit_context(initial, {}, config, counter)
        request = ToolRequest("lookup_reaction_evidence", {}, public.state.state_hash)
        observation = registry.execute(request, public); g._private_guard(observation.to_dict())
        # This is a deterministic engineering trace for token measurement only,
        # not a teacher response or a data record eligible for training.
        reason = {"state_hash": public.state.state_hash,
                  "summary": "Read the available public evidence before discussing family-level chemistry.",
                  "uncertainty": "No current-design catalytic identity or address is established.",
                  "evidence_refs": [g.allowed_evidence_refs(public)[0]]}
        final = {"type": "tool_call", **request.to_dict()}
        visible = "<reasoning>" + g.canonical(reason) + "</reasoning>\n<final>" + g.canonical(final) + "</final>"
        after_public = c.advance(public, visible, observation)
        after_messages = c.messages_for(req, after_public, review, 5, 3, **kwargs)
        after = g.audit_context(after_messages, {}, config, counter)
        extra_headroom = config["max_tokens"]
        with_headroom = after["input_tokens"] + extra_headroom + config["max_tokens"] + config["context_safety_margin"]
        audits.append({"request_id": req.request_id, "parent_id": req.parent_id,
            "public_sha256": req.public_sha256, "length": public.state.length,
            "initial": before, "after_actual_lookup_with_engineering_stub": after,
            "tool_observation_sha256": g.digest(observation.to_dict()),
            "state_unchanged": public.state.state_hash == after_public.state.state_hash,
            "additional_history_headroom_probe_tokens": extra_headroom,
            "fits_with_additional_headroom_probe": with_headroom <= config["context_length"],
            "actual_second_turn_budget_verified": False,
            "history_source": "fixed_engineering_protocol_stub_not_teacher",
            "note": "Extra headroom is a sensitivity probe, not a proven bound on re-tokenized future assistant history."})
    files = [a.config, a.shard / "requests.jsonl", a.shard / "reviews.json", a.shard / "evidence_map.json",
             a.release / "ops/teacher/collect_episodes.py", a.release / "ops/teacher/generate_drafts.py",
             a.release / "ops/teacher/episode_constraints_v4.py", a.release / "src/agent_dplmbit/availability.py"]
    receipt = {"schema": "agent_dplmbit.teacher_queue_context_probe.v1", "release": str(a.release),
        "source_sha256": {str(path): g.sha_file(path) for path in files},
        "tokenizer_sha256": counter.sha256, "structured_output": True, "availability_summary": True,
        "records": audits, "requests": len(audits),
        "initial_fits": sum(r["initial"]["fits"] for r in audits),
        "post_lookup_engineering_stub_fits": sum(r["after_actual_lookup_with_engineering_stub"]["fits"] for r in audits),
        "post_lookup_extra_headroom_probe_fits": sum(r["fits_with_additional_headroom_probe"] for r in audits),
        "initial_input_token_range": [min(r["initial"]["input_tokens"] for r in audits), max(r["initial"]["input_tokens"] for r in audits)],
        "post_lookup_input_token_range": [min(r["after_actual_lookup_with_engineering_stub"]["input_tokens"] for r in audits),
                                          max(r["after_actual_lookup_with_engineering_stub"]["input_tokens"] for r in audits)],
        "teacher_calls": 0, "GPU_jobs": 0, "approved_for_training": False,
        "dispatch_ready": False, "validation_gate_passed": False, "blocking_reason": "missing_stage_validation",
        "scope": "CPU exact tokenizer on public inputs and real readonly reports; synthetic history probe is not a teacher episode",
        "runtime_requirement": "Recheck exact complete context before EVERY actual teacher call; no truncation or discarded parents"}
    receipt["receipt_sha256"] = g.digest(receipt)
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists() and a.output.read_text() != text:
        raise ValueError("Refusing to overwrite a different context receipt")
    a.output.write_text(text)
    print(json.dumps({k: receipt[k] for k in ("requests", "initial_fits", "post_lookup_engineering_stub_fits",
        "post_lookup_extra_headroom_probe_fits", "initial_input_token_range", "post_lookup_input_token_range", "teacher_calls")}))


if __name__ == "__main__":
    main()
