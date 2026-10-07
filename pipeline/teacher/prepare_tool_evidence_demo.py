#!/usr/bin/env python3
"""Build one reviewed, literature-assisted R1 request; never an approved SFT row.

The program schedules a real read-only lookup. This is not evidence that an
agent learned to choose that tool. The source card preserves its hypothesis
status and family scope; natural target files are never opened.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import generate_drafts as teacher
from agent_dplmbit.schema import PublicExample, ToolRequest, ContractError
from agent_dplmbit.tools import ToolRegistry


def write_new(path, value, *, jsonl=False):
    text = teacher.canonical(value) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ContractError(f"Refusing to overwrite a different artifact: {path}")
    else:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--evidence-map", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    original = teacher.read_requests(args.requests)
    reviews = teacher.read_reviews(args.reviews)
    evidence = json.loads(Path(args.evidence_map).read_text())
    registry = ToolRegistry(reaction_evidence=evidence)
    selected = [r for r in original if r.public.request.function_text in evidence]
    if len(selected) != 1:
        raise ContractError("This scoped demonstration needs exactly one matching request")
    base = selected[0]
    old_review = reviews[base.request_id]
    if (base.split != "train" or base.public.state.phase != "initial"
            or base.public.observations or base.public.history
            or not old_review.get("ready")
            or old_review.get("public_sha256") != base.public_sha256):
        raise ContractError("Expected an already reviewed, unobserved train R1 input")
    call = ToolRequest("lookup_reaction_evidence", {}, base.public.state.state_hash)
    observation = registry.execute(call, base.public)
    if observation.result.get("status") != "available":
        raise ContractError("The literature lookup returned no evidence")
    public = PublicExample(base.public.request, base.public.state, (observation,))
    teacher._private_guard(public.to_dict())
    request_id = base.request_id + "_literature_demo_v1"
    evidence_hash = teacher.sha_file(args.evidence_map)
    review_record = {
        "reviewer": "root_primary_abstract_and_source_scope_review_20261003",
        "input_review_sha256": old_review["review_sha256"],
        "evidence_map_sha256": evidence_hash,
        "review_scope": "Public source claims and explicit hypothesis scope checked; no target-specific action certified",
        "tool_actually_executed": True,
        "tool_selected_by": "deterministic_demonstration_builder_not_a_trained_agent",
        "public_sha256": teacher.digest(public.to_dict()),
        "natural_target_files_read": False,
        "approved_for_training": False,
    }
    row = {"schema": teacher.REQUEST_SCHEMA, "request_id": request_id,
           "parent_id": base.parent_id, "split": base.split, "public": public.to_dict()}
    review = {"request_id": request_id, "ready": True,
              "public_sha256": teacher.digest(public.to_dict()),
              "reviewer": review_record["reviewer"], "review_sha256": teacher.digest(review_record),
              "allowed_final_types": ["condition"], "allow_bound_positions": False,
              "reason": "Source-scoped literature hypotheses supplied by a real lookup; candidate drafts remain unapproved."}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "requests.jsonl", row)
    write_new(out / "reviews.json", {"schema": teacher.REVIEW_SCHEMA, "requests": [review]})
    write_new(out / "input_review.json", review_record)
    write_new(out / "tool_trace.json", {"request": call.to_dict(), "response": observation.to_dict(),
                                       "selected_by": "deterministic_demonstration_builder"})
    teacher.read_requests(out / "requests.jsonl")
    teacher.read_reviews(out / "reviews.json")
    print(json.dumps({"request_id": request_id, "parent_id": base.parent_id,
                      "evidence_records": len(observation.result["records"]),
                      "output_dir": str(out), "approved_for_training": False}))


if __name__ == "__main__":
    main()
