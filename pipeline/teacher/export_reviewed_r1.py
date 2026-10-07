#!/usr/bin/env python3
"""Export certified public goals as all-MASK train/R1 inputs; no teacher call.

This transports existing input-readiness certificates, never creates chemical
expert labels or approves a teacher answer. Natural protein files are not read.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lthpc"))
import build_uniform_public_manifest as uniform
import pilot_r1 as pilot
from agent_dplmbit.schema import ContractError, PublicExample, PublicRequest, PublicState


def export(goals_path, certificates_path, output_dir):
    goals = [json.loads(line) for line in Path(goals_path).read_text().splitlines() if line.strip()]
    certificates = uniform.read_certificates(certificates_path)
    requests, reviews, seen, evidence = [], [], set(), {}
    for goal in goals:
        if (goal.get("schema") != "agent_dplmbit.uniform_public_goal.v1" or goal.get("split") != "train"
                or goal.get("readiness", {}).get("ready") is not True):
            raise ContractError("Only certified ready train public goals can enter the R1 pilot")
        parent = goal["parent_id"]
        if not isinstance(parent, str) or not parent or parent in seen:
            raise ContractError("R1 pilot requires unique nonempty parent IDs")
        seen.add(parent)
        if set(goal["public_goal"]) != uniform.PUBLIC_FIELDS:
            raise ContractError("R1 pilot must preserve the complete seven-field public input")
        digest = uniform.canonical_hash(goal["public_goal"])
        cert = certificates.get(("train", parent))
        if (goal.get("public_goal_sha256") != digest
                or uniform.certificate_status(cert, digest) != "hash_bound_certificate_present"
                or cert.get("candidate_or_SFT_labels_approved") is not False):
            raise ContractError("Missing, stale or incorrect-scope input certificate for " + parent)
        for source in cert["evidence_sources"]:
            if not pilot.teacher.valid_sha(source.get("sha256")):
                raise ContractError("Certificate evidence needs a SHA256")
            if "path" in source:
                path = Path(source["path"])
                if not path.is_absolute():
                    path = Path(certificates_path).resolve().parent / path
                if not path.is_file() or uniform.file_hash(path) != source["sha256"]:
                    raise ContractError("Certified local evidence changed or is missing: " + str(path))
                evidence[str(path)] = source["sha256"]
            elif not isinstance(source.get("url"), str) or not source["url"].startswith("https://"):
                raise ContractError("Untraceable public evidence source")
        request = PublicRequest.from_dict(uniform.as_public_request(goal))
        length = request.final_length
        state = PublicState(length, "initial", (None,)*length, (None,)*length, (True,)*length)
        public = PublicExample(request, state)
        pilot.teacher._private_guard(public.to_dict())
        request_id = "r1_" + pilot.teacher.digest([parent, digest])[:24]
        requests.append({"schema": pilot.teacher.REQUEST_SCHEMA, "request_id": request_id,
                         "parent_id": parent, "split": "train", "public": public.to_dict()})
        reviews.append({"request_id": request_id, "public_sha256": pilot.teacher.digest(public.to_dict()),
                        "ready": True, "reviewer": cert["reviewer"],
                        "review_sha256": uniform.canonical_hash(cert), "allowed_final_types": ["condition"],
                        "allow_bound_positions": False,
                        "reason": "Inherited passed source-consistency certificate; not candidate/SFT approval."})
    if not requests:
        raise ContractError("No certified train goals supplied")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pilot._write_same(directory / "requests.jsonl", "".join(pilot.teacher.canonical(r)+"\n" for r in requests))
    pilot._write_same(directory / "reviews.json", pilot.teacher.canonical({"schema": pilot.teacher.REVIEW_SCHEMA,
                                                                        "requests": reviews})+"\n")
    receipt = {"schema": "agent_dplmbit.reviewed_r1_export.v1", "count": len(requests),
        "goals_sha256": uniform.file_hash(goals_path), "certificates_sha256": uniform.file_hash(certificates_path),
        "exporter_sha256": uniform.file_hash(__file__), "uniform_exporter_sha256": uniform.file_hash(uniform.__file__),
        "verified_local_evidence_sha256": evidence, "request_ids": [r["request_id"] for r in requests],
        "requests_sha256": uniform.file_hash(directory / "requests.jsonl"),
        "reviews_sha256": uniform.file_hash(directory / "reviews.json"),
        "scope": "all_mask_initial_public_input_readiness_only_no_protein_history_or_expert_labels",
        "natural_protein_files_read": False, "teacher_called": False,
        "teacher_drafts_approved_for_training": False}
    pilot._write_same(directory / "export_receipt.json", pilot.teacher.canonical(receipt)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", required=True)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.goals, args.certificates, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
