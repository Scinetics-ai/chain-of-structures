#!/usr/bin/env python3
"""Read-only contract gate for static expert four-round snapshots, not scientific approval.

Source paths are relative to the manifest directory and may not escape it. All JSON
objects have explicit schemas: hidden targets cannot be smuggled into extra input
fields. Source hashes/audit records bind claims; they do not prove those claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = Path(__file__).with_name("four_round_contract_20261003.json")
AA = set("ACDEFGHIKLMNPQRSTVWY")
PUBLIC_FIELDS = {
    "function_text", "final_length", "reaction_smiles", "substrate_smiles",
    "ec_numbers", "rhea_id", "master_rhea_id",
}


class ContractError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def keys(value: Any, expected: Any, where: str) -> None:
    require(isinstance(value, dict), f"{where}: expected object")
    require(set(value) == set(expected), f"{where}: missing or unsupported fields")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def nonempty(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{where}: empty string")


def sha(value: Any, where: str) -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"{where}: invalid sha256")


def read_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in out, f"{path}: duplicate JSON key {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ContractError(f"invalid {x}")))


def artifact(ref: Any, root: Path, where: str, *, as_json: bool = False) -> Any:
    keys(ref, {"path", "sha256"}, where)
    nonempty(ref["path"], where)
    sha(ref["sha256"], where)
    relative = Path(ref["path"])
    require(not relative.is_absolute(), f"{where}: absolute source path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"{where}: source escapes manifest directory")
    require(path.is_file(), f"{where}: missing source artifact")
    raw = path.read_bytes()
    require(bool(raw), f"{where}: empty source artifact")
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], f"{where}: source hash mismatch")
    return read_json(path) if as_json else None


def state(value: Any, length: int, slots: list[str], width: int, where: str) -> None:
    keys(value, {"length", "slot_ids", "aa", "bits"}, where)
    require(type(value["length"]) is int and value["length"] == length,
            f"{where}: length changed")
    require(value["slot_ids"] == slots, f"{where}: addresses changed")
    for name in ("aa", "bits"):
        require(isinstance(value[name], list) and len(value[name]) == length,
                f"{where}: {name} length mismatch")
    for a in value["aa"]:
        require(a is None or (isinstance(a, str) and a in AA), f"{where}: invalid amino acid")
    for row in value["bits"]:
        require(row is None or (isinstance(row, list) and len(row) == width
                and all(type(b) is int and b in (0, 1) for b in row)), f"{where}: invalid bits")


def slot_set(value: Any, slots: list[str], where: str) -> set[str]:
    require(isinstance(value, list) and bool(value) and all(isinstance(x, str) for x in value),
            f"{where}: empty or invalid membership")
    require(len(value) == len(set(value)) and set(value) <= set(slots), f"{where}: invalid addresses")
    return set(value)


def validate_record(record: Any, contract: dict, root: Path) -> None:
    keys(record, contract["record_fields"], "record")
    nonempty(record["parent_id"], "parent_id")
    require(record["split"] in {"train", "validation", "test"}, "invalid split")
    length = record["length"]
    require(type(length) is int and length > 0, "invalid length")
    slots = [f"slot_{i:06d}" for i in range(1, length + 1)]
    require(record["slot_ids"] == slots, "slots must be the fixed full-length canvas")
    public = record["public_request"]
    keys(public, PUBLIC_FIELDS, "public_request")
    require(type(public["final_length"]) is int and public["final_length"] == length,
            "public request length mismatch")
    for name in PUBLIC_FIELDS - {"final_length"}:
        require(public[name] not in (None, "", []), f"public_request.{name}: missing value")
    public_source = artifact(record["public_request_source"], root, "public_request_source", as_json=True)
    require(public_source == public, "public request differs from its bound source")

    codec = record["codec"]
    keys(codec, contract["codec_fields"], "codec")
    nonempty(codec["tokenizer_version"], "codec.tokenizer_version")
    sha(codec["checkpoint_sha256"], "codec.checkpoint_sha256")
    width = codec["bit_width"]
    require(type(width) is int and width == 13, "DPLM-bit contract requires recorded 13-bit codec")
    require(codec["encoding_context"] in {"level_local", "full_terminal_then_mask"},
            "undeclared codec encoding context")
    audit = artifact(codec["observability_audit"], root, "codec.observability_audit", as_json=True)
    keys(audit, {"status", "tokenizer_version", "checkpoint_sha256", "encoding_context",
                 "method", "evidence"}, "codec audit")
    require(audit["status"] == "passed_level_local_observability", "codec locality audit not passed")
    for name in ("tokenizer_version", "checkpoint_sha256", "encoding_context"):
        require(audit[name] == codec[name], f"codec audit {name} mismatch")
    nonempty(audit["method"], "codec audit method")
    artifact(audit["evidence"], root, "codec audit evidence")

    hierarchy = record["private_hierarchy"]
    keys(hierarchy, contract["private_hierarchy_fields"], "private_hierarchy")
    require(hierarchy["construction_method"] == "source_annotated_hierarchy",
            "random masks or unnamed construction cannot define structural levels")
    core = slot_set(hierarchy["core_slots"], slots, "core_slots")
    site = slot_set(hierarchy["active_site_slots"], slots, "active_site_slots")
    require(core <= site, "active site must include catalytic core")
    evidence = hierarchy["core_evidence"]
    require(isinstance(evidence, list) and len(evidence) == len(core), "core needs evidence for every residue")
    covered = set()
    for entry in evidence:
        keys(entry, {"slot_id", "source_residue_id", "evidence_type", "evidence_code",
                     "source_quality", "source", "mapping"}, "core evidence")
        require(entry["slot_id"] in core and entry["slot_id"] not in covered, "core evidence address mismatch")
        covered.add(entry["slot_id"])
        nonempty(entry["source_residue_id"], "source residue")
        require(entry["evidence_type"] in {"curated_explicit_catalytic_annotation",
                                            "experimentally_supported_catalytic_annotation"},
                "family/EC hypotheses are not strict catalytic labels")
        nonempty(entry["evidence_code"], "catalytic evidence code")
        require(entry["source_quality"] in {"experimental_annotation", "curated_nonexperimental_annotation"},
                "ungraded catalytic source quality")
        if entry["evidence_type"] == "experimentally_supported_catalytic_annotation":
            require(entry["source_quality"] == "experimental_annotation", "experimental evidence claim mismatch")
        artifact(entry["source"], root, "catalytic source")
        mapping = artifact(entry["mapping"], root, "residue mapping", as_json=True)
        keys(mapping, {"parent_id", "slot_id", "source_residue_id", "status"}, "residue mapping")
        require(mapping == {"parent_id": record["parent_id"], "slot_id": entry["slot_id"],
                            "source_residue_id": entry["source_residue_id"], "status": "verified"},
                "unverified or inconsistent residue mapping")
    definition = hierarchy["active_site_definition"]
    keys(definition, {"method", "source", "selected_slots"}, "active_site_definition")
    require(definition["method"] in {"curated_active_site_annotation", "source_structure_ligand_contacts"},
            "random masks cannot define active site")
    require(slot_set(definition["selected_slots"], slots, "selected_slots") == site,
            "active-site source selection mismatch")
    artifact(definition["source"], root, "active-site source and selection recipe")

    levels = hierarchy["levels"]
    level_names = [r["target_level"] for r in contract["rounds"]]
    keys(levels, level_names, "levels")
    for name in level_names:
        state(levels[name], length, slots, width, f"level.{name}")
    full = levels["full"]
    require(all(a is not None for a in full["aa"]) and all(b is not None for b in full["bits"]),
            "full target is incomplete")
    for name in level_names:
        aa_slots = core if name == "catalytic_core" else site if name in {"active_site", "backbone"} else set(slots)
        bit_slots = core if name == "catalytic_core" else site if name == "active_site" else set(slots)
        expected = {"length": length, "slot_ids": slots,
                    "aa": [a if s in aa_slots else None for s, a in zip(slots, full["aa"])],
                    "bits": [(full["bits"][i] if codec["encoding_context"] == "full_terminal_then_mask"
                              or name in {"backbone", "full"} else levels[name]["bits"][i])
                             if s in bit_slots else None for i, s in enumerate(slots)]}
        require(levels[name] == expected, f"{name}: inconsistent hierarchy or future-token leakage")
        require(all((b is not None) == (s in bit_slots) for s, b in zip(slots, levels[name]["bits"])),
                f"{name}: missing required level structure tokens")

    rounds = record["rounds"]
    require(isinstance(rounds, list) and len(rounds) == 4, "incomplete four-round record")
    empty = {"length": length, "slot_ids": slots, "aa": [None] * length, "bits": [None] * length}
    for item, spec in zip(rounds, contract["rounds"]):
        keys(item, contract["round_fields"], "round")
        for name in ("round_id", "input_level", "target_level"):
            require(item[name] == spec[name], "round order or structural level mismatch")
        require(item["agent_action_labels"] == "not_provided", "reconstruction is not expert action truth")
        inp = item["student_input"]
        keys(inp, contract["student_input_fields"], "student_input")
        require(inp["public_request_sha256"] == digest(public), "public request binding mismatch")
        expected = empty if spec["input_level"] == "none" else levels[spec["input_level"]]
        state(inp["state"], length, slots, width, "student_input.state")
        require(inp["state"] == expected, f"{spec['round_id']}: input contains future or wrong-level state")
        require(item["private_target"] == levels[spec["target_level"]], "round private target mismatch")


def validate_manifest(path: Path, contract_path: Path = DEFAULT_CONTRACT) -> dict:
    contract = read_json(contract_path)
    manifest = read_json(path)
    keys(manifest, contract["manifest_fields"], "manifest")
    require(manifest["schema_version"] == "four_round_manifest_v1", "unsupported manifest version")
    require(manifest["contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "contract byte hash mismatch")
    nonempty(manifest["dataset_id"], "dataset_id")
    for name in ("data_route", "trajectory_claim", "representation"):
        require(manifest[name] == contract[name], f"{name}: unsupported claim")
    records = manifest["records"]
    require(isinstance(records, list) and bool(records), "no complete four-round records")
    seen = set()
    for record in records:
        validate_record(record, contract, path.parent)
        require(record["parent_id"] not in seen, "duplicate parent or cross-split parent")
        seen.add(record["parent_id"])
    return {"contract_validation": "passed", "records": len(records), "rounds": 4 * len(records),
            "semantic_approval": False, "training_authorized": False,
            "scope": "schema_hashes_visibility_and_recorded_audit_only"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    try:
        result = validate_manifest(args.manifest, args.contract)
    except (ContractError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"contract_validation": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
