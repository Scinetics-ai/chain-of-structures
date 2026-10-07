#!/usr/bin/env python3
"""A0 tool-protocol demonstrations, explicitly not expert protein edits.

Creates actual bounded tool executions on an offline observed subset. Public
records contain no full sequence/bit target, no fabricated geometry and no
natural core labels. The policy learns calling/reading tools and finishing, not
the truth of a proposed amino acid or a future generation benefit.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import heapq
import json
from pathlib import Path
import random

from agent_dplmbit.schema import AA_ALPHABET, PublicExample, PublicRequest, PublicState, ToolRequest
from agent_dplmbit.tools import ToolRegistry

VIEWS = (("initial", 0.), ("denoising_low", .25), ("denoising_mid", .60), ("denoising_late", .90))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def select_sources(path: Path, count: int, seed: int):
    """Lowest deterministic hash ranks; selection never uses sequence/length/EC."""
    heap, seen = [], set()
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            rid = row["parent_id"]
            if rid in seen:
                raise ValueError("Duplicate parent_id in natural source file")
            seen.add(rid)
            rank = int(hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest(), 16)
            item = (-rank, rid, row)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    if len(heap) < count:
        raise ValueError(f"Requested {count} parents but only {len(heap)} available")
    return [item[2] for item in sorted(heap, reverse=True)]


def public_view(source: dict, view_name: str, visibility: float, seed: int):
    """Random subsets are observations for protocol practice, not expert trajectories."""
    seq, bits = source["sequence"], source["bits"]
    length = len(seq)
    if not length or len(bits) != length:
        raise ValueError("Invalid source lengths")
    key = f"{seed}:{source['parent_id']}:{view_name}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
    visible_count = min(length - 1, int(length * visibility))
    aa_seen = set(rng.sample(range(length), visible_count))
    struct_seen = set(rng.sample(range(length), visible_count))
    state = PublicState(
        length=length, phase="initial" if view_name == "initial" else "denoising",
        aa=tuple(AA_ALPHABET.index(a) if i in aa_seen else None for i, a in enumerate(seq)),
        bits=tuple(tuple(b) if i in struct_seen else None for i, b in enumerate(bits)),
        editable_mask=(True,) * length, candidate_source="observed")
    return PublicExample(PublicRequest(source["function_text"], length), state)


def protocol_records(source: dict, *, seed: int, source_path: str, source_hash: str, max_items: int = 16):
    registry = ToolRegistry(max_items=max_items)
    for view_name, visibility in VIEWS:
        current = public_view(source, view_name, visibility, seed)
        # Queries depend solely on currently visible values; no private labels.
        visible = [i + 1 for i, aa in enumerate(current.state.aa) if aa is not None]
        remaining = [i + 1 for i, aa in enumerate(current.state.aa) if aa is None]
        slots = (visible[:max_items // 2] + remaining[:max_items - min(len(visible), max_items // 2)])
        state_request = ("get_current_state", {"slots": slots})
        evidence_request = ("lookup_reaction_evidence", {})
        requests = [evidence_request, state_request] if view_name == "initial" else [state_request, evidence_request]
        for turn_index in range(len(requests) + 1):
            action = {"finish": True}
            request = None
            if turn_index < len(requests):
                name, arguments = requests[turn_index]
                request = ToolRequest(name, arguments, current.state.state_hash)
                action = request.to_dict()
            serialized = current.to_dict()
            # Round-trip ensures tool histories refer to the actual current state.
            assert PublicExample.from_dict(serialized).state.state_hash == current.state.state_hash
            yield {
                "schema": "agent_dplmbit.protocol_sft.v1", "parent_id": source["parent_id"],
                "split": source["split"], "public": serialized,
                "assistant_target": json.dumps(action, sort_keys=True, separators=(",", ":")),
                "provenance": {
                    "scope": "A0_tool_protocol_only_not_chemical_or_future_edit_supervision",
                    "source_path": source_path, "source_sha256": source_hash,
                    "view": view_name, "visibility": visibility, "seed": seed,
                    "trajectory_type": "offline_random_observed_subset_not_expert_design_trajectory",
                    "assistant_turn": turn_index, "tool_backend": "actual_ToolRegistry_execution",
                    "reaction_evidence_status": "unavailable_no_external_evidence_map",
                    "geometry_status": "unavailable_no_coordinates_supplied",
                    "contains_private_full_targets": False,
                },
            }
            if request is not None:
                observation = registry.execute(request, current)
                history = current.history + ({"role": "assistant", "state_hash": current.state.state_hash,
                                              "content": json.dumps(request.to_dict(), sort_keys=True)},)
                current = replace(current, observations=current.observations + (observation,), history=history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-parents", type=int, default=2048)
    parser.add_argument("--validation-parents", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "protocol_sft_audit_v1", "scope": "A0_tool_protocol_only",
               "not_supervised": ["chemical_reasoning", "residue_preferences", "future_positions", "generation_benefit"],
               "tools": ["get_current_state", "lookup_reaction_evidence"], "views": list(VIEWS),
               "selection": "deterministic_hash_rank_of_parent_id_no_length_filter", "splits": {}}
    for split, path, count in (("train", args.train, args.train_parents),
                               ("validation", args.validation, args.validation_parents)):
        source_hash = digest(path)
        chosen = select_sources(path, count, args.seed)
        out = args.out_dir / f"protocol_{split}.jsonl"
        temporary = out.with_name(out.name + ".tmp")
        rows = 0
        with temporary.open("w") as fh:
            for source in chosen:
                if source["split"] != split:
                    raise ValueError("Split mismatch")
                for row in protocol_records(source, seed=args.seed, source_path=str(path), source_hash=source_hash):
                    fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                    rows += 1
        temporary.replace(out)
        lengths = [len(r["sequence"]) for r in chosen]
        receipt["splits"][split] = {"parents": len(chosen), "assistant_turn_records": rows,
                                     "length_min": min(lengths), "length_max": max(lengths),
                                     "gt512": sum(n > 512 for n in lengths),
                                     "source_sha256": source_hash, "output_sha256": digest(out),
                                     "output_path": str(out)}
        print(split, receipt["splits"][split], flush=True)
    (args.out_dir / "protocol_sft_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
