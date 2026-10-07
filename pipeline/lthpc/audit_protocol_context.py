#!/usr/bin/env python3
"""CPU tokenize long and deterministic sampled protocol contexts; never truncate."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from transformers import AutoTokenizer
from agent_dplmbit.policy import prepare_messages
from agent_dplmbit.schema import PublicExample
from agent_dplmbit.system import language_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=16384)
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--sample-count", type=int, default=256)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    selected, file_counts = {}, {}
    # All final-turn contexts with chain>=512 plus a hash-random 256 final turns.
    for path in args.inputs:
        candidates = []
        total = 0
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                total += 1
                if row["provenance"]["assistant_turn"] != 2:
                    continue
                identity = f"{row['split']}:{row['parent_id']}:{row['provenance']['view']}:2"
                rank = hashlib.sha256(identity.encode()).hexdigest()
                candidates.append((rank, identity, row))
                if row["public"]["state"]["length"] >= 512:
                    selected[identity] = row
        for _, identity, row in sorted(candidates)[:args.sample_count]:
            selected[identity] = row
        file_counts[str(path)] = {"records": total, "final_turns": len(candidates)}
    results = []
    for identity, row in sorted(selected.items()):
        public = PublicExample.from_dict(row["public"])
        prefix = prepare_messages(tokenizer, language_messages(public))["input_ids"].shape[1]
        # Match current agent_inputs: literal assistant target IDs; reserve one
        # additional EOS even if the specific trainer has not added it yet.
        target = len(tokenizer(row["assistant_target"], add_special_tokens=False)["input_ids"])
        total = prefix + target + args.queries + 1
        results.append({"record_id": identity, "length": public.state.length,
                        "prefix_tokens": prefix, "target_tokens": target,
                        "reserved_queries_and_eos": args.queries + 1,
                        "total_tokens": total, "overflow": total > args.limit})
    result = {"scope": "all_final_turns_length_ge512_plus_hash256_per_file_not_exhaustive_short_contexts",
              "code_file": inspect.getsourcefile(language_messages),
              "language_messages_sha256": hashlib.sha256(inspect.getsource(language_messages).encode()).hexdigest(),
              "tokenizer_path": args.tokenizer, "max_context": args.limit,
              "source_counts": file_counts, "audited_records": len(results),
              "overflow_records": [r for r in results if r["overflow"]],
              "max_total_tokens": max(r["total_tokens"] for r in results),
              "max_record": max(results, key=lambda r:r["total_tokens"]), "records": results}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
