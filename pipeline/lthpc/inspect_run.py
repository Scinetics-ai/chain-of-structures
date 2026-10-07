"""Print a compact, read-only summary of a running experiment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    path = args.run / "metrics.jsonl"
    latest = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # The writer may be appending the final line.
            if event.get("event") in {"start", "train", "validation", "finished"}:
                latest[event["event"]] = event
    for kind in ("start", "train", "validation", "finished"):
        if kind not in latest:
            continue
        event = latest[kind]
        keys = ("event", "step", "scope", "world_size", "train_records", "assistant_ce",
                "assistant_ce_mean_record", "assistant_ce_mean_token", "validation_records",
                "unclipped_gradient_norm", "elapsed_seconds", "reason", "acceptance_gate_passed")
        summary = {k: event[k] for k in keys if k in event}
        if kind == "validation":
            audit = event.get("autonomous_next_action", {})
            summary["autonomous_next_action"] = {k: audit[k] for k in
                ("examples", "json_valid", "schema_or_execution_valid", "exact_action_match") if k in audit}
        print(json.dumps(summary, ensure_ascii=False))
    status = args.run / "exit_code.txt"
    if status.exists():
        print(json.dumps({"process_exit_code": int(status.read_text().strip())}))
    elif not latest:
        print(json.dumps({"status": "no training metrics yet; inspect run.log for initialization or errors"}))


if __name__ == "__main__":
    main()
