"""Re-run all paired model cases in a new process and compare deterministic outputs."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    original_path = ROOT / "results/generation_records.jsonl"
    if not original_path.exists():
        raise FileNotFoundError("Run the original 32-case generation experiment first")
    original_bytes = original_path.read_bytes()
    subprocess.run([sys.executable, str(ROOT / "generation_runner.py"), "--count", "32",
                    "--max-new-tokens", "192", "--out", "reproducibility_run"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "validate_replication.py"),
                    "--generation-dir", "reproducibility_run", "--report",
                    "results/replay_validation_report.json"], cwd=ROOT, check=True)
    replay_bytes = (ROOT / "reproducibility_run/generation_records.jsonl").read_bytes()
    first = [json.loads(x) for x in original_bytes.decode().splitlines()]
    second = [json.loads(x) for x in replay_bytes.decode().splitlines()]
    fields = ["id", "mode", "output", "tokens", "completed", "json_valid", "schema_valid",
              "argument_exact", "content_argument_exact", "whole_markdown_fence",
              "schema_sha256", "prompt_sha256", "prompt_tokens_sha256", "prompt_token_count"]
    mismatches = []
    if len(first) != 64 or len(second) != 64:
        mismatches.append("Expected 64 paired records in each run")
    for a, b in zip(first, second):
        differing = [key for key in fields if a[key] != b[key]]
        if differing:
            mismatches.append({"id": a["id"], "mode": a["mode"], "fields": differing})
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": "same fixed public model, seed, greedy inputs and hardware; new model process",
              "records_per_run": [len(first), len(second)], "comparison_fields": fields,
              "original_records_sha256": hashlib.sha256(original_bytes).hexdigest(),
              "replay_records_sha256": hashlib.sha256(replay_bytes).hexdigest(),
              "timings_expected_to_differ": True, "all_deterministic_fields_equal": not mismatches,
              "mismatches": mismatches, "both_runs_independently_validated": True,
              "limits": "not a cross-machine, cross-dtype or freshly rebuilt dependency environment guarantee"}
    (ROOT / "results/reproducibility_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2), flush=True)
    if mismatches:
        raise SystemExit("Model outputs differed in the same-hardware repeat; see report")


if __name__ == "__main__":
    main()
