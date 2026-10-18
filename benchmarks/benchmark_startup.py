"""Fresh-process CLI startup samples, serial and excluding uv launcher overhead."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=20)
args = parser.parse_args()
rows = []
result = {
    "python": sys.version,
    "executable": sys.executable,
    "repetitions": args.repetitions,
    "warmup": 0,
    "sampling": "Expensive fresh-process samples; no stable p95 claim from default N=20.",
    "environment": {"LANGFUSE_ENABLED": "false"},
    "rows": rows,
}
env = dict(os.environ, LANGFUSE_ENABLED="false")
for repetition in range(args.repetitions):
    for executable in ("dana-agent", "dana-code", "dana-acp"):
        command = [str(Path(sys.executable).parent / executable), "--help"]
        started = time.monotonic_ns()
        try:
            completed = subprocess.run(command, env=env, capture_output=True, timeout=30, check=False)
            row = {
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "help_present": b"usage" in completed.stdout.lower() or b"usage" in completed.stderr.lower(),
            }
        except subprocess.TimeoutExpired:
            row = {"error": "timeout", "exit_code": None}
        row.update(command=command, repetition=repetition, elapsed_ns=time.monotonic_ns() - started)
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"samples": len(rows), "output": str(args.output)}))
