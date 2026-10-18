"""Matched, seeded baseline/gate/controller runs; configuration accompanies each CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"benchmarks/results/portable")
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for noisy in (False, True):
        for policy, controller in (("legacy", "p"), ("geometric_memoryless", "p"),
                                   ("geometric_hysteresis", "p"), ("geometric_hysteresis", "pid")):
            label = f"{policy}_{controller}_{'noisy' if noisy else 'clean'}"
            command = [sys.executable, str(ROOT/"run_dynamic_cases.py"), "--seed", "42",
                       "--num-trials", str(args.trials), "--speed-min", "0.3", "--speed-max", "2.0",
                       "--threat-gate", policy, "--controller", controller,
                       "--post-encounter-time", "0.5", "--spiking-config", "spiking_reference.json",
                       "--csv", str((args.output/(label+".csv")).relative_to(ROOT))
                       if args.output.is_relative_to(ROOT) else str(args.output/(label+".csv"))]
            if controller == "pid":
                command += ["--pid-preset", "conventional-reference"]
            if noisy:
                command += ["--position-noise", "0.01", "--ttc-noise", "0.10",
                            "--latency", "0.020", "--dropout", "0.05"]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            (args.output/(label+".txt")).write_text(result.stdout+result.stderr, encoding="utf-8")
            runs.append({"label": label, "arguments": command[2:], "exit_code": result.returncode,
                         "csv": label+".csv"})
            if result.returncode:
                raise SystemExit(result.stderr)
            print(label, "PASS", flush=True)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    (args.output/"manifest.json").write_text(json.dumps({"python": sys.version, "runs": runs,
        "source_sha256": sources}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
