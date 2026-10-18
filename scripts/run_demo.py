"""Run one physical collision-course encounter and save its scientific trace."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/demo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/"run_dynamic_cases.py"), "--seed", "42",
        "--num-trials", "1", "--trajectory", "collision_course_center", "--speed-min", "1.2",
        "--speed-max", "1.2", "--threat-gate", "geometric_hysteresis", "--post-encounter-time", "0.5",
        "--csv", str(args.output/"encounter.csv"), "--ticks-csv", str(args.output/"ticks.csv")], check=True)
    print("Scientific trace:", args.output/"ticks.csv")

if __name__ == "__main__":
    main()
