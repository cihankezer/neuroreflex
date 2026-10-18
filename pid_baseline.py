"""Capture/verify the pre-PID scientific baseline without storing huge tick logs."""

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

from perception import PerceptionConfig
from reflex_release import ReflexReleaseConfig
from run_dynamic_cases import run_dynamic_cases
from run_static_cases import load_static_cases, run_cases, DEFAULT_WORKBOOK
from simulation_core import nominal_controller, student_controller
from threat_gate import ThreatGateConfig
from trajectories import ALL_TRAJECTORY_TYPES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot():
    data = {"p_source_sha256": hashlib.sha256((inspect.getsource(student_controller)
            + inspect.getsource(nominal_controller)).encode()).hexdigest()}
    data["static"] = [asdict(r) for r in run_cases(load_static_cases(DEFAULT_WORKBOOK))]
    for name, gate, config in (
        ("legacy_clean", "legacy", PerceptionConfig()),
        ("hysteresis_clean", "geometric_hysteresis", PerceptionConfig()),
        ("hysteresis_corrupt", "geometric_hysteresis",
         PerceptionConfig(position_sigma_m=0.005, ttc_relative_sigma=0.05,
                          latency_s=0.04, dropout_probability=0.1)),
    ):
        results = run_dynamic_cases(num_trials=36, seed=17,
            trajectory_types=ALL_TRAJECTORY_TYPES, perception_config=config,
            threat_gate=ThreatGateConfig(mode=gate), reflex_release=ReflexReleaseConfig(),
            post_encounter_time_s=1.0)
        data[name] = [{"case_id": r.case_id, "speed": r.speed_m_s,
                      "perception_seed": r.perception_seed,
                      "metrics": asdict(r.metrics), "diagnostics": asdict(r.diagnostics),
                      "release": asdict(r.release_metrics), "ticks_sha256": digest(r.ticks)}
                     for r in results]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    current = snapshot()
    if args.verify:
        previous = json.loads(args.verify.read_text(encoding="utf-8"))
        if previous != current:
            raise SystemExit("FAIL: P baseline drift (compare snapshot groups).")
        print("PASS: P source, 16 static cases and 108 dynamic cases/tick hashes identical.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Saved baseline: {args.output}")


if __name__ == "__main__":
    main()
