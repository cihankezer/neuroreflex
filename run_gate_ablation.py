"""Matched legacy/geometric experiments and explicit safety/selectivity comparisons."""

import argparse
import csv
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from perception import PerceptionConfig
from run_dynamic_cases import run_dynamic_cases, write_dynamic_csv, write_tick_csv
from run_static_cases import summarize_results
from threat_gate import ThreatGateConfig


BENCHMARKS = (("fixed", 6, 1.2, 1.2), ("intrusion", 24, 0.3, 2.0),
              ("stress", 24, 2.0, 6.0))


def summarize_ablation(results):
    """Safety rates keep their existing denominators; actuator stats pool time.

    Saturation ratio and joint RMS use summed reflex-active seconds, not an
    unweighted average of per-trial ratios. No reflex time is undefined.
    """
    summary = {r["Metric"]: r["Value"] for r in summarize_results(results)}
    active = sum(r.diagnostics.reflex_active_time_s for r in results)
    saturated = sum(r.diagnostics.saturated_reflex_time_s for r in results)
    row = {
        "trials": len(results),
        "threatening_trajectories": summary["baseline_threatening_cases"],
        "non_threatening_trajectories": summary["baseline_non_threatening_cases"],
        "collisions": summary["collisions"],
        "collision_rate_percent": summary["collision_rate"],
        "false_triggers": summary["false_triggers"],
        "false_trigger_rate_percent": summary["false_trigger_rate"],
        "reaction_latency_median_ms": summary["reaction_latency_median"],
        "reaction_latency_p95_ms": summary["reaction_latency_p95"],
        "reaction_latency_samples": summary["reaction_latency_samples"],
        "pre_intrusion_responses": summary["pre_intrusion_responses"],
        "threats_without_response": summary["threats_without_response"],
        "threats_with_late_response": summary["threats_with_late_response"],
        "min_clearance_m": summary["min_clearance_worst"] / 1000,
        "reflex_active_time_s": active,
        "saturated_reflex_time_s": saturated,
        "saturation_ratio": saturated / active if active else None,
    }
    for joint in ("shoulder", "elbow"):
        name = f"{joint}_velocity_residual_rms_deg_s"
        weighted = sum(r.diagnostics.reflex_active_time_s * (getattr(r.diagnostics, name) or 0)**2
                       for r in results)
        row[name] = math.sqrt(weighted / active) if active else None
    return row


def compare_matched(legacy, geometric):
    """Compare events per matched path, including new collisions and missed threats."""
    if len(legacy) != len(geometric):
        raise ValueError("Matched sets must have the same length.")
    prevented, new_false, missed, late, collisions, delays_ms = [], [], [], [], [], []
    for old, new in zip(legacy, geometric):
        identity = lambda r: (r.case_id, r.trajectory_type, r.speed_m_s, r.perception_seed,
                              r.metrics.baseline_contact, r.metrics.baseline_min_clearance_m)
        if identity(old) != identity(new):
            raise ValueError("Ablations require identical physical paths and perception seeds.")
        if old.metrics.false_trigger and not new.metrics.false_trigger:
            prevented.append(old.case_id)
        if not old.metrics.false_trigger and new.metrics.false_trigger:
            new_false.append(old.case_id)
        if old.metrics.baseline_contact and old.metrics.first_reflex_time_s is not None \
                and new.metrics.first_reflex_time_s is None:
            missed.append(old.case_id)
        if old.metrics.reflex_before_baseline_contact is True \
                and new.metrics.reflex_before_baseline_contact is False:
            late.append(old.case_id)
        if not old.metrics.contact and new.metrics.contact:
            collisions.append(old.case_id)
        if old.metrics.baseline_contact and old.metrics.first_reflex_time_s is not None \
                and new.metrics.first_reflex_time_s is not None:
            delays_ms.append(1000 * (new.metrics.first_reflex_time_s - old.metrics.first_reflex_time_s))
    return {"false_triggers_prevented": len(prevented), "new_false_triggers": len(new_false),
            "new_missed_threats": len(missed), "new_late_threat_responses": len(late),
            "new_collisions": len(collisions),
            "false_triggers_prevented_cases": ";".join(prevented),
            "new_false_trigger_cases": ";".join(new_false),
            "new_missed_threat_cases": ";".join(missed),
            "new_late_threat_response_cases": ";".join(late),
            "new_collision_cases": ";".join(collisions),
            "paired_threat_response_samples": len(delays_ms),
            "paired_reflex_delay_median_ms": float(np.median(delays_ms)) if delays_ms else None,
            "paired_reflex_delay_p95_ms": float(np.percentile(delays_ms, 95)) if delays_ms else None,
            "paired_reflex_delay_max_ms": max(delays_ms) if delays_ms else None}


def run_ablations(output_dir, seed=42, safety_margin_m=0.03, prediction_horizon_s=1.0,
                  margin_sweep=()):
    gate = ThreatGateConfig("geometric", safety_margin_m, prediction_horizon_s)
    for margin in margin_sweep:
        ThreatGateConfig("geometric", margin, prediction_horizon_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, comparisons, sensitivity = [], [], []
    for label, count, speed_min, speed_max in BENCHMARKS:
        for noisy in (False, True):
            perception = PerceptionConfig(0.01, 0.10, 0.020, 0.05) if noisy else PerceptionConfig()
            corruption = "noisy" if noisy else "clean"
            common = dict(num_trials=count, seed=seed, perception_config=perception,
                          speed_min=speed_min, speed_max=speed_max)
            runs = {}
            for mode in ("legacy", "geometric"):
                selected_gate = ThreatGateConfig(mode, gate.safety_margin_m, gate.prediction_horizon_s)
                results = run_dynamic_cases(**common, threat_gate=selected_gate)
                runs[mode] = results
                stem = f"{label}_{corruption}_{mode}"
                configuration = {"seed": seed, "num_trials": count, "speed_min": speed_min,
                    "speed_max": speed_max, **asdict(perception),
                    **{"gate_" + k: v for k, v in asdict(selected_gate).items()}}
                write_dynamic_csv(output_dir / (stem + ".csv"), results, configuration)
                write_tick_csv(output_dir / (stem + "_ticks.csv"), results)
                row = {"benchmark": label, "perception": corruption, "mode": mode,
                    "seed": seed, "safety_margin_m": safety_margin_m,
                    "prediction_horizon_s": prediction_horizon_s, **summarize_ablation(results)}
                rows.append(row)
                print(f"{stem}: collisions={row['collisions']}/{count}, "
                      f"FP={row['false_triggers']}/{row['non_threatening_trajectories']}, "
                      f"minimum={row['min_clearance_m'] * 1000:.3f} mm")
            comparison = {"benchmark": label, "perception": corruption,
                          **compare_matched(runs["legacy"], runs["geometric"])}
            comparisons.append(comparison)
            for margin in margin_sweep:
                results = (runs["geometric"] if margin == safety_margin_m else
                           run_dynamic_cases(**common,
                               threat_gate=ThreatGateConfig("geometric", margin, prediction_horizon_s)))
                sensitivity.append({"benchmark": label, "perception": corruption,
                    "safety_margin_m": margin, "prediction_horizon_s": prediction_horizon_s,
                    **summarize_ablation(results), **compare_matched(runs["legacy"], results)})
    for name, records in (("comparison_summary", rows), ("matched_tradeoffs", comparisons),
                          ("margin_sensitivity", sensitivity)):
        if records:
            with (output_dir / (name + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return rows, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/gate_ablation"))
    parser.add_argument("--gate-safety-margin", type=float, default=0.03)
    parser.add_argument("--prediction-horizon", type=float, default=1.0)
    parser.add_argument("--margin-sweep", type=float, nargs="*", default=(),
                        help="Optional additional margins to compare against the matched legacy runs.")
    args = parser.parse_args(argv)
    try:
        run_ablations(args.output_dir, args.seed, args.gate_safety_margin,
                      args.prediction_horizon, args.margin_sweep)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
