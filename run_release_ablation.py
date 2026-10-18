"""Matched reflex-release experiments, specified sweep, and candidate verification."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from perception import PerceptionConfig
from reflex_release import ReflexReleaseConfig
from release_metrics import active_episodes
from run_dynamic_cases import run_dynamic_cases, write_dynamic_csv, write_tick_csv
from run_gate_ablation import BENCHMARKS, compare_matched, summarize_ablation
from threat_gate import ThreatGateConfig


RELEASE_MARGINS = (0.04, 0.05, 0.06, 0.08)
RELEASE_DWELLS = (0.0, 0.04, 0.08, 0.12, 0.20)


def summarize_release_results(results):
    row = summarize_ablation(results)
    metrics = [r.release_metrics for r in results]
    for key in ("activation_count", "release_count", "reactivation_count"):
        row[key] = sum(getattr(m, key) for m in metrics)
    row["chatter_events"] = sum(m.number_of_gate_chatter_events for m in metrics)
    row["encounters_with_chatter"] = sum(m.number_of_gate_chatter_events > 0 for m in metrics)
    row["max_transitions"] = max(m.transition_count for m in metrics)
    row["median_transitions"] = float(np.median([m.transition_count for m in metrics]))
    row["active_at_encounter_end"] = sum(m.active_at_encounter_end for m in metrics)
    for key in ("minimum_clearance_after_final_release_m", "minimum_clearance_during_recovery_m"):
        values = [getattr(m, key) for m in metrics if getattr(m, key) is not None]
        row[key] = min(values) if values else None
    return row


def _write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_policy_set(seed, policy, release, post_time, enter_margin=0.03):
    runs = {}
    for label, count, speed_min, speed_max in BENCHMARKS:
        for noisy in (False, True):
            noise = "noisy" if noisy else "clean"
            perception = PerceptionConfig(0.01, 0.10, 0.020, 0.05) if noisy else PerceptionConfig()
            runs[(label, noise)] = run_dynamic_cases(count, seed, perception, speed_min, speed_max,
                threat_gate=ThreatGateConfig(policy, enter_margin), reflex_release=release,
                post_encounter_time_s=post_time)
    return runs


def run_release_sweep(output_dir, seed=42, post_time=0.5):
    """All 20 required margin/dwell combinations, plus five motion ablations.

    Activation is fixed at 30 mm. Compare motion direction at a fixed 60 mm
    release margin for each required dwell, avoiding unrelated parameter search.
    No candidate is silently selected by this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, pooled = [], []
    for margin, dwell, motion in ([(m, d, "none") for m in RELEASE_MARGINS for d in RELEASE_DWELLS]
                                  + [(0.06, d, "receding") for d in RELEASE_DWELLS]):
        release = ReflexReleaseConfig(margin, dwell, motion)
        runs = run_policy_set(seed, "geometric_hysteresis", release, post_time)
        combined = []
        for (benchmark, noise), results in runs.items():
            combined.extend(results)
            rows.append({"seed": seed, "benchmark": benchmark, "perception": noise,
                "enter_margin_m": 0.03, "release_margin_m": margin, "release_dwell_s": dwell,
                "release_motion": motion, "post_encounter_time_s": post_time,
                **summarize_release_results(results)})
        pooled.append({"seed": seed, "release_margin_m": margin, "release_dwell_s": dwell,
            "release_motion": motion, "post_encounter_time_s": post_time,
            **summarize_release_results(combined)})
        print(f"release={margin:.3f}, dwell={dwell:.3f}, motion={motion}: "
              f"collisions={pooled[-1]['collisions']}, FP={pooled[-1]['false_triggers']}, "
              f"reactivations={pooled[-1]['reactivation_count']}, "
              f"chatter={pooled[-1]['chatter_events']}, "
              f"minimum={pooled[-1]['min_clearance_m'] * 1000:.3f}mm, "
              f"active={pooled[-1]['reflex_active_time_s']:.3f}s", flush=True)
        _write_rows(output_dir / "release_sweep.csv", rows)
        _write_rows(output_dir / "release_sweep_pooled.csv", pooled)
    return rows, pooled


def run_matched_release(output_dir, release, seeds=(42, 7, 123), post_time=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, comparisons, episodes = [], [], []
    for seed in seeds:
        runs = {}
        for policy in ("legacy", "geometric_memoryless", "geometric_hysteresis"):
            runs[policy] = run_policy_set(seed, policy, release, post_time)
            for (benchmark, noise), results in runs[policy].items():
                stem = f"seed{seed}_{benchmark}_{noise}_{policy}"
                perception = (PerceptionConfig(0.01, 0.10, 0.020, 0.05)
                              if noise == "noisy" else PerceptionConfig())
                _, count, speed_min, speed_max = next(b for b in BENCHMARKS if b[0] == benchmark)
                configuration = {"seed": seed, "policy": policy, "enter_margin_m": 0.03,
                    "prediction_horizon_s": 1.0, "post_encounter_time_s": post_time,
                    "num_trials": count, "speed_min": speed_min, "speed_max": speed_max,
                    "urgency": 1.0, "ttc_threshold_s": 1.0,
                    **asdict(perception), **asdict(release)}
                write_dynamic_csv(output_dir / (stem + ".csv"), results, configuration)
                write_tick_csv(output_dir / (stem + "_ticks.csv"), results)
                rows.append({"seed": seed, "benchmark": benchmark, "perception": noise,
                    "policy": policy, "enter_margin_m": 0.03, **asdict(release),
                    "post_encounter_time_s": post_time, **summarize_release_results(results)})
                for r in results:
                    for i, episode in enumerate(active_episodes(r.ticks, r.metrics.duration_s)):
                        episodes.append({"seed": seed, "benchmark": benchmark, "perception": noise,
                            "policy": policy, "case_id": r.case_id, "episode": i + 1, **episode})
                print(f"{stem}: collisions={rows[-1]['collisions']}, FP={rows[-1]['false_triggers']}, "
                      f"reactivations={rows[-1]['reactivation_count']}, "
                      f"min={rows[-1]['min_clearance_m'] * 1000:.3f}mm", flush=True)
        for key in runs["legacy"]:
            for reference in ("legacy", "geometric_memoryless"):
                comparisons.append({"seed": seed, "benchmark": key[0], "perception": key[1],
                    "reference": reference, **compare_matched(runs[reference][key],
                                                               runs["geometric_hysteresis"][key])})
    _write_rows(output_dir / "release_comparison.csv", rows)
    _write_rows(output_dir / "release_tradeoffs.csv", comparisons)
    _write_rows(output_dir / "reflex_episodes.csv", episodes)
    return rows, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/release_ablation"))
    parser.add_argument("--sweep", action="store_true", help="Run the required seed-42 release sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 7, 123))
    parser.add_argument("--gate-release-margin", type=float, default=0.06)
    parser.add_argument("--release-dwell", type=float, default=0.04)
    parser.add_argument("--release-motion", choices=("none", "receding"), default="receding")
    parser.add_argument("--post-encounter-time", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        release = ReflexReleaseConfig(args.gate_release_margin, args.release_dwell, args.release_motion)
        if args.sweep:
            run_release_sweep(args.output_dir, post_time=args.post_encounter_time)
        else:
            run_matched_release(args.output_dir, release, args.seeds, args.post_encounter_time)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
