"""Reproducible staged PID tuning, matched benchmarks and report generation."""

import argparse
import csv
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import statistics

from motor_controller import (JointGains, PIDConfig, PIDController, JOINTS, JOINT_BOUNDS,
                              CONVENTIONAL_REFERENCE_PATH, load_conventional_reference)
from motor_metrics import fixed_segment_metrics, tracking_metrics
from perception import PerceptionConfig, PerceivedState
from reflex_release import ReflexReleaseConfig
from run_dynamic_cases import run_dynamic_cases, write_dynamic_csv, write_tick_csv
from simulation_core import CONTROL_DT, clamp
from threat_gate import ThreatGateConfig
from trajectories import ALL_TRAJECTORY_TYPES, TrueBallState


PROFILES = (
    ("legacy_clean", "legacy", PerceptionConfig()),
    ("hysteresis_clean", "geometric_hysteresis", PerceptionConfig()),
    ("hysteresis_corrupt", "geometric_hysteresis",
     PerceptionConfig(position_sigma_m=.005, ttc_relative_sigma=.05,
                      latency_s=.04, dropout_probability=.1)),
)


def make_config(kp, ki=0, kd=0, tau=.04):
    return PIDConfig(JointGains(kp, ki, kd), JointGains(kp, ki, kd), tau)


def write_rows(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def characterize(config, label):
    rows = []
    for name, delta in (("small", 3), ("medium", 15), ("large_saturated", 60), ("reversal", 30)):
        controller = PIDController(config)
        controller.prepare("REFLEX_ACTIVE", (98, 130))
        q = [35.0, 55.0]
        ticks = []
        # Reversal keeps state across the target change, as a moving IK target does.
        targets = [[35 + delta, 55 - delta]]
        if name == "reversal":
            targets.append([35 - delta, 55 + delta])
        for segment, target in enumerate(targets):
            initial, angles, segment_ticks = list(q), [], []
            for _ in range(200):
                errors = [t - position for t, position in zip(target, q)]
                requests, details = controller.update(errors, CONTROL_DT, (98, 130), q)
                applied = [clamp(v, -limit, limit) for v, limit in zip(requests, (98, 130))]
                tick = {"time_s": (len(ticks) + 1) * CONTROL_DT, "reflex_active": True, **details}
                for i, joint in enumerate(JOINTS):
                    tick.update({f"{joint}_angle_deg": q[i],
                                 f"{joint}_saturated": abs(requests[i]) > (98, 130)[i],
                                 f"applied_{joint}_velocity_deg_s": applied[i]})
                    q[i] = clamp(q[i] + applied[i] * CONTROL_DT, *JOINT_BOUNDS[i])
                ticks.append(tick)
                segment_ticks.append({**tick, "time_s": (len(segment_ticks) + 1) * CONTROL_DT})
                angles.append(list(q))
            rows.append({"candidate": label, "step": name, "segment": segment,
                         **tracking_metrics(segment_ticks, 4),
                         **fixed_segment_metrics(angles, target, initial)})
    return rows


def suite(controller, config, seed, trials):
    return [(name, run_dynamic_cases(num_trials=trials, seed=seed,
             trajectory_types=ALL_TRAJECTORY_TYPES, controller=controller, pid_config=config,
             perception_config=perception, threat_gate=ThreatGateConfig(mode=gate),
             reflex_release=ReflexReleaseConfig(), post_encounter_time_s=1.0))
            for name, gate, perception in PROFILES]


def summarize(results):
    tracking = [tracking_metrics(r.ticks, r.metrics.duration_s) for r in results]
    def pooled(suffix):
        samples = [(m[f"{j}_{suffix}"], m[f"{j}_tracking_time_s"])
                   for m in tracking for j in JOINTS if m.get(f"{j}_{suffix}") is not None]
        weight = sum(w for _, w in samples)
        return math.sqrt(sum(v * v * w for v, w in samples) / weight) if weight else None
    active = sum(r.diagnostics.reflex_active_time_s for r in results)
    tracking_time = sum(m.get(f"{j}_tracking_time_s", 0) for m in tracking for j in JOINTS)
    latencies = [r.metrics.reaction_latency_s for r in results if r.metrics.baseline_contact
                 and r.metrics.reaction_latency_s is not None and r.metrics.reaction_latency_s >= 0]
    residual_sum = sum(r.diagnostics.reflex_active_time_s * v * v for r in results
                       for v in (r.diagnostics.shoulder_velocity_residual_rms_deg_s,
                                 r.diagnostics.elbow_velocity_residual_rms_deg_s) if v is not None)
    return {
        "trials": len(results), "collisions": sum(r.metrics.contact for r in results),
        "worst_clearance_mm": 1000 * min(r.metrics.min_clearance_m for r in results),
        "median_clearance_mm": 1000 * statistics.median(r.metrics.min_clearance_m for r in results),
        "false_triggers": sum(r.metrics.false_trigger for r in results),
        "non_threats": sum(not r.metrics.baseline_contact for r in results),
        "missed": sum(r.metrics.baseline_contact and r.metrics.first_reflex_time_s is None for r in results),
        "late": sum(r.metrics.baseline_contact and r.metrics.reflex_before_baseline_contact is False for r in results),
        "latency_median_ms": 1000 * statistics.median(latencies) if latencies else None,
        "tracking_rmse_deg": pooled("tracking_rmse_deg"),
        "peak_error_deg": max(m.get(f"{j}_peak_error_deg", 0) for m in tracking for j in JOINTS),
        "saturation_reflex_fraction": sum(r.diagnostics.saturated_reflex_time_s for r in results) / active if active else None,
        "saturation_tracking_fraction": sum(m.get(f"{j}_saturation_fraction", 0) * m.get(f"{j}_tracking_time_s", 0)
                                            for m in tracking for j in JOINTS) / tracking_time if tracking_time else None,
        "request_applied_rms_deg_s": math.sqrt(residual_sum / (2 * active)) if active else None,
        "mean_reflex_active_s": active / len(results),
        "joint_stop_count": sum(m.get(f"{j}_joint_stop_count", 0) for m in tracking for j in JOINTS),
        "i_rms_deg_s": pooled("i_rms_deg_s"), "d_rms_deg_s": pooled("d_rms_deg_s"),
    }


def flatten(groups):
    return [r for _, results in groups for r in results]


def safety_violations(candidate, baseline):
    violations = 0
    for c, p in zip(candidate, baseline):
        assert (c.case_id, c.speed_m_s, c.perception_seed) == (p.case_id, p.speed_m_s, p.perception_seed)
        violations += int(c.metrics.contact)
        violations += int(c.metrics.false_trigger and not p.metrics.false_trigger)
        violations += int(c.metrics.baseline_contact and c.metrics.first_reflex_time_s is None)
        violations += int(c.metrics.baseline_contact and c.metrics.reflex_before_baseline_contact is False
                          and p.metrics.reflex_before_baseline_contact is not False)
        # Fixed, predeclared clearance constraint; do not trade safety for RMSE.
        violations += int(c.metrics.min_clearance_m < max(.03, .9 * p.metrics.min_clearance_m))
    return violations


def verify_matched_inputs(candidate, baseline):
    """Verify actual input traces, beyond merely recording identical seeds."""
    for c, p in zip(candidate, baseline):
        assert len(c.ticks) == len(p.ticks)
        for ct, pt in zip(c.ticks, p.ticks):
            keys = ["true_" + f.name for f in fields(TrueBallState)]
            keys += ["perceived_" + f.name for f in fields(PerceivedState)]
            keys += ["true_ttc_s", "time_s", "perception_available",
                     "shoulder_velocity_limit_deg_s", "elbow_velocity_limit_deg_s"]
            assert {k: ct.get(k) for k in keys} == {k: pt.get(k) for k in keys}


def tune(out):
    if CONVENTIONAL_REFERENCE_PATH.exists():
        raise RuntimeError("Conventional tuning is complete and frozen. Use run_pid_final_validation.py or the conventional-reference preset.")
    # Historical search retained below for provenance; it is no longer an active workflow.
    baseline = flatten(suite("p", None, 17, 18))
    candidates, steps, trials = [], [], []
    def evaluate(stage, config):
        gains = config.shoulder
        label = f"{stage}_p{gains.kp:g}_i{gains.ki:g}_d{gains.kd:g}_t{config.derivative_tau_s:g}"
        print(f"Tuning {label}", flush=True)
        char = characterize(config, label)
        steps.extend(char)
        groups = suite("pid", config, 17, 18)
        results = flatten(groups)
        summary = summarize(results)
        violations = safety_violations(results, baseline)
        unsettled = sum(r[f"{j}_settling_s"] is None for r in char for j in JOINTS)
        overshoot = max(r[f"{j}_overshoot_deg"] for r in char for j in JOINTS)
        row = {"candidate": label, "stage": stage, "kp": gains.kp, "ki": gains.ki,
               "kd": gains.kd, "tau": config.derivative_tau_s, "safety_violations": violations,
               "unsettled_segments": unsettled, "max_step_overshoot_deg": overshoot, **summary}
        candidates.append(row)
        for profile, results in groups:
            trials.extend({"candidate": label, "profile": profile, "case_id": r.case_id,
                           "speed": r.speed_m_s, "perception_seed": r.perception_seed,
                           **asdict(r.metrics), **tracking_metrics(r.ticks, r.metrics.duration_s)} for r in results)
        # Rank feasible candidates by safety margin, tracking, saturation and transient behavior.
        return row
    def rank(row):
        return (row["safety_violations"], row["unsettled_segments"], -row["worst_clearance_mm"],
                row["tracking_rmse_deg"], row["saturation_reflex_fraction"], row["max_step_overshoot_deg"])
    coarse = [evaluate("P", make_config(kp)) for kp in (2.5, 4, 6, 9, 12, 16)]
    best_p = min(coarse, key=rank)
    pd = [best_p] + [evaluate("PD", make_config(best_p["kp"], kd=kd)) for kd in (.02, .05, .1, .2)]
    best_pd = min(pd, key=rank)
    integral = [best_pd] + [evaluate("I", make_config(best_pd["kp"], ki=ki, kd=best_pd["kd"]))
                          for ki in (.05, .2, .5)]
    best = min(integral, key=rank)
    local = [best] + [evaluate("local", make_config(kp, best["ki"], best["kd"]))
                      for kp in (best["kp"] * .875, best["kp"] * 1.125)]
    # Filter sensitivity is inspected only for a nonzero derivative pathway.
    local += [evaluate("filter", make_config(best_p["kp"], kd=.05, tau=tau)) for tau in (.02, .08)]
    best = min(local, key=rank)
    feasible = [r for r in candidates if r["safety_violations"] == 0 and r["unsettled_segments"] == 0]
    write_rows(out / "tuning.csv", candidates)
    write_rows(out / "characterization.csv", steps)
    write_rows(out / "tuning_trials.csv", trials)
    if not feasible:
        raise RuntimeError("No feasible PID candidate; retain P. Inspect all saved failed experiments.")
    best = min(feasible, key=rank)
    config = make_config(best["kp"], best["ki"], best["kd"], best["tau"])
    # Diagnostic ablation after locking selection; never select on held-out results.
    evaluate("ablation", make_config(best["kp"]))
    write_rows(out / "tuning.csv", candidates)
    write_rows(out / "characterization.csv", steps)
    write_rows(out / "tuning_trials.csv", trials)
    (out / "selected.json").write_text(json.dumps({"config": asdict(config), "training": best}, indent=2))
    print(f"Selected {best['candidate']} (training seed 17 only)", flush=True)
    return config


def load_selected(out):
    return load_conventional_reference()


def replay(results, config, controller_name):
    """Open-loop target replay: both controllers see identical P-recorded targets/modes/limits."""
    rows = []
    for result in results:
        controller = PIDController(config)
        q, ticks = [35., 55.], []
        for source in result.ticks:
            mode = source["reflex_state"]
            limits = [source[f"{j}_velocity_limit_deg_s"] for j in JOINTS]
            controller.prepare(mode, limits)
            if mode == "PERCEPTION_HOLD":
                continue
            targets = [source[f"{j}_angle_deg"] + source[f"{j}_error_deg"] for j in JOINTS]
            errors = [target - angle for target, angle in zip(targets, q)]
            if controller_name == "pid":
                requests, details = controller.update(errors, CONTROL_DT, limits, q)
            else:
                ttc = source.get("perceived_ttc_s", math.inf)
                kp = 2.5 if mode == "NOMINAL" else (12 if ttc < .2 else 9 if ttc < .4 else 6 if ttc < .7 else 4)
                requests = [kp * e for e in errors]
                details = {f"{j}_error_deg": e for j, e in zip(JOINTS, errors)}
            tick = {"time_s": source["time_s"], **details}
            for i, joint in enumerate(JOINTS):
                velocity = clamp(requests[i], -limits[i], limits[i])
                tick.update({f"{joint}_angle_deg": q[i], f"{joint}_saturated": abs(requests[i]) > limits[i],
                             f"applied_{joint}_velocity_deg_s": velocity})
                q[i] = clamp(q[i] + velocity * CONTROL_DT, *JOINT_BOUNDS[i])
            ticks.append(tick)
        rows.append({"controller": controller_name, "case_id": result.case_id,
                     **tracking_metrics(ticks, result.metrics.duration_s)})
    return rows


def evaluate(out, config):
    summary, tracking, replay_rows, pairs = [], [], [], []
    # Training seed is reported separately; gains are locked before held-out runs.
    for seed in (17, 23, 41):
        previous = None
        for controller in ("p", "pid"):
            print(f"Evaluation seed={seed} controller={controller}", flush=True)
            groups = suite(controller, config, seed, 36)
            summary.append({"seed": seed, "profile": "all", "controller": controller, **summarize(flatten(groups))})
            for profile, results in groups:
                summary.append({"seed": seed, "profile": profile, "controller": controller, **summarize(results)})
                stem = out / f"{controller}_{profile}_seed{seed}"
                configuration = {"controller": controller, "seed": seed, "num_trials": 36,
                                 "trajectory_types": str(ALL_TRAJECTORY_TYPES), "post_encounter_time_s": 1.0,
                                 "control_dt": CONTROL_DT, "geometry_dt": .002,
                                 "gate": profile, "pid_config": json.dumps(asdict(config)),
                                 "perception_config": json.dumps(asdict(next(p for n, _, p in PROFILES if n == profile)))}
                write_dynamic_csv(stem.with_suffix(".csv"), results, configuration)
                # One full matched trace pair suffices for inspecting reset and anti-windup.
                if seed == 23 and profile == "hysteresis_corrupt":
                    write_tick_csv(out / f"{controller}_ticks.csv", results)
                tracking.extend({"seed": seed, "profile": profile, "controller": controller,
                                 "case_id": r.case_id, **tracking_metrics(r.ticks, r.metrics.duration_s)} for r in results)
                if controller == "p":
                    for name in ("p", "pid"):
                        replay_rows.extend({"seed": seed, "profile": profile, **r} for r in replay(results, config, name))
            if controller == "p":
                previous = flatten(groups)
            else:
                verify_matched_inputs(flatten(groups), previous)
                violations = safety_violations(flatten(groups), previous)
                summary[-len(groups)-1]["safety_violations"] = violations
                for i, (c, p) in enumerate(zip(flatten(groups), previous)):
                    pairs.append({"seed": seed, "case_id": c.case_id,
                                  "profile": PROFILES[i // 36][0],
                                  "p_clearance_mm": p.metrics.min_clearance_m * 1000,
                                  "pid_clearance_mm": c.metrics.min_clearance_m * 1000,
                                  "clearance_change_mm": (c.metrics.min_clearance_m - p.metrics.min_clearance_m) * 1000,
                                  "clearance_retained_fraction": c.metrics.min_clearance_m / p.metrics.min_clearance_m})
                print(f"Seed {seed}: PID safety constraint violations={violations}", flush=True)
    write_rows(out / "comparison.csv", summary)
    write_rows(out / "tracking.csv", tracking)
    write_rows(out / "target_replay.csv", replay_rows)
    write_rows(out / "paired_clearance.csv", pairs)
    write_rows(out / "selected_steps.csv", characterize(config, "selected"))


def report(out):
    from pid_report import generate_report
    generate_report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("characterize", "tune", "evaluate", "report", "all"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/pid"))
    args = parser.parse_args()
    if args.stage in ("tune", "all") and CONVENTIONAL_REFERENCE_PATH.exists():
        parser.error("Conventional tuning is frozen; use run_pid_final_validation.py all instead.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "characterize":
        write_rows(args.output / "initial_steps.csv", [row for kp in (2.5, 4, 6, 9, 12, 16)
                   for row in characterize(make_config(kp), f"P{kp}")])
    if args.stage in ("tune", "all"):
        tune(args.output)
    if args.stage in ("evaluate", "all"):
        evaluate(args.output, load_selected(args.output))
    if args.stage in ("report", "all"):
        report(args.output)


if __name__ == "__main__":
    main()
