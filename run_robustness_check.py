"""Small fixed-policy robustness check; no tuning and no Cartesian sweep."""

import argparse
import csv
from dataclasses import asdict, fields, replace
import math
from pathlib import Path
import random
import statistics

from dynamic_metrics import evaluate_dynamic_encounter
from perception import PerceivedState, PerceptionConfig, PerceptionModel
from reflex_release import ReflexReleaseConfig, current_clearance_rate
from run_dynamic_cases import DynamicTrialResult, write_dynamic_csv, write_tick_csv
from run_release_ablation import summarize_release_results
from simulation_core import CONTROL_DT, RobotState, SimulationCore, clamp, forward_kinematics, surface_clearance
from threat_gate import ThreatGateConfig, evaluate_threat_gate
from trajectories import ALL_TRAJECTORY_TYPES, AcceleratedTrajectory, make_trajectory


BASELINE = PerceptionConfig(.01, .10, .020, .05)
CONFIGURATIONS = {
    "A_baseline": BASELINE,
    "B_mild_velocity": replace(BASELINE, velocity_sigma_m_s=.05),
    "C_moderate_velocity": replace(BASELINE, velocity_sigma_m_s=.10),
    "D_velocity_bias": replace(BASELINE, velocity_sigma_m_s=.05, velocity_scale=.9),
    "E_wrong_direction": replace(BASELINE, velocity_sigma_m_s=.05, velocity_sign_flip_probability=.01),
    "F_stronger_stress": replace(BASELINE, velocity_sigma_m_s=.10,
        velocity_sign_flip_probability=.05, dropout_probability=.10),
    "G_burst_100ms": replace(BASELINE, burst_duration_s=.10),
    "H_burst_200ms": replace(BASELINE, burst_duration_s=.20),
    # One additional combined condition covers the requested 20% dropout and
    # suggested 0.20 m/s noise / 1.1 scale without another parameter sweep.
    "I_combined_upper_stress": replace(BASELINE, velocity_sigma_m_s=.20,
        velocity_scale=1.1, velocity_sign_flip_probability=.05, dropout_probability=.20),
}
SPEEDS = (.35, 1.2)


def crossing_time(trajectory):
    b = trajectory.initial
    if isinstance(trajectory, AcceleratedTrajectory):
        speed = -b.velocity_z
        return 2 * b.z / (speed + math.sqrt(speed**2 - 2 * trajectory.acceleration_z * b.z))
    return -b.z / b.velocity_z if b.velocity_z < 0 else None


def write_rows(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def observe_trial(result, trajectory):
    """Evaluation-only reconstruction; never feeds ground truth to the policy.

    Compare corrupt motion against the acquisition-time uncorrupted velocity,
    at the SAME noisy position and actual current arm. This isolates velocity
    corruption rather than conflating it with latency or a different arm pose.
    """
    evidence, samples = [], []
    hold_entries = stops = mismatches = false_receding_releases = wrong_releases = 0
    hold_time = 0.0
    previous_hold = False
    previous_stops = [False, False]
    plane = crossing_time(trajectory)
    for tick in result.ticks:
        time = tick["time_s"]
        start = time - CONTROL_DT
        weight = max(0.0, min(CONTROL_DT, result.metrics.duration_s - start))
        holding = tick["reflex_state"] == "PERCEPTION_HOLD"
        hold_entries += holding and not previous_hold
        hold_time += weight if holding else 0
        previous_hold = holding
        if holding:
            assert not tick["recovery_active"]
            assert tick["applied_shoulder_velocity_deg_s"] == tick["applied_elbow_velocity_deg_s"] == 0
        pose = RobotState(tick["shoulder_angle_deg"], tick["elbow_angle_deg"])
        times = [start + .002 * i for i in range(1, 11) if .002 * i <= weight + 1e-12]
        times.append(start + weight)
        if plane is not None and start < plane <= start + weight:
            times.append(plane)
        for sample_time in sorted(set(times)):
            elapsed = sample_time - start
            sample_pose = RobotState(
                clamp(pose.shoulder_deg + tick["applied_shoulder_velocity_deg_s"] * elapsed, -90, 120),
                clamp(pose.elbow_deg + tick["applied_elbow_velocity_deg_s"] * elapsed, -150, 150))
            samples.append((sample_time, surface_clearance(sample_pose, trajectory.state_at(sample_time))[0]))
        at_stop = [sample_pose.shoulder_deg <= -90 or sample_pose.shoulder_deg >= 120,
                   abs(sample_pose.elbow_deg) >= 150]
        stops += sum(now and not old for now, old in zip(at_stop, previous_stops))
        previous_stops = at_stop
        if not tick["perception_available"]:
            continue
        perceived = PerceivedState(**{f.name: tick["perceived_" + f.name] for f in fields(PerceivedState)})
        acquired = trajectory.state_at(perceived.observed_time_s)
        uncorrupted = replace(perceived, velocity_x=acquired.velocity_x,
            velocity_y=acquired.velocity_y, velocity_z=acquired.velocity_z)
        ex, ey, hx, hy = forward_kinematics(pose.shoulder_deg, pose.elbow_deg)
        links = (((0, 0, 0), (ex, ey, 0)), ((ex, ey, 0), (hx, hy, 0)))
        rate = current_clearance_rate(uncorrupted, links)
        correct_motion = rate >= -1e-12
        differs = correct_motion != tick["release_condition_motion"]
        wrong_sign = sum(getattr(perceived, "velocity_" + a) * getattr(acquired, "velocity_" + a)
                         for a in "xyz") < 0
        mismatches += differs
        release = tick["reflex_release_event"]
        false_receding = release and not correct_motion and tick["release_condition_motion"]
        false_receding_releases += false_receding
        wrong_releases += release and wrong_sign
        if differs or release:
            correct_gate = evaluate_threat_gate(uncorrupted, links, config=ThreatGateConfig("geometric_hysteresis"))
            evidence.append({"Case_ID": result.case_id, **tick,
                "uncorrupted_acquisition_velocity_x": acquired.velocity_x,
                "uncorrupted_acquisition_velocity_y": acquired.velocity_y,
                "uncorrupted_acquisition_velocity_z": acquired.velocity_z,
                "uncorrupted_clearance_rate_m_s": rate,
                "uncorrupted_receding_condition": correct_motion,
                "uncorrupted_predicted_min_clearance_m": correct_gate.predicted_min_clearance_m,
                "velocity_direction_wrong": wrong_sign,
                "motion_condition_differs": differs,
                "false_receding_release": false_receding,
                "true_clearance_at_event_m": surface_clearance(pose, trajectory.state_at(time))[0]})
    for row in evidence:
        if row["reflex_release_event"]:
            after = [c for t, c in samples if t + 1e-12 >= row["time_s"]]
            row["minimum_true_clearance_after_this_release_m"] = min(after) if after else None
            row["minimum_true_clearance_during_recovery_m"] = result.release_metrics.minimum_clearance_during_recovery_m
    activation = result.release_metrics.first_activation_time_s
    intrusion = result.metrics.intrusion_time_s
    extra = {"perception_hold_entries": hold_entries, "perception_hold_time_s": hold_time,
        "joint_stop_count": stops, "motion_condition_mismatch_ticks": mismatches,
        "false_receding_release_count": false_receding_releases,
        "releases_with_wrong_velocity_direction": wrong_releases,
        "first_activation_latency_s": activation - intrusion if activation is not None and intrusion is not None else None}
    return extra, evidence


def run_configuration(seed, name, config, output_dir):
    rng = random.Random(seed)
    results, extras, evidence, trials, definitions = [], [], [], [], []
    for speed in SPEEDS:
        for kind in ALL_TRAJECTORY_TYPES:
            trajectory = make_trajectory(kind, speed)
            if config.burst_duration_s:
                event = crossing_time(trajectory)
                if event is None:
                    event = trajectory.duration_s / 2
                perception_config = replace(config, burst_start_s=max(.02, event - .5))
            else:
                perception_config = config
            trajectory = replace(trajectory, duration_s=trajectory.duration_s + .5)
            perception_seed = rng.getrandbits(64)
            sim = SimulationCore(trajectory=trajectory,
                perception=PerceptionModel(perception_config, perception_seed),
                threat_gate=ThreatGateConfig("geometric_hysteresis"), reflex_release=ReflexReleaseConfig())
            metrics = evaluate_dynamic_encounter(sim)
            result = DynamicTrialResult(f"R{len(results)+1:03}", kind, speed, perception_seed,
                sim.result, metrics, sim.diagnostic_summary, sim.control_diagnostics, sim.release_metrics)
            result_extra, result_evidence = observe_trial(result, trajectory)
            metadata = {"seed": seed, "configuration": name, "trajectory": kind, "speed_m_s": speed}
            results.append(result); extras.append(result_extra)
            evidence.extend({**metadata, **row} for row in result_evidence)
            trials.append({**metadata, "Case_ID": result.case_id, "perception_seed": perception_seed,
                **asdict(metrics), **asdict(result.diagnostics), **asdict(result.release_metrics), **result_extra})
            definitions.append({**metadata, "Case_ID": result.case_id, "perception_seed": perception_seed,
                "duration_s": trajectory.duration_s, **asdict(perception_config)})
    row = {"seed": seed, "configuration": name, **asdict(config), **summarize_release_results(results)}
    for key in ("perception_hold_entries", "perception_hold_time_s", "joint_stop_count",
                "motion_condition_mismatch_ticks", "false_receding_release_count",
                "releases_with_wrong_velocity_direction"):
        row[key] = sum(e[key] for e in extras)
    latencies = [e["first_activation_latency_s"] for r,e in zip(results,extras)
                 if r.metrics.baseline_contact and e["first_activation_latency_s"] is not None]
    row["first_activation_latency_median_ms"] = statistics.median(latencies) * 1000 if latencies else None
    stem = f"seed{seed}_{name}"
    write_dynamic_csv(output_dir/(stem+'.csv'), results, {"seed": seed, **asdict(config),
        "threat_gate": "geometric_hysteresis", "enter_margin_m": .03, "prediction_horizon_s": 1,
        **asdict(ReflexReleaseConfig()), "post_encounter_time_s": .5, "speeds": SPEEDS})
    write_tick_csv(output_dir/(stem+'_ticks.csv'), results)
    write_rows(output_dir/(stem+'_trials.csv'), trials)
    write_rows(output_dir/(stem+'_definitions.csv'), definitions)
    print(f"{stem}: collision={row['collisions']}, false={row['false_triggers']}, "
        f"missed={row['threats_without_response']}, min={row['min_clearance_m']*1000:.3f}mm, "
        f"reactivations={row['reactivation_count']}, false receding releases={row['false_receding_release_count']}", flush=True)
    return row, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,))
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=tuple(CONFIGURATIONS))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/robustness_check"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, evidence = [], []
    for seed in args.seeds:
        for name in args.configurations:
            row, observations = run_configuration(seed, name, CONFIGURATIONS[name], args.output_dir)
            summaries.append(row); evidence.extend(observations)
            write_rows(args.output_dir/'robustness_summary.csv', summaries)
            write_rows(args.output_dir/'motion_evidence.csv', evidence)


if __name__ == "__main__":
    main()
