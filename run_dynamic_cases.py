"""Seeded physical trajectory benchmark with optional perception corruption."""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import random

from diagnostics import TrialDiagnostics
from dynamic_metrics import evaluate_dynamic_encounter
from performance_metrics import EncounterMetrics, GEOMETRY_DT, INTRUSION_CLEARANCE, REACTION_THRESHOLD
from perception import PerceptionConfig, PerceptionModel, PerceivedState
from run_static_cases import summarize_results
from simulation_core import (ARM_RADIUS, BALL_RADIUS, CONTROL_DT, ELBOW_MAX, L1, L2,
                             NEUTRAL_ELBOW, NEUTRAL_SHOULDER, SHOULDER_MAX, SimulationCore)
from trajectories import ALL_TRAJECTORY_TYPES, TRAJECTORY_TYPES, TrueBallState, make_trajectory
from threat_gate import GateDecision, ThreatGateConfig
from reflex_release import ReflexReleaseConfig, ReleaseDecision
from release_metrics import ReflexEncounterMetrics
from motor_controller import PID_FIELDS, add_controller_arguments, config_from_args
from spiking_config import spiking_config_from_args
from state_activation_metrics import (StateActivationMetrics, calculate_state_activation_metrics,
                                      summarize_state_activation_metrics)


@dataclass(frozen=True)
class DynamicTrialResult:
    case_id: str
    trajectory_type: str
    speed_m_s: float
    perception_seed: int
    result: str
    metrics: EncounterMetrics
    diagnostics: TrialDiagnostics
    ticks: list
    release_metrics: ReflexEncounterMetrics | None = None

    @property
    def state_activation_metrics(self):
        return calculate_state_activation_metrics(self.ticks, self.metrics.baseline_contact,
                                                  self.metrics.duration_s)


def run_dynamic_cases(num_trials=24, seed=0, perception_config=None,
                      speed_min=0.3, speed_max=2.0, trajectory_types=TRAJECTORY_TYPES,
                      urgency=1.0, ttc_threshold=1.0, geometry_dt=GEOMETRY_DT,
                      intrusion_clearance=INTRUSION_CLEARANCE,
                      reaction_threshold=REACTION_THRESHOLD, threat_gate=None,
                      reflex_release=None, post_encounter_time_s=0.0,
                      controller="p", pid_config=None, spiking_config=None, reuse_spiking_runtime=False):
    if not isinstance(num_trials, int) or num_trials <= 0:
        raise ValueError("Number of trials must be a positive integer.")
    if not all(math.isfinite(v) for v in (speed_min, speed_max)) \
            or not 0 < speed_min <= speed_max:
        raise ValueError("Speed bounds must be finite, positive, and ordered.")
    if not trajectory_types or any(t not in ALL_TRAJECTORY_TYPES for t in trajectory_types):
        raise ValueError("Supply known trajectory types.")
    if not math.isfinite(post_encounter_time_s) or post_encounter_time_s < 0:
        raise ValueError("Post-encounter observation time must be finite and nonnegative.")
    rng = random.Random(seed)
    results = []
    shared_controller = None
    if controller == "spiking-pid" and reuse_spiking_runtime:
        from spiking_controller import SpikingController
        shared_controller = SpikingController(spiking_config)
    for i in range(num_trials):
        if shared_controller is not None:
            shared_controller.begin_trial()
        kind = trajectory_types[i % len(trajectory_types)]
        speed = rng.uniform(speed_min, speed_max)
        perception_seed = rng.getrandbits(64)
        trajectory = make_trajectory(kind, speed)
        if post_encounter_time_s:
            from dataclasses import replace
            trajectory = replace(trajectory, duration_s=trajectory.duration_s + post_encounter_time_s)
        simulation = SimulationCore(trajectory=trajectory, urgency=urgency,
            ttc_threshold=ttc_threshold, perception=PerceptionModel(perception_config, perception_seed),
            threat_gate=threat_gate, reflex_release=reflex_release,
            controller=shared_controller or controller, pid_config=pid_config, spiking_config=spiking_config)
        try:
            metrics = evaluate_dynamic_encounter(simulation, geometry_dt,
                                                intrusion_clearance, reaction_threshold)
        except BaseException:
            if shared_controller is not None:
                shared_controller.close()
            raise
        finally:
            if shared_controller is None and getattr(simulation.controller, "is_spiking_controller", False):
                simulation.controller.close()
        diagnostics = simulation.diagnostic_summary
        results.append(DynamicTrialResult(f"D{i + 1:04}", kind, speed, perception_seed,
            simulation.result, metrics, diagnostics, simulation.control_diagnostics,
            simulation.release_metrics))
    if shared_controller is not None:
        shared_controller.close()
    return results


def write_dynamic_csv(path, results, configuration):
    """SI units; trial, summary and configuration rows share a rectangular CSV."""
    metric_fields = [f.name for f in fields(EncounterMetrics)]
    diagnostic_fields = [f.name for f in fields(TrialDiagnostics)]
    columns = ["Row_Type", "Case_ID", "trajectory_type", "speed_m_s", "perception_seed", "Result"]
    columns += metric_fields + diagnostic_fields + [f.name for f in fields(ReflexEncounterMetrics)]
    columns += [f.name for f in fields(StateActivationMetrics)]
    columns += ["Metric", "Value", "Unit", "Numerator", "Denominator"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in results:
            row = {"Row_Type": "trial", "Case_ID": r.case_id,
                   "trajectory_type": r.trajectory_type, "speed_m_s": r.speed_m_s,
                   "perception_seed": r.perception_seed, "Result": r.result,
                   **asdict(r.metrics), **asdict(r.diagnostics),
                   **(asdict(r.release_metrics) if r.release_metrics is not None else {}),
                   **asdict(r.state_activation_metrics)}
            writer.writerow({k: int(v) if isinstance(v, bool) else v for k, v in row.items()})
        writer.writerows({"Row_Type": "summary", **r} for r in summarize_results(results))
        writer.writerows({"Row_Type": "summary", **r} for r in summarize_state_activation_metrics(results))
        for key, value in configuration.items():
            writer.writerow({"Row_Type": "configuration", "Metric": key, "Value": value})


def write_tick_csv(path, results):
    base = ["Case_ID", "time_s", "reflex_active", "perception_available", "shoulder_angle_deg",
            "elbow_angle_deg", "shoulder_error_deg", "elbow_error_deg"]
    for joint in ("shoulder", "elbow"):
        base += [f"requested_{joint}_velocity_deg_s", f"applied_{joint}_velocity_deg_s",
                 f"{joint}_velocity_limit_deg_s", f"{joint}_saturated"]
    base += ["true_" + f.name for f in fields(TrueBallState)] + ["true_ttc_s"]
    base += ["perceived_" + f.name for f in fields(PerceivedState)]
    base += [f.name for f in fields(GateDecision)]
    base += [f.name for f in fields(ReleaseDecision)]
    if any("pid_mode" in tick for r in results for tick in r.ticks):
        base += PID_FIELDS
    if any("spiking_backend" in tick for r in results for tick in r.ticks):
        from spiking_controller import SPIKING_FIELDS
        base += SPIKING_FIELDS
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base)
        writer.writeheader()
        for r in results:
            for tick in r.ticks:
                row = {"Case_ID": r.case_id, **tick}
                writer.writerow({k: int(v) if isinstance(v, bool) else v for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=24)
    parser.add_argument("--trajectory", nargs="+", choices=ALL_TRAJECTORY_TYPES, default=TRAJECTORY_TYPES)
    parser.add_argument("--speed-min", type=float, default=0.3)
    parser.add_argument("--speed-max", type=float, default=2.0)
    parser.add_argument("--position-noise", type=float, default=0.0, help="Independent XY Gaussian sigma in metres.")
    parser.add_argument("--ttc-noise", type=float, default=0.0, help="Relative Gaussian TTC sigma.")
    parser.add_argument("--latency", type=float, default=0.0, help="Buffered observation delay in seconds.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Acquisition dropout probability per tick.")
    parser.add_argument("--hold-timeout", type=float, default=0.06)
    parser.add_argument("--outlier-probability", type=float, default=0.0)
    parser.add_argument("--outlier-multiplier", type=float, default=5.0)
    parser.add_argument("--velocity-noise", type=float, default=0.0, help="Independent XYZ velocity sigma (m/s).")
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--velocity-sign-flip", type=float, default=0.0, help="Whole-vector flip probability per acquisition.")
    parser.add_argument("--burst-start", type=float, default=0.0)
    parser.add_argument("--burst-duration", type=float, default=0.0, help="Fresh-acquisition blackout in seconds.")
    parser.add_argument("--urgency", type=float, default=1.0, help="Fixed input urgency; existing mapping retained.")
    parser.add_argument("--ttc-threshold", type=float, default=1.0)
    parser.add_argument("--geometry-dt", type=float, default=GEOMETRY_DT)
    parser.add_argument("--intrusion-clearance", type=float, default=INTRUSION_CLEARANCE)
    parser.add_argument("--reaction-threshold", type=float, default=REACTION_THRESHOLD)
    parser.add_argument("--threat-gate", choices=("legacy", "geometric", "geometric_memoryless",
                        "geometric_hysteresis"), default="legacy")
    parser.add_argument("--gate-safety-margin", "--gate-enter-margin", dest="gate_safety_margin",
                        type=float, default=0.03,
                        help="Additional predicted surface clearance envelope in metres (default: 0.03).")
    parser.add_argument("--prediction-horizon", type=float, default=1.0,
                        help="Future obstacle-path prediction interval in seconds (default: 1.0).")
    parser.add_argument("--gate-release-margin", type=float, default=0.06)
    parser.add_argument("--release-dwell", type=float, default=0.04, help="Safe-release dwell in seconds.")
    parser.add_argument("--release-motion", choices=("none", "receding"), default="receding")
    parser.add_argument("--post-encounter-time", type=float, default=0.0,
                        help="Extra true-path observation seconds; defaults preserve old horizons.")
    parser.add_argument("--csv", type=Path, default=Path("dynamic_metrics.csv"))
    parser.add_argument("--ticks-csv", type=Path, help="Optional per-control-tick diagnostics.")
    args = parser.parse_args(argv)
    try:
        if args.ticks_csv and args.csv.resolve() == args.ticks_csv.resolve():
            raise ValueError("Trial and tick CSV paths must differ.")
        config = PerceptionConfig(args.position_noise, args.ttc_noise, args.latency,
            args.dropout, args.hold_timeout, args.outlier_probability, args.outlier_multiplier,
            args.velocity_noise, args.velocity_scale, args.velocity_sign_flip,
            args.burst_start, args.burst_duration)
        gate = ThreatGateConfig(args.threat_gate, args.gate_safety_margin, args.prediction_horizon)
        release = ReflexReleaseConfig(args.gate_release_margin, args.release_dwell, args.release_motion)
        results = run_dynamic_cases(args.num_trials, args.seed, config, args.speed_min,
            args.speed_max, args.trajectory, args.urgency, args.ttc_threshold,
            args.geometry_dt, args.intrusion_clearance, args.reaction_threshold, gate, release,
            args.post_encounter_time, args.controller, config_from_args(args),
            spiking_config_from_args(args) if args.controller == "spiking-pid" else None)
        configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        configuration.update(asdict(config))
        if args.controller == "pid":
            configuration["pid_config"] = str(asdict(config_from_args(args)))
        if args.controller == "spiking-pid":
            configuration["spiking_config"] = str(asdict(spiking_config_from_args(args)))
        configuration.update({"gate_" + k: v for k, v in asdict(gate).items()})
        configuration.update({"release_" + k: v for k, v in asdict(release).items()})
        configuration.update(control_dt=CONTROL_DT, ball_radius=BALL_RADIUS, arm_radius=ARM_RADIUS,
            link_1_length=L1, link_2_length=L2, shoulder_max=SHOULDER_MAX, elbow_max=ELBOW_MAX,
            neutral_shoulder=NEUTRAL_SHOULDER, neutral_elbow=NEUTRAL_ELBOW)
        write_dynamic_csv(args.csv, results, configuration)
        if args.ticks_csv:
            write_tick_csv(args.ticks_csv, results)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    summary = {r["Metric"]: r["Value"] for r in summarize_results(results)}
    print(f"Trials: {len(results)}; collisions: {summary['collisions']}; "
          f"false triggers: {summary['false_triggers']}/{summary['baseline_non_threatening_cases']}")
    print("Case   Trajectory                 Speed  Contact  Clearance(mm)  False  Saturation")
    for r in results:
        ratio = r.diagnostics.saturation_ratio
        print(f"{r.case_id}  {r.trajectory_type:26} {r.speed_m_s:5.2f}  {int(r.metrics.contact):7}  "
              f"{r.metrics.min_clearance_m * 1000:13.3f}  {int(r.metrics.false_trigger):5}  "
              + (f"{ratio:.1%}" if ratio is not None else "NA"))
    print(f"CSV: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
