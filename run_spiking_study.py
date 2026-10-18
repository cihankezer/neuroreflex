"""Ordered A--F neural-equivalence study; calibration uses seed-17 D only."""

import argparse
from dataclasses import asdict, replace
import hashlib
import gzip
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.signal import lfilter

from motor_controller import JOINTS, JOINT_BOUNDS, PIDController, load_conventional_reference
from motor_metrics import tracking_metrics
from run_pid_study import PROFILES, suite, verify_matched_inputs, write_rows
from run_pid_final_validation import verify_archived_results, trial_row, aggregate
from run_dynamic_cases import run_dynamic_cases, write_dynamic_csv, write_tick_csv
from reflex_release import ReflexReleaseConfig
from threat_gate import ThreatGateConfig
from spiking_config import SPIKING_PRESET, load_spiking_config
from spiking_controller import run_spike_batch, SpikingController
from simulation_core import CONTROL_DT, clamp

OUT = Path("benchmark_results/spiking")
MODE = {"NOMINAL": 0, "REFLEX_ACTIVE": 1, "PERCEPTION_HOLD": 2}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def signal_metrics(target, actual, limits=None, dt=.02, keys=None):
    target, actual = np.asarray(target), np.asarray(actual)
    residual = actual-target
    tc, ac = target-target.mean(), actual-actual.mean()
    denominator = np.linalg.norm(tc)*np.linalg.norm(ac)
    correlation = float(tc@ac/denominator) if denominator else None
    # Positive lag means the neural signal follows the target; max +/- 100 ms.
    candidates = []
    for lag in range(-5, 6):
        x = target[max(0, -lag):len(target)-max(0, lag)]
        y = actual[max(0, lag):len(actual)-max(0, -lag)]
        if keys is not None:
            identity = np.asarray(keys)
            valid = identity[max(0,-lag):len(target)-max(0,lag)] == identity[max(0,lag):len(actual)-max(0,-lag)]
            x, y = x[valid], y[valid]
        if len(x):
            candidates.append((float(np.mean((x-y)**2)), lag))
    best = min(candidates)
    result = {"samples": len(target), "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))), "max_error": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)), "correlation": correlation,
        "target_rms": float(np.sqrt(np.mean(target**2))), "actual_rms": float(np.sqrt(np.mean(actual**2))),
        "target_peak": float(np.max(np.abs(target))), "actual_peak": float(np.max(np.abs(actual))),
        "sign_disagreement_fraction": float(np.mean(np.sign(target) != np.sign(actual))),
        "lag_ms": best[1]*dt*1000 if denominator else None,
        "lag_adjusted_rmse": float(np.sqrt(best[0])) if denominator else None}
    if limits is not None:
        limits = np.asarray(limits)
        result["saturation_status_disagreement_fraction"] = float(np.mean((np.abs(target)>limits) != (np.abs(actual)>limits)))
        result["saturation_transition_difference_count"] = int(abs(np.count_nonzero(np.diff(np.abs(target)>limits))-
                                                                     np.count_nonzero(np.diff(np.abs(actual)>limits))))
    return result


def source_hashes():
    from simulation_core import student_controller, nominal_controller
    return {"conventional_reference_sha256": hashlib.sha256(Path("conventional_reference.json").read_bytes()).hexdigest(),
        "pid_class_sha256": hashlib.sha256(inspect.getsource(PIDController).encode()).hexdigest(),
        "legacy_p_source_sha256": hashlib.sha256((inspect.getsource(student_controller)+inspect.getsource(nominal_controller)).encode()).hexdigest()}


def collect(out, trials):
    """Recreate frozen references, verify archived science, persist complete inputs."""
    all_rows, trace_rows, metadata = [], [], []
    for seed in (17, 23, 41):
        for profile, results in suite("pid", load_conventional_reference(), seed, trials):
            print(f"Reference traces: {seed}/{profile}/{trials}", flush=True)
            verify_archived_results(results, "pid", profile, seed)
            for result in results:
                key = f"{seed}/{profile}/{result.case_id}"
                metadata.append({"key": key, "seed": seed, "profile": profile, "case_id": result.case_id,
                    "duration_s": result.metrics.duration_s, "trajectory": result.trajectory_type,
                    "speed_m_s": result.speed_m_s, "perception_seed": result.perception_seed})
                trace_rows.extend({"key": key, **tick} for tick in result.ticks)
                all_rows.append(trial_row(result, seed, profile, "pid", .04))
    write_rows(out/"reference_trials.csv", all_rows)
    # Lossless float JSON preserves all original diagnostics and replay targets.
    # Explicit string tags preserve unbounded TTC/geometry diagnostics in valid JSON.
    for row in trace_rows:
        for field, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[field] = "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    with gzip.open(out/"reference_traces.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"trials": metadata, "ticks": trace_rows, "source_hashes": source_hashes()}, stream, allow_nan=False)
    return trace_rows, metadata


def trace_arrays(rows):
    errors = np.array([[row.get(f"{j}_error_deg") or 0 for j in JOINTS] for row in rows])
    modes = np.array([MODE[row["reflex_state"]] for row in rows])
    return errors, modes


def neural_trials(rows, config):
    """One batched Lava actor with explicit reset between independent trials."""
    errors, modes = trace_arrays(rows)
    # Insert hold intervals between trials so graph state cannot cross boundaries.
    expanded_errors, expanded_modes, keep = [], [], []
    previous = None
    for k, row in enumerate(rows):
        if row["key"] != previous:
            expanded_errors.append([0., 0.]); expanded_modes.append(2)
        keep.append(len(expanded_modes))
        expanded_errors.append(errors[k]); expanded_modes.append(modes[k])
        previous = row["key"]
    batch = run_spike_batch(expanded_errors, expanded_modes, config)
    steps = round(.02/config.neural_dt_s)
    batch["history"] = batch["history"][keep]
    batch["input_clipped"] = batch["input_clipped"][keep]
    batch["raw_projection"] = batch["raw_projection"].reshape(-1, steps, 2)[keep]
    return batch


def stage_ab(out, config):
    # Each plateau includes 200 ms transient and 800 ms stationary observation.
    levels = (-1., -.75, -.25, 0., .25, .75, 1., 1.1)
    errors = np.repeat(np.array(levels)[:, None]*np.array(config.error_radius_deg)[None, :], 50, axis=0)
    mode = np.ones(len(errors))
    batch = run_spike_batch(errors, mode, config)
    stationary = np.tile(np.arange(50)>=10, len(levels))
    target = np.clip(errors, -np.array(config.error_radius_deg), config.error_radius_deg)
    rows = []
    for i, j in enumerate(JOINTS):
        rows.append({"stage": "A", "joint": j, "units": "deg",
            **signal_metrics(target[stationary, i], batch["history"][stationary, i, 0]),
            "input_clipped_intervals": int(batch["input_clipped"][:, i].sum()),
            "mean_firing_rate_hz": float(batch["history"][:, i, 8].mean()),
            "peak_firing_rate_hz": float(batch["history"][:, i, 9].max())})
    t = np.arange(600)*.02
    signals = np.column_stack((25*np.sin(2*np.pi*.6*t), -20*np.sin(2*np.pi*.6*t)))
    signals[100:180] = (15, -15); signals[300:380] = (-30, 30)
    for label, code, kp in (("avoidance", 1, 5.25), ("nominal", 0, 2.5)):
        response = run_spike_batch(signals, np.full(len(t), code), config)
        for i, joint in enumerate(JOINTS):
            rows.append({"stage": "B", "mode": label, "joint": joint, "units": "deg/s",
                         **signal_metrics(kp*signals[:, i], response["history"][:, i, 1])})
    write_rows(out/"stage_ab.csv", rows)
    np.savez_compressed(out/"representation.npz", errors=errors, history=batch["history"], clipping=batch["input_clipped"])
    return rows


def dual_features(raw, modes, keys, config, short, long):
    """Offline synapse-only calibration driven by ARCHIVED ACTUAL Lava spikes."""
    steps = raw.shape[1]
    result = np.zeros((len(raw), 2))
    previous = None
    ash, al, ap = (np.exp(-config.neural_dt_s/t) for t in (short, long, config.p_synapse_tau_s))
    for k in range(len(raw)):
        reset = k == 0 or keys[k] != keys[k-1] or modes[k] != modes[k-1]
        if modes[k] == 2:
            previous = None
            continue
        if reset or previous is None:
            p = np.zeros(2)
            for spike_projection in raw[k]:
                p = ap*p+(1-ap)*spike_projection
            state_s = state_l = p.copy()
            previous = modes[k]
        else:
            short_values, _ = lfilter([1-ash], [1, -ash], raw[k], axis=0, zi=(ash*state_s)[None, :])
            long_values, _ = lfilter([1-al], [1, -al], raw[k], axis=0, zi=(al*state_l)[None, :])
            state_s, state_l = short_values[-1], long_values[-1]
            result[k] = np.mean(short_values-long_values,axis=0)*config.error_radius_deg
    return result


def calibrate_d(out, config, rows, batch):
    _, modes = trace_arrays(rows)
    keys = [r["key"] for r in rows]
    target = np.array([[r.get(f"{j}_d_deg_s") or 0 for j in JOINTS] for r in rows])
    train = np.array([r["key"].startswith("17/") for r in rows])
    candidates, features = [], {}
    for short in (.002, .005, .01):
        for long in (.04, -.02/math.log(2/3), .06):
            feature = dual_features(batch["raw_projection"], modes, keys, config, short, long)
            denominator = np.sum(feature[train]**2)
            scale = float(np.sum(feature[train]*target[train]) / (.1*denominator)) if denominator else 25.
            # Positive derivative scaling only; predetermined physical range.
            scale = float(np.clip(scale, 1., 100.))
            prediction = .1*scale*feature
            row = {"tau_short_s": short, "tau_long_s": long, "derivative_scale_per_s": scale,
                   "training_rmse_deg_s": float(np.sqrt(np.mean((prediction[train]-target[train])**2)))}
            for seed in (23, 41):
                mask = np.array([r["key"].startswith(f"{seed}/") for r in rows])
                row[f"validation_seed{seed}_rmse_deg_s"] = float(np.sqrt(np.mean((prediction[mask]-target[mask])**2)))
            candidates.append(row); features[(short, long)] = feature
    selected = min(candidates, key=lambda row: row["training_rmse_deg_s"])
    write_rows(out/"derivative_calibration.csv", candidates)
    calibrated = replace(config, tau_short_s=selected["tau_short_s"], tau_long_s=selected["tau_long_s"],
                         derivative_scale_per_s=selected["derivative_scale_per_s"])
    save_json(out/"selected_spiking.json", {"name": "nef-spiking-reference", "status": "seed17_D_signal_calibrated",
        "selection": "Minimum seed-17 joint-pooled D-contribution RMSE; no safety outcomes used.",
        "training_seed": 17, "validation_seeds": [23, 41], "config": asdict(calibrated),
        "conventional_reference_sha256": source_hashes()["conventional_reference_sha256"]})
    # Check offline calibration is an exact rearrangement of the graph's linear synapses.
    base = dual_features(batch["raw_projection"], modes, keys, config, config.tau_short_s, config.tau_long_s)
    mismatch = float(np.max(np.abs(.1*config.derivative_scale_per_s*base-batch["history"][:, :, 4])))
    if mismatch > 1e-8:
        raise AssertionError(f"Calibration/graph synapse mismatch: {mismatch}")
    print(f"D calibration selected {selected}; graph equivalence error {mismatch:.3g}", flush=True)
    return calibrated, mismatch


def canonical_d(errors, modes):
    controller = PIDController(load_conventional_reference())
    result = []
    for error, code in zip(errors, modes):
        controller.prepare(("NOMINAL", "REFLEX_ACTIVE", "PERCEPTION_HOLD")[int(code)], (98, 130))
        if code == 2:
            controller.hold_diagnostics(); result.append([0., 0.])
        else:
            _, details = controller.update(error, .02, (98, 130))
            result.append([details[f"{j}_d_deg_s"] for j in JOINTS])
    return np.asarray(result)


def stage_c_synthetic(out, config):
    t = np.arange(300)*.02
    sequences = {"constant": np.full(300, 15.), "step": np.where(t<1, 0., 30.),
        "ramp": np.minimum(t*8, 40), "sinusoid": 25*np.sin(2*np.pi*.8*t),
        "reversal": np.where(t<1, 0., np.where(t<3, 30., -30.))}
    rows, diagnostics = [], []
    for label, signal in sequences.items():
        errors = np.column_stack((signal, -signal))
        modes = np.ones(len(signal))
        reference = canonical_d(errors, modes)
        batch = run_spike_batch(errors, modes, config)
        for i, joint in enumerate(JOINTS):
            rows.append({"stage": "C", "sequence": label, "joint": joint,
                **signal_metrics(reference[:, i], batch["history"][:, i, 4]),
                "tail_residual_std_deg_s": float(np.std(batch["history"][-50:, i, 4]-reference[-50:,i]))})
        diagnostics.extend({"sequence": label, "tick": k, "joint": j,
            "input_error_deg": float(errors[k,i]), "reference_d_deg_s": float(reference[k,i]),
            "short_error_deg": float(batch["history"][k,i,2]), "long_error_deg": float(batch["history"][k,i,3]),
            "spiking_d_deg_s": float(batch["history"][k,i,4])}
            for k in range(len(t)) for i,j in enumerate(JOINTS))
    write_rows(out/"stage_c_synthetic.csv", rows)
    write_rows(out/"derivative_synthetic_ticks.csv", diagnostics)
    return rows


def stage_cd_recorded(out, config, rows):
    batch = neural_trials(rows, config)
    stats, diagrows = [], []
    for seed in (17, 23, 41):
        mask = np.array([r["key"].startswith(f"{seed}/") for r in rows])
        for i,j in enumerate(JOINTS):
            for stage, suffix, index in (("C", f"{j}_d_deg_s", 4), ("D", f"requested_{j}_velocity_deg_s", 5)):
                reference = np.array([r.get(suffix) or 0. for r in rows])[mask]
                limits = np.array([r[f"{j}_velocity_limit_deg_s"] for r in rows])[mask]
                stats.append({"stage": stage, "seed": seed, "joint": j, "units": "deg/s",
                    **signal_metrics(reference, batch["history"][mask,i,index], limits if stage == "D" else None,
                                     keys=[r["key"] for r, keep in zip(rows,mask) if keep]),
                    "input_clipped_intervals": int(batch["input_clipped"][mask,i].sum()),
                    "error_spikes": int(batch["history"][mask,i,6].sum()),
                    "output_spikes": int(batch["history"][mask,i,7].sum()),
                    "mean_firing_rate_hz": float(batch["history"][mask,i,8].mean()),
                    "peak_firing_rate_hz": float(batch["history"][mask,i,9].max()),
                    # Dense conceptual fanout: error neuron -> N output neurons x P/short/long.
                    # Actual fused CPU uses factorization and fewer operations. No hardware energy inference.
                    "conceptual_dense_synaptic_events": int(batch["history"][mask,i,6].sum())*config.neurons_per_ensemble*3})
    from lava_nef import DIAG
    for k,row in enumerate(rows):
        for i,j in enumerate(JOINTS):
            reference = row.get(f"requested_{j}_velocity_deg_s") or 0.
            diagrows.append({"key": row["key"], "time_s": row["time_s"], "joint": j,
                "input_error_deg": row.get(f"{j}_error_deg"), "normalized_error": float(np.clip((row.get(f"{j}_error_deg") or 0)/config.error_radius_deg[i],-1,1)),
                "input_clipped": bool(batch["input_clipped"][k,i]), "mode": row["reflex_state"],
                "reference_requested_velocity_deg_s": reference,
                "command_residual_deg_s": float(batch["history"][k,i,5])-reference,
                **{name: float(batch["history"][k,i,d]) for d,name in enumerate(DIAG)}})
    write_rows(out/"stage_cd_recorded.csv", stats)
    write_rows(out/"open_loop_ticks.csv", diagrows)
    return stats


def replay_one(source, config, spiking, existing_controller=None):
    """Same targets/modes/limits and direct-velocity plant as existing replay."""
    controller = existing_controller or (SpikingController(config) if spiking else PIDController(load_conventional_reference()))
    if existing_controller is not None:
        controller.begin_trial()
    q, ticks = [35.,55.], []
    try:
        for row in source:
            mode = row["reflex_state"]
            limits = [row[f"{j}_velocity_limit_deg_s"] for j in JOINTS]
            controller.prepare(mode, limits)
            if mode == "PERCEPTION_HOLD":
                controller.hold_diagnostics()
                continue  # Same unchanged pose/no target as the existing replay.
            target = [row[f"{j}_angle_deg"]+row[f"{j}_error_deg"] for j in JOINTS]
            requests, details = controller.update([t-p for t,p in zip(target,q)], .02, limits, q)
            tick = {"time_s": row["time_s"], "reflex_active": mode=="REFLEX_ACTIVE", **details}
            for i,j in enumerate(JOINTS):
                velocity = clamp(requests[i], -limits[i], limits[i])
                tick.update({f"{j}_angle_deg": q[i], f"{j}_saturated": abs(requests[i])>limits[i],
                             f"applied_{j}_velocity_deg_s": velocity})
                q[i] = clamp(q[i]+velocity*.02, *JOINT_BOUNDS[i])
            ticks.append(tick)
    finally:
        if spiking and existing_controller is None:
            controller.close()
    return ticks


def stage_e(out, config, rows, metadata):
    from collections import defaultdict
    bykey = defaultdict(list)
    for row in rows:
        bykey[row["key"]].append(row)
    summaries, commands = [], []
    shared_controller=SpikingController(config)
    try:
        for trial in metadata:
            # Primary replay held-out seeds across all three profiles and all nine paths.
            if trial["seed"] == 17:
                continue
            ticks = {}
            for name, spiking in (("pid", False), ("spiking-pid", True)):
                ticks[name] = replay_one(bykey[trial["key"]], config, spiking,
                                         shared_controller if spiking else None)
                summaries.append({"controller": name, **trial,
                    **tracking_metrics(ticks[name], trial["duration_s"])})
            for i,j in enumerate(JOINTS):
                reference = np.array([t[f"requested_{j}_velocity_deg_s"] for t in ticks["pid"]])
                actual = np.array([t[f"requested_{j}_velocity_deg_s"] for t in ticks["spiking-pid"]])
                commands.append({**trial, "joint": j, "signal": "requested", **signal_metrics(reference, actual)})
                commands.append({**trial, "joint": j, "signal": "applied", **signal_metrics(
                    [t[f"applied_{j}_velocity_deg_s"] for t in ticks["pid"]],
                    [t[f"applied_{j}_velocity_deg_s"] for t in ticks["spiking-pid"]])})
            if trial["case_id"] == "D0001":
                print(f"Target replay {trial['seed']}/{trial['profile']}", flush=True)
    finally:
        shared_controller.close()
    write_rows(out/"target_replay.csv", summaries)
    write_rows(out/"target_replay_commands.csv", commands)
    return summaries


def stage_f(out, config, trials):
    rows, pairs = [], []
    for seed in (23,41):
        for profile, gate, perception in PROFILES:
            print(f"Closed loop {seed}/{profile}/{trials}", flush=True)
            common = dict(num_trials=trials, seed=seed, trajectory_types=__import__("trajectories").ALL_TRAJECTORY_TYPES,
                perception_config=perception, threat_gate=ThreatGateConfig(mode=gate),
                reflex_release=ReflexReleaseConfig(), post_encounter_time_s=1.)
            reference = run_dynamic_cases(**common, controller="pid", pid_config=load_conventional_reference())
            verify_archived_results(reference, "pid", profile, seed)
            neural = run_dynamic_cases(**common, controller="spiking-pid", spiking_config=config,
                                       reuse_spiking_runtime=True)
            verify_matched_inputs(neural, reference)
            for name, results in (("pid", reference), ("spiking-pid", neural)):
                write_dynamic_csv(out/f"{name}_{profile}_seed{seed}.csv", results,
                    {"spiking_config": json.dumps(asdict(config)), "matched_inputs_verified": True, "seed": seed})
                rows.extend(trial_row(r, seed, profile, name, .04) for r in results)
            if seed==23 and profile=="hysteresis_corrupt":
                write_tick_csv(out/"spiking_ticks.csv", neural)
            for ref, neu in zip(reference, neural):
                pairs.append({"seed": seed,"profile": profile,"case_id":ref.case_id,
                    "pd_contact": ref.metrics.contact,"spiking_contact":neu.metrics.contact,
                    "pd_clearance_mm":1000*ref.metrics.min_clearance_m,
                    "spiking_clearance_mm":1000*neu.metrics.min_clearance_m,
                    "clearance_change_mm":1000*(neu.metrics.min_clearance_m-ref.metrics.min_clearance_m),
                    "pd_state_activation_false_trigger":ref.state_activation_metrics.state_activation_false_trigger,
                    "spiking_state_activation_false_trigger":neu.state_activation_metrics.state_activation_false_trigger})
    write_rows(out/"closed_loop_trials.csv",rows)
    write_rows(out/"closed_loop_pairs.csv",pairs)
    summary=[]
    for controller in ("pid","spiking-pid"):
        for profile in ("all",)+tuple(p[0] for p in PROFILES):
            group=[r for r in rows if r["controller"]==controller and (profile=="all" or r["profile"]==profile)]
            summary.append({"controller":controller,"profile":profile,**aggregate(group),
                "missed_threats":sum(r["baseline_contact"] and r["first_reflex_time_s"] is None for r in group),
                "late_threats":sum(r["baseline_contact"] and r["reflex_before_baseline_contact"] is False for r in group),
                "reflex_active_time_s":sum(r["reflex_active_time_s"] for r in group)})
    write_rows(out/"closed_loop_summary.csv",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect","representation","signals","open-loop","replay","closed-loop","all"),default="signals")
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--trials",type=int,default=36)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.trials < 9:
        parser.error("Use >=9 trials to cover all nine existing trajectories.")
    started=time.perf_counter()
    initial_hashes=source_hashes()
    if args.stage in ("collect","all"):
        rows,metadata=collect(args.out,args.trials)
    else:
        with gzip.open(args.out/"reference_traces.json.gz", "rt", encoding="utf-8") as stream:
            document=json.load(stream)
        rows,metadata=document["ticks"],document["trials"]
        if document["source_hashes"] != initial_hashes:
            raise AssertionError("Frozen reference sources differ from trace provenance.")
    config=load_spiking_config()
    if args.stage == "representation":
        stage_ab(args.out, config)
    if args.stage in ("signals","all"):
        # Always reproduce the analytic starting mapping even after preset publication.
        config=replace(config,tau_short_s=.005,tau_long_s=-.02/math.log(2/3),derivative_scale_per_s=25.)
        save_json(args.out/"initial_spiking.json",{"config":asdict(config),"readout":"20ms mean"})
        print("Stage A/B",flush=True); stage_ab(args.out,config)
        print("Stage C spike collection",flush=True); batch=neural_trials(rows,config)
        config,mismatch=calibrate_d(args.out,config,rows,batch)
        print("Stage C synthetic",flush=True); stage_c_synthetic(args.out,config)
        print("Stage C/D recorded",flush=True); stage_cd_recorded(args.out,config,rows)
    if args.stage in ("replay","closed-loop","all"):
        if not (args.out/"stage_cd_recorded.csv").exists():
            raise RuntimeError("Run and review A--D signal validation before replay or closed loop.")
        config=load_spiking_config(args.out/"selected_spiking.json")
    if args.stage == "open-loop":
        config=load_spiking_config(args.out/"selected_spiking.json")
        stage_c_synthetic(args.out,config)
        stage_cd_recorded(args.out,config,rows)
    if args.stage in ("replay","all"):
        print("Stage E",flush=True); stage_e(args.out,config,rows,metadata)
    if args.stage in ("closed-loop","all"):
        if not (args.out/"target_replay.csv").exists():
            raise RuntimeError("Run and review Stage E target replay before closed loop.")
        print("Stage F",flush=True); stage_f(args.out,config,args.trials)
    if initial_hashes != source_hashes():
        raise AssertionError("Frozen controller source changed during experiment.")
    save_json(args.out/f"provenance_{args.stage}.json",{"stage":args.stage,"source_hashes":initial_hashes,
        "spiking_config":asdict(config),"trials_per_profile":args.trials,"elapsed_wall_s":time.perf_counter()-started})
    print(f"Completed {args.stage} in {time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":
    main()
