"""Final fixed-gain filter validation and conventional-reference freeze evidence.

No gain search. Reuse the original three profiles, three seeds, nine paths,
36 trials/profile, 20 ms controller and 2 ms geometry sampling unchanged.
"""

import argparse
import csv
from dataclasses import asdict, fields, replace
import hashlib
import json
import math
from pathlib import Path

from motor_controller import CONVENTIONAL_REFERENCE_PATH, JOINTS, load_conventional_reference
from motor_metrics import tracking_metrics
from reflex_release import ReflexReleaseConfig
from run_dynamic_cases import write_dynamic_csv, write_tick_csv
from run_pid_study import PROFILES, characterize, flatten, suite, verify_matched_inputs, write_rows
from threat_gate import ThreatGateConfig
from simulation_core import CONTROL_DT


STUDY = Path(__file__).resolve().parent / "benchmark_results/pid"
TAUS = (.02, .04, .08)
SEEDS = (17, 23, 41)


def trial_row(result, seed, profile, controller, tau=None):
    return {"seed": seed, "profile": profile, "controller": controller,
            "tau_s": tau, "case_id": result.case_id, "trajectory": result.trajectory_type,
            "speed_m_s": result.speed_m_s, "perception_seed": result.perception_seed,
            **asdict(result.metrics), **asdict(result.diagnostics),
            **asdict(result.state_activation_metrics),
            **tracking_metrics(result.ticks, result.metrics.duration_s)}


def aggregate(rows):
    active = sum(r["reflex_active_time_s"] for r in rows)
    joint_time = sum(r[f"{j}_tracking_time_s"] for r in rows for j in JOINTS)
    errors = sum(r[f"{j}_tracking_rmse_deg"] ** 2 * r[f"{j}_tracking_time_s"]
                 for r in rows for j in JOINTS)
    residual = sum(r[f"{j}_velocity_residual_rms_deg_s"] ** 2 * r["reflex_active_time_s"]
                   for r in rows for j in JOINTS if r[f"{j}_velocity_residual_rms_deg_s"] is not None)
    non_threats = sum(not r["baseline_contact"] for r in rows)
    false_trials = sum(r["state_activation_false_trigger"] for r in rows)
    return {"trials": len(rows), "collisions": sum(r["contact"] for r in rows),
            "worst_clearance_mm": 1000 * min(r["min_clearance_m"] for r in rows),
            "tracking_rmse_deg": math.sqrt(errors / joint_time),
            "peak_error_deg": max(r[f"{j}_peak_error_deg"] for r in rows for j in JOINTS),
            "reflex_saturation_ratio": sum(r["saturated_reflex_time_s"] for r in rows) / active if active else None,
            "request_applied_rms_deg_s": math.sqrt(residual / (2 * active)) if active else None,
            "state_activation_false_trigger_trials": false_trials,
            "state_activation_non_threat_trials": non_threats,
            "state_activation_false_trigger_rate": 100 * false_trials / non_threats if non_threats else None,
            "state_activation_false_trigger_events": sum(r["state_activation_false_trigger_events"] for r in rows),
            "legacy_false_trigger_trials": sum(r["false_trigger"] for r in rows)}


def verify_archived_results(results, controller, profile, seed):
    """Prove that the selected PD and P scientific records have not drifted."""
    path = STUDY / f"{controller}_{profile}_seed{seed}.csv"
    with path.open(encoding="utf-8-sig") as stream:
        previous = {r["Case_ID"]: r for r in csv.DictReader(stream) if r["Row_Type"] == "trial"}
    for result in results:
        original = previous[result.case_id]
        for record in (result.metrics, result.diagnostics, result.release_metrics):
            for f in fields(record):
                value = getattr(record, f.name)
                encoded = "" if value is None else str(int(value)) if isinstance(value, bool) else str(value)
                if original[f.name] != encoded:
                    raise AssertionError(f"Archived {controller} drift: {seed}/{profile}/{result.case_id}/{f.name}")


def save_groups(out, groups, controller, config, seed, verify_archive=False):
    rows = []
    tau = config.derivative_tau_s if controller == "pid" else None
    for profile, results in groups:
        if verify_archive:
            verify_archived_results(results, controller, profile, seed)
        _, gate, perception = next(p for p in PROFILES if p[0] == profile)
        stem = f"{controller}" + (f"_tau{tau:g}" if tau is not None else "") + f"_{profile}_seed{seed}"
        write_dynamic_csv(out / (stem + ".csv"), results,
            {"controller": controller, "preset": "conventional-reference", "seed": seed,
             "control_dt_s": CONTROL_DT, "geometry_dt_s": .002, "num_trials": 36,
             "post_encounter_time_s": 1, "config": json.dumps(asdict(config)),
             "perception_config": json.dumps(asdict(perception)),
             "threat_gate_config": json.dumps(asdict(ThreatGateConfig(mode=gate))),
             "reflex_release_config": json.dumps(asdict(ReflexReleaseConfig()))})
        rows.extend(trial_row(r, seed, profile, controller, tau) for r in results)
        if seed == 23 and profile == "hysteresis_corrupt" and (controller == "p" or tau == .04):
            path = out / f"{controller}_ticks.csv"
            write_tick_csv(path, results)
            if path.read_bytes() != (STUDY / f"{controller}_ticks.csv").read_bytes():
                raise AssertionError(f"Archived {controller} tick CSV drift.")
    return rows


def validate(out, include_filters=True):
    config = load_conventional_reference()
    taus = TAUS if include_filters else (.04,)
    all_rows, steps = [], []
    for seed in SEEDS:
        print(f"Final validation: seed={seed}, P baseline", flush=True)
        baseline = suite("p", config, seed, 36)
        all_rows.extend(save_groups(out, baseline, "p", config, seed, verify_archive=True))
        for tau in taus:
            print(f"Final validation: seed={seed}, fixed gains, tau={tau:g}", flush=True)
            variant = replace(config, derivative_tau_s=tau)
            groups = suite("pid", variant, seed, 36)
            verify_matched_inputs(flatten(groups), flatten(baseline))
            all_rows.extend(save_groups(out, groups, "pid", variant, seed, verify_archive=tau == .04))
            del groups
    by_case = {(r["controller"], r["tau_s"], r["seed"], r["profile"], r["case_id"]): r for r in all_rows}
    for row in all_rows:
        key = row["seed"], row["profile"], row["case_id"]
        p = by_case[("p", None, *key)]
        reference = by_case[("pid", .04, *key)]
        row["clearance_change_vs_p_mm"] = (row["min_clearance_m"] - p["min_clearance_m"]) * 1000
        row["clearance_change_vs_tau04_mm"] = (row["min_clearance_m"] - reference["min_clearance_m"]) * 1000
        row["state_activation_differs_from_p"] = row["state_activation_false_trigger"] != p["state_activation_false_trigger"]
        row["state_event_count_differs_from_p"] = row["state_activation_count"] != p["state_activation_count"]
    comparisons = []
    for tau in taus:
        character = characterize(replace(config, derivative_tau_s=tau), f"final_tau{tau:g}")
        steps.extend({"tau_s": tau, **r} for r in character)
        rows = [r for r in all_rows if r["controller"] == "pid" and r["tau_s"] == tau]
        comparisons.append({"tau_s": tau, **aggregate(rows),
            "worst_paired_clearance_change_vs_p_mm": min(r["clearance_change_vs_p_mm"] for r in rows),
            "worst_paired_clearance_change_vs_tau04_mm": min(r["clearance_change_vs_tau04_mm"] for r in rows),
            "max_fixed_step_overshoot_deg": max(r[f"{j}_overshoot_deg"] for r in character for j in JOINTS),
            "max_fixed_step_target_crossings": max(r[f"{j}_target_crossings_outside_tolerance"] for r in character for j in JOINTS),
            "unsettled_joint_segments": sum(r[f"{j}_settling_s"] is None for r in character for j in JOINTS)})
    false_triggers = []
    for controller, tau in (("p", None), ("pid", .04)):
        for profile in ("all", *(p[0] for p in PROFILES)):
            rows = [r for r in all_rows if r["controller"] == controller and r["tau_s"] == tau
                    and (profile == "all" or r["profile"] == profile)]
            false_triggers.append({"controller": controller, "profile": profile, **aggregate(rows)})
    regressions = []
    for row in all_rows:
        if row["controller"] != "pid" or row["tau_s"] == .04:
            continue
        key = row["seed"], row["profile"], row["case_id"]
        reference = by_case[("pid", .04, *key)]
        if ((reference["contact"] and not row["contact"])
                or (reference["min_clearance_m"] < .03 <= row["min_clearance_m"])
                or (reference["min_clearance_m"] < .9 * row["min_clearance_m"])):
            regressions.append({"seed": row["seed"], "profile": row["profile"], "case_id": row["case_id"],
                                "alternative_tau": row["tau_s"], "reference_clearance_m": reference["min_clearance_m"],
                                "alternative_clearance_m": row["min_clearance_m"]})
    reference_step = next(r for r in comparisons if r["tau_s"] == .04)
    stability_problem = bool(reference_step["unsettled_joint_segments"] or reference_step["max_fixed_step_target_crossings"] > 2)
    write_rows(out / "filter_sensitivity.csv", comparisons)
    write_rows(out / "activation_false_triggers.csv", false_triggers)
    write_rows(out / "matched_trials.csv", all_rows)
    write_rows(out / "fixed_steps.csv", steps)
    (out / "validation_checks.json").write_text(json.dumps({
        "archived_p_and_selected_pd_metrics_identical": True,
        "archived_p_and_selected_pd_tick_csvs_byte_identical": True,
        "matched_actual_input_traces_verified": True, "seeds": SEEDS, "taus_s": taus,
        "tau04_meaningful_safety_regressions": regressions, "tau04_fixed_step_stability_problem": stability_problem,
        "activation_false_trigger_differences_p_vs_pd": sum(r["state_activation_differs_from_p"] for r in all_rows
                                                           if r["controller"] == "pid" and r["tau_s"] == .04),
        "activation_event_count_differences_p_vs_pd": sum(r["state_event_count_differs_from_p"] for r in all_rows
                                                        if r["controller"] == "pid" and r["tau_s"] == .04),
        "canonical_reference_sha256": hashlib.sha256(CONVENTIONAL_REFERENCE_PATH.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"tau=0.04 safety flags: {len(regressions)}; stability flag: {stability_problem}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("filters", "frozen", "report", "all"))
    parser.add_argument("--output", type=Path, default=STUDY / "final_validation")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("filters", "frozen", "all"):
        validate(args.output, include_filters=args.stage != "frozen")
    if args.stage in ("report", "all"):
        from pid_final_report import generate_final_section
        generate_final_section(args.output, STUDY / "PID_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
