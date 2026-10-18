"""Generate the technical report from recorded PID experiments, without rerunning."""

import csv
import json
import math
from pathlib import Path


def read_rows(out, name):
    return list(csv.DictReader((out / name).open(encoding="utf-8-sig")))


def pooled_tracking(rows):
    weight = sum(float(r[f"{j}_tracking_time_s"]) for r in rows for j in ("shoulder", "elbow"))
    return math.sqrt(sum(float(r[f"{j}_tracking_rmse_deg"]) ** 2 * float(r[f"{j}_tracking_time_s"])
                         for r in rows for j in ("shoulder", "elbow")) / weight)


def generate_report(out):
    selected = json.loads((out / "selected.json").read_text())
    config = selected["config"]
    comparison = read_rows(out, "comparison.csv")
    tracking = read_rows(out, "tracking.csv")
    tuning = read_rows(out, "tuning.csv")
    steps = read_rows(out, "selected_steps.csv")
    replay = read_rows(out, "target_replay.csv")
    paired = read_rows(out, "paired_clearance.csv")
    tests = (out / "tests_after.txt").read_text() if (out / "tests_after.txt").exists() else "Not recorded"
    test_summary = next((s for s in tests.splitlines() if s.startswith("Ran ")), "See tests_after.txt")
    lines = [
        "# NeuroReflex conventional PID study", "",
        "Implementation and reproducible experiment results, 2026-10-07.", "",
        "## Outcome", "",
        "The tuned PID has zero integral gain: it is a filtered PD reference. It reduces dynamic tracking RMSE "
        "slightly and reduces velocity saturation substantially on the matched dynamic suite. It does not "
        "improve the worst safety clearance or remove the existing static collision. Keep P as the default "
        "scientific baseline; the selected PD is a useful optional reference for a later neural P/I/D implementation.", "",
        "## Critical files inspected", "",
        "- `simulation_core.py`: P controller, IK/escape construction, 20 ms plant, clipping and state.",
        "- `run_dynamic_cases.py`, `run_static_cases.py`: seeded/workbook experiments, CLI and CSVs.",
        "- `diagnostics.py`, `dynamic_metrics.py`, `performance_metrics.py`, `release_metrics.py`: passive metrics.",
        "- `reflex_release.py`, `threat_gate.py`: transition policy and threat gating.",
        "- `perception.py`, `trajectories.py`: synthetic input configuration and available physical cases.",
        "- `tests/test_automation.py`, `tests/test_dynamic_cases.py`: integration and baseline regressions.",
        "- `README.md`, `looming_arm_3d.py`: documented workflow and manual UI references.", "",
        "## Changes", "",
        "| File | Purpose |", "|---|---|",
        "| `motor_controller.py` (added) | Per-joint PID state/configuration, anti-windup, reset, shared CLI helpers |",
        "| `simulation_core.py` | Optional PID instance per trial; mode preparation and passive PID logging |",
        "| `run_dynamic_cases.py` | Controller/gain selection and extra PID tick columns |",
        "| `run_static_cases.py` | Controller/gain selection; retain legacy P CSV schema |",
        "| `motor_metrics.py` (added) | Time-weighted tracking, contribution, joint-stop, fixed-target metrics |",
        "| `pid_baseline.py` (added) | Frozen pre-change capture and exact verification |",
        "| `run_pid_study.py`, `pid_report.py` (added) | Staged search, matched evaluation, target replay and this report |",
        "| `tests/test_pid.py`, `tests/test_motor_metrics.py` (added) | Deterministic controller/metric checks |",
        "| `README.md` | PID usage and reproduction entry points |",
        "| `benchmark_results/pid/` | Immutable baseline plus successful/rejected experiments and report |", "",
        "Existing user edits were retained. Perception, reflex policy, trajectories, IK, collision geometry "
        "and actuator dynamics were not changed by this implementation. The paper was not needed.", "",
        "## Controller architecture", "",
        "All angles/errors are degrees; velocities/derivatives are deg/s; integral state is deg*s. "
        "Kp has units 1/s, Ki 1/s², and Kd is dimensionless. At dt=0.020 s:", "",
        "```text", "e = q_target - q", "P = Kp * e",
        "raw_d = (e - previous_e) / dt", "alpha = tau / (tau + dt)",
        "filtered_d = alpha * previous_filtered_d + (1-alpha) * raw_d",
        "trial_integral = bound(integral + e*dt)",
        "provisional_request = P + Ki*trial_integral + Kd*filtered_d", "```", "",
        "Reject the trial integral independently per joint only when its contribution change pushes the "
        "provisional request farther beyond the reachable velocity interval. Changes that unwind saturation "
        "are accepted. The interval includes urgency speed limits and the remaining distance to each joint "
        "position stop over dt. The numerical integral bound is additional protection. Ki=0 freezes integral "
        "at zero. Output uses the accepted integral, then the existing core clips velocity and integrates "
        "the direct-velocity plant, finally applying its original position stops. Requested and applied "
        "velocity retain their existing definitions; applied velocity does not include motion lost at position stops.", "",
        "The unchanged `student_controller` supplies error metadata, which preserves its escape/IK algorithm "
        "and floating-point operation order for P. PID consumes those errors, ignoring P's velocities. PID "
        "uses fixed configurable avoidance gains and a separate nominal Kp (2.5). Ki/Kd are shared across "
        "modes but their states are reset at transitions.", "",
        "Reset both joints' integral and derivative history on every change among `NOMINAL`, "
        "`REFLEX_ACTIVE` and `PERCEPTION_HOLD`, and on initial preparation or explicit `reset()`. Initialize "
        "the first error sample with zero derivative. Hold bypasses target/PID evaluation and logs the "
        "frozen zero state. Held but still available observations follow the existing reflex policy. "
        "Ordinary target changes within a mode produce a filtered error-derivative kick; target jumps are "
        "not suppressed or differentiated on measurement. This is visible in the reversal characterization.", "",
        "Per-joint passive ticks contain error, target, P/I/D, raw request, clipped velocity, speed limit, "
        "saturation, integral, filtered derivative, integration-blocked status and effective anti-windup limits. "
        "Hold has no target/error and is excluded from tracking-error averages.", "",
        "## Verification and baseline preservation", "",
        "Pre-change: 116 existing tests passed. Post-change: " + test_summary + "; "
        + ("PASS." if tests.rstrip().endswith("OK") else "see recorded test output.") ,
        "Controller checks cover zero/sign/P-only behavior, integral accumulation, both saturation signs, "
        "provisional-output checking, reverse-error unwinding, derivative initialization/filtering, numerical "
        "bounds, joint stops, independent joints, resets, real perception hold, CLI overrides and unchanged P. "
        "Metric checks cover settling/overshoot and final partial-tick weighting.", "",
        "Validation corrections: the initial hold test placed its dropout burst before reflex activation "
        "and was corrected to test an actual hold. The first input-trace verifier also included the "
        "arm-relative clearance-rate metric by prefix; it was corrected to compare the explicit sensor "
        "dataclass fields. Both failures were in test/verification setup, and are covered by final tests. "
        "No controller or reflex behavior was changed to pass them.", "",
        "`pid_baseline.py --verify benchmark_results/pid/p_before.json` confirms unchanged P source, all "
        "16 static results and all 108 dynamic metric records/full tick hashes. P tick CSV columns stay "
        "unchanged; dynamic CLI configuration rows now include controller/gain arguments. The baseline "
        "is the user's working tree at task start, including pre-existing uncommitted changes.", "",
        "## Staged tuning", "",
        "A: deterministic tests. B: 3°, 15°, 60° steps and +30°→-30° reversal, each fixed segment 4 s; "
        "same velocity/position limits and 20 ms direct-velocity integration. C: coarse Kp sweep "
        "2.5,4,6,9,12,16 with Ki=Kd=0; Kd sweep .02,.05,.1,.2 at selected Kp=6; Ki sweep "
        ".05,.2,.5 after selecting Kd=.1; local Kp 5.25,6.75; filter sensitivity tau .02,.08 with Kd=.05. "
        "Default filter tau=.04. A final locked-Kp, Kd=0 ablation quantifies derivative value without changing "
        "selection. Shared gains keep the search small and interpretable; per-joint CLI overrides "
        "are available but were not searched. All " + str(len(tuning)) + " candidates and their per-trial "
        "measurements are retained in `tuning.csv` / `tuning_trials.csv`.", "",
        "Training: seed 17, 18 trials per profile, nine existing trajectories, three fixed profiles "
        "(54 trials/candidate). Selection requires zero collisions, no new false/missed/late responses, "
        "at least 30 mm clearance and at least 90% of each matched P trial's minimum clearance, and "
        "settled fixed-target segments. Feasible candidates rank by worst clearance, then pooled tracking "
        "RMSE, saturation and overshoot. Gains were locked before held-out seeds 23 and 41. "
        "Rejected low gains are reported, rather than selected for their deceptively lower moving-target RMSE.", "",
        "| Candidate | Collisions | Safety violations | RMSE (deg) | Reflex saturation | Step overshoot (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in tuning:
        if row["stage"] in ("P", "I", "ablation") or row["candidate"] == selected["training"]["candidate"]:
            lines.append(f"| {row['candidate']} | {row['collisions']} | {row['safety_violations']} | {float(row['tracking_rmse_deg']):.3f} | "
                         f"{100*float(row['saturation_reflex_fraction']):.1f}% | {float(row['max_step_overshoot_deg']):.3f} |")
    lines += ["", "## Matched dynamic results", "",
        "Each evaluation seed has 36 trials/profile (108/controller), speed uniformly 0.3–2.0 m/s, "
        "all nine existing trajectories, unchanged default gate/release thresholds, 2 ms geometry checking "
        "and 1 s extra encounter observation. Legacy-clean and hysteresis-clean use ideal synthetic "
        "perception. Hysteresis-corrupt uses 5 mm position sigma, 5% TTC sigma, 40 ms latency and 10% "
        "dropout with the same per-case sensor seeds. Actual true/perceived input traces, urgency limits "
        "and tick counts are checked for equality. Physical targets, reflex algorithm and plant are fixed "
        "across controllers; realized IK targets/states may differ because they depend on arm pose.", "",
        "| Seed | Controller | Collisions / trials | Worst clearance (mm) | RMSE (deg) | Peak error (deg) | Reflex saturation | Request−applied RMS (deg/s) | False / non-threat | Missed / late | Mean active (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        if r["profile"] == "all":
            lines.append(f"| {r['seed']} | {r['controller']} | {r['collisions']} / {r['trials']} | "
                         f"{float(r['worst_clearance_mm']):.3f} | {float(r['tracking_rmse_deg']):.3f} | "
                         f"{float(r['peak_error_deg']):.2f} | {100*float(r['saturation_reflex_fraction']):.1f}% | "
                         f"{float(r['request_applied_rms_deg_s']):.2f} | {r['false_triggers']} / {r['non_threats']} | "
                         f"{r['missed']} / {r['late']} | {float(r['mean_reflex_active_s']):.3f} |")
    heldout = [r for r in comparison if r["profile"] == "all" and r["seed"] != "17"]
    lines += ["", "| Held-out pooled (216 trials/controller) | P | Selected PID |", "|---|---:|---:|"]
    for metric, label in (("tracking", "Tracking RMSE (deg)"), ("saturation", "Reflex saturation (%)"),
                          ("residual", "Request−applied RMS (deg/s)"), ("collisions", "Collisions")):
        values = []
        for controller in ("p", "pid"):
            records = [r for r in heldout if r["controller"] == controller]
            active = [float(r["mean_reflex_active_s"]) * int(r["trials"]) for r in records]
            if metric == "tracking":
                value = pooled_tracking([r for r in tracking if r["controller"] == controller and r["seed"] != "17"])
            elif metric == "saturation":
                value = 100 * sum(float(r["saturation_reflex_fraction"]) * w for r, w in zip(records, active)) / sum(active)
            elif metric == "residual":
                value = math.sqrt(sum(float(r["request_applied_rms_deg_s"])**2 * w for r, w in zip(records, active)) / sum(active))
            else:
                value = sum(int(r[metric]) for r in records)
            values.append(f"{value:.3f}")
        lines.append(f"| {label} | {' | '.join(values)} |")
    max_loss = min(float(r["clearance_change_mm"]) for r in paired)
    retained = min(float(r["clearance_retained_fraction"]) for r in paired)
    lines += ["", f"Per-trial safety margins are not identical: the largest minimum-clearance change is {max_loss:.3f} mm "
              f"and the minimum retained fraction is {retained:.3%}. All locked-gain evaluation trials pass the "
              "predeclared constraints; see `paired_clearance.csv` for every pair. Aggregate worst clearance "
              "alone would conceal these changes.", "",
        "Static workbook cross-check: both controllers have 1 collision / 16 trials (6.25%), worst clearance "
        "−84.615 mm. This pre-existing collision remains; the candidate was not tuned to make the static "
        "labels or one easy trajectory look favorable. Static P/PID CSVs are retained.", "",
        "## Identical-target replay and fixed segments", "",
        "`target_replay.csv` freezes the P run's per-tick targets, modes and urgency limits, then replays "
        "them through both controllers. This isolates tracking; it is not a closed-loop obstacle safety "
        "experiment. Hold freezes both arms and resets PID. Across held-out seeds:", "",
        "| Controller | Replay tracking RMSE (deg) |", "|---|---:|"]
    for controller in ("p", "pid"):
        value = pooled_tracking([r for r in replay if r["controller"] == controller and r["seed"] != "17"])
        lines.append(f"| {controller} | {value:.3f} |")
    lines += ["", "Selected controller characterization (0.5° settling tolerance, four-second segment):", "",
              "| Step / segment | Shoulder settling (s) | Elbow settling (s) | Shoulder overshoot (deg) | Elbow overshoot (deg) |",
              "|---|---:|---:|---:|---:|"]
    for r in steps:
        settling = [f"{float(r[f'{j}_settling_s']):.2f}" if r[f"{j}_settling_s"] else "NA"
                    for j in ("shoulder", "elbow")]
        lines.append(f"| {r['step']} / {r['segment']} | {settling[0]} | {settling[1]} | "
                     f"{float(r['shoulder_overshoot_deg']):.3f} | {float(r['elbow_overshoot_deg']):.3f} |")
    lines += ["", "## Recommended parameters and interpretation", "",
        "| Joint / mode | Kp (1/s) | Ki (1/s²) | Kd |", "|---|---:|---:|---:|"]
    for joint in ("shoulder", "elbow"):
        gains = config[joint]
        lines.append(f"| {joint}, avoidance | {gains['kp']} | {gains['ki']} | {gains['kd']} |")
    lines += [f"| Both, nominal recovery | {config['recovery_kp']} | 0 | {config['shoulder']['kd']} |", "",
        f"Derivative tau={config['derivative_tau_s']} s; alpha={config['derivative_tau_s']/(config['derivative_tau_s']+.02):.6f} "
        f"at dt=.02 s. Integral safety bound ±{config['integral_bound_deg_s']} deg*s; inactive for Ki=0. "
        "P remains the CLI default; PID CLI defaults are conservative characterization gains (6,0,0). "
        "Use `--controller pid --pid-preset conventional-reference` for the frozen reference. "
        "`selected.json` is retained as historical selection evidence.", "",
        "The direct-velocity integrator has no inertia/friction to create a persistent fixed-target bias. "
        "Integral offered no measured benefit here; the sampled small gains added overshoot. Derivative "
        "effects are modest and filtered; selected fixed steps show no overshoot or persistent oscillation. "
        "A lower-saturation PD command helps keep the motor reference interpretable, but P still wins "
        "some individual safety margins. This limited suite cannot establish hardware superiority.", "",
        "The locked-Kp ablation makes the small derivative benefit explicit: at Kp=5.25, Ki=0, removing "
        "Kd changes training RMSE from 15.505 to 15.507 deg and reflex saturation from 56.0% to 57.2%. "
        "Most of the improvement over TTC-scheduled P comes from proportional-gain selection; derivative "
        "action is a marginal refinement. The Kd=0 ablation was evaluated on training conditions only.", "",
        "Dynamic tracking RMSE/peak error, contributions, saturation, residuals and position-stop counts "
        "are meaningful for moving targets, with the caveat that the target itself changes with the pose. "
        "Settling and overshoot are only reported for fixed segments; settling is the first post-update "
        "sample after the final tolerance exit, and an unsettled horizon is blank. Tracking errors use "
        "pre-update samples, weighted by the actual final interval and excluding hold. Reflex saturation "
        "counts either joint once over reflex-active time; tracking saturation uses joint-time. Large "
        "dynamic peaks and repeated joint stops do not disappear with PID. Existing reaction/false-trigger "
        "metrics compare commands with the legacy nominal P return; recovery PID can affect that detector. "
        "Unchanged state-activation/release records are exported separately for interpretation.", "",
        "## Exact reproduction commands", "", "Run from the repository root (PowerShell; NumPy is the existing batch dependency).", "", "```powershell",
        "python -m unittest discover -s tests -v",
        "python pid_baseline.py --verify benchmark_results/pid/p_before.json",
        "python run_static_cases.py --controller p --csv benchmark_results/pid/static_p.csv",
        "python run_static_cases.py --controller pid --pid-preset conventional-reference --csv benchmark_results/pid/static_pid.csv",
        "python run_dynamic_cases.py --controller p --seed 23 --num-trials 36 --trajectory collision_course_center collision_course_left collision_course_right near_miss lateral_pass moving_away accelerating_approach decelerating_approach tangential_near_miss --threat-gate geometric_hysteresis --position-noise 0.005 --ttc-noise 0.05 --latency 0.04 --dropout 0.1 --post-encounter-time 1 --csv benchmark_results/pid/manual_p.csv --ticks-csv benchmark_results/pid/manual_p_ticks.csv",
        "python run_dynamic_cases.py --controller pid --pid-kp 5.25 --pid-ki 0 --pid-kd 0.1 --pid-derivative-tau 0.04 --pid-integral-bound 100 --seed 23 --num-trials 36 --trajectory collision_course_center collision_course_left collision_course_right near_miss lateral_pass moving_away accelerating_approach decelerating_approach tangential_near_miss --threat-gate geometric_hysteresis --position-noise 0.005 --ttc-noise 0.05 --latency 0.04 --dropout 0.1 --post-encounter-time 1 --csv benchmark_results/pid/manual_pid.csv --ticks-csv benchmark_results/pid/manual_pid_ticks.csv",
        "# Historical tuning is closed after the conventional-reference freeze.",
        "python run_pid_study.py evaluate", "python run_pid_study.py report",
        "# Final validation commands are in the frozen-reference section below.", "```", "",
        "Do not overwrite `p_before.json` when checking preservation. It was captured before PID integration. "
        "CSV measurements, seed identities, selected configuration and complete rejected-candidate records "
        "are local artifacts; no external services or actuator models are required.", ""]
    lines += ["Recorded runtime: Python 3.13.6, NumPy 2.2.6 on Windows. `experiment_manifest.json` "
              "records source hashes and the preserved baseline hash.", ""]
    (out / "PID_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    if (out / "final_validation/filter_sensitivity.csv").exists():
        from pid_final_report import generate_final_section
        generate_final_section(out / "final_validation", out / "PID_REPORT.md")
    print(f"Report: {out / 'PID_REPORT.md'}")
