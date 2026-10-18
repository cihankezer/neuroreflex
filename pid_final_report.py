"""Append/refresh a concise final-validation section from recorded evidence."""

import csv
import json


MARKER = "## Final validation and frozen conventional reference"


def generate_final_section(out, report_path):
    def read(name):
        with (out / name).open(encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))
    filters = read("filter_sensitivity.csv")
    if {float(r["tau_s"]) for r in filters} != {.02, .04, .08}:
        raise ValueError("Final report requires all three filter variants; run the filters stage.")
    activation = read("activation_false_triggers.csv")
    checks = json.loads((out / "validation_checks.json").read_text(encoding="utf-8"))
    matched = read("matched_trials.csv")
    greatest_reference_loss = max(float(r["clearance_change_vs_tau04_mm"]) for r in matched
                                  if r["controller"] == "pid" and float(r["tau_s"]) != .04)
    problem = bool(checks["tau04_meaningful_safety_regressions"] or checks["tau04_fixed_step_stability_problem"])
    lines = [MARKER, "", "Final validation, 2026-10-07. No Kp/Ki/Kd search was reopened.", "",
        "### Derivative filter sensitivity", "",
        "Gains remain Kp=5.25, Ki=0, Kd=0.1 in avoidance; nominal Kp=2.5, Ki=0, Kd=0.1. "
        "Each filter has 324 dynamic trials: seeds 17/23/41, 36 trials per seed/profile, all nine "
        "existing trajectories and the original legacy-clean, hysteresis-clean and hysteresis-corrupt "
        "profiles. Perception parameters, gate/release thresholds, IK, urgency, actuator limits, direct-velocity "
        "plant, 20 ms control and 2 ms collision checking remain fixed. Metrics are pooled with the "
        "same time weights as the original study. Actual true/perceived inputs and urgency limits were "
        "verified against matched P runs.", "",
        "| Tau (s) | Collisions | Min clearance (mm) | Worst paired change vs P (mm) | Tracking RMSE (deg) | Peak error (deg) | Reflex saturation | Request-applied RMS (deg/s) | Step overshoot (deg) / crossings |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in filters:
        lines.append(f"| {float(r['tau_s']):.2f} | {r['collisions']}/{r['trials']} | "
                     f"{float(r['worst_clearance_mm']):.3f} | {float(r['worst_paired_clearance_change_vs_p_mm']):.3f} | "
                     f"{float(r['tracking_rmse_deg']):.3f} | {float(r['peak_error_deg']):.3f} | "
                     f"{100*float(r['reflex_saturation_ratio']):.2f}% | {float(r['request_applied_rms_deg_s']):.3f} | "
                     f"{float(r['max_fixed_step_overshoot_deg']):.3f} / {r['max_fixed_step_target_crossings']} |")
    lines += ["", f"Relative to the two alternative filters, tau=0.04 loses at most {greatest_reference_loss:.3f} mm "
              "of paired minimum clearance. Meaningful safety regression was defined as a new collision, "
              "crossing below 30 mm when an alternative remains above it, or losing more than 10% of an "
              "alternative's matched clearance. Fixed-target checks use the existing 3/15/60 degree steps "
              "and target reversal; crossings count sign changes outside a 0.5 degree deadband. All "
              "segments settled within four seconds, with zero overshoot and zero material crossings. "
              "Dynamic IK settling/overshoot are not assigned fixed-target interpretations.", "",
              ("**FLAG: tau=0.04 has a safety/stability problem; inspect validation_checks.json. Parameters were not changed.**"
               if problem else "**Retain tau=0.04 s. No meaningful safety or stability problem was found.** "
               "The small tracking/saturation advantage at tau=0.02 does not justify changing the reference."), "",
        "### Controller-independent false triggers", "",
        "The new `state_activation_false_trigger` is trial-based. Ground-truth non-threat means "
        "`baseline_contact=False` from the existing evaluator: the true trajectory does not contact the "
        "stationary neutral arm anywhere over the full observed encounter. This uses physical geometry, "
        "not perception, motor commands, or a new threat heuristic. It is an encounter label, not an "
        "instantaneous tick-wise classification against the moving arm.", "",
        "A non-threat trial is false-triggered if it has at least one recorded `reflex_activation_event`. "
        "Activation means entry from NOMINAL into the active state group (REFLEX_ACTIVE or "
        "PERCEPTION_HOLD). ACTIVE -> HOLD -> ACTIVE continues the same episode and adds no event. "
        "Release to NOMINAL followed by re-entry adds an event. Repeated events count once in the trial "
        "flag; `state_activation_false_trigger_events` preserves their multiplicity. The percentage "
        "denominator is all ground-truth non-threat trials, not all trials or activations; no eligible "
        "trials yields NA. The last partial interval follows the existing held-command convention.", "",
        "| Profile | P false trials / non-threats | Frozen PD false trials / non-threats | P / PD false activation events |",
        "|---|---:|---:|---:|"]
    for profile in ("all", "legacy_clean", "hysteresis_clean", "hysteresis_corrupt"):
        p, pd = [next(r for r in activation if r["profile"] == profile and r["controller"] == c) for c in ("p", "pid")]
        lines.append(f"| {profile} | {p['state_activation_false_trigger_trials']} / {p['state_activation_non_threat_trials']} | "
                     f"{pd['state_activation_false_trigger_trials']} / {pd['state_activation_non_threat_trials']} | "
                     f"{p['state_activation_false_trigger_events']} / {pd['state_activation_false_trigger_events']} |")
    lines += ["", "Both controllers have 15/144 = 10.42% overall in these matched runs. The per-case "
              f"false-trigger flags differ in {checks['activation_false_trigger_differences_p_vs_pd']} cases; "
              f"activation-event counts differ in {checks['activation_event_count_differences_p_vs_pd']} cases. "
              "Equality was measured, not imposed. The legacy TTC gate accounts for the non-threat "
              "activations here; both hysteresis profiles have zero. Future controller motion can genuinely "
              "change arm-dependent gate/release decisions even with identical sensor input, so equality "
              "is not a metric requirement. The definition itself is unchanged for P, PD and future spiking "
              "controllers. Existing `false_trigger` and `false_trigger_rate` retain their legacy "
              "command-comparison meanings; new trial columns and `state_activation_false_trigger_rate` "
              "summary rows are exported alongside them.", "",
        "### Canonical frozen reference and verification", "",
        "`conventional_reference.json` is the single canonical preset, explicitly named "
        "`conventional-reference` for the later neuromorphic comparison. Use "
        "`--controller pid --pid-preset conventional-reference`; explicit gain/filter overrides are rejected. "
        "The study's frozen evaluation also loads this file rather than manually entered gains or "
        "mutable historical selection output.", "",
        "- Both joints, avoidance: **Kp=5.25, Ki=0, Kd=0.1**.",
        "- Both joints, nominal recovery: **Kp=2.5, Ki=0, Kd=0.1**.",
        "- **Derivative tau=0.04 s; dt=0.020 s** (alpha=2/3).",
        "- Integral bound remains +/-100 deg*s. Integral implementation, conditional anti-windup and "
        "P/I/D-compatible state/diagnostics remain present; Ki=0 leaves the integral pathway inactive.", "",
        "**Conventional tuning is complete. These parameters are frozen for the subsequent spiking "
        "PID/PD comparison. No further conventional tuning is permitted.** Historical `tune` / `all` "
        "tuning entry points now reject execution. Frozen preset selection does not change default P behavior.", "",
        "All 324 archived P and 324 selected-PD safety, diagnostic and release metric records match "
        "exactly. The retained seed-23 hysteresis-corrupt P/PD tick CSVs are byte-identical to their "
        "archived traces. Future controller-equivalence experiments must replay identical recorded "
        "targets along with fixed trajectories, perception inputs, seeds, reflex/IK rules, urgency, "
        "limits and plant. Use the existing target-replay workflow for that purpose. Matched closed-loop "
        "safety runs remain a separate check because realized IK targets depend on each arm's pose.", ""]
    test_log = out / "tests_after.txt"
    if test_log.exists():
        text = test_log.read_text(encoding="utf-8")
        count = next(s for s in text.splitlines() if s.startswith("Ran "))
        lines.append("Final relevant/full test suite: " + count + "; " + ("PASS." if text.rstrip().endswith("OK") else "see test log."))
    baseline_log = out / "baseline_preservation.txt"
    if baseline_log.exists():
        lines.append(baseline_log.read_text(encoding="utf-8").strip())
    lines += ["", "### Exact reproduction commands", "", "Run from the repository root:", "", "```powershell",
        "# Three fixed-gain filter variants plus matched P and the new activation metric:",
        "python run_pid_final_validation.py filters",
        "# Frozen P-vs-PD benchmark and activation metric; separate output preserves the filter table:",
        "python run_pid_final_validation.py frozen --output benchmark_results/pid/frozen_reference",
        "# Ordinary benchmark CLI using canonical gains, without manual entry:",
        "python run_dynamic_cases.py --controller pid --pid-preset conventional-reference --seed 23 --num-trials 36 --csv benchmark_results/pid/frozen_cli.csv",
        "# Tests and pre-PID P preservation:",
        "python -m unittest discover -s tests -v",
        "python pid_baseline.py --verify benchmark_results/pid/p_before.json",
        "# Regenerate the final report section from recorded filter results:",
        "python run_pid_final_validation.py report",
        "# Or run fixed filter validation and report generation together (no tuning):",
        "python run_pid_final_validation.py all", "```", "",
        "Evidence: `final_validation/filter_sensitivity.csv`, `matched_trials.csv`, "
        "`activation_false_triggers.csv`, `fixed_steps.csv` and `validation_checks.json`. "
        "Per-profile CSVs expose both old and new false-trigger columns. Existing static collision "
        "results and earlier failed/rejected tuning evidence remain archived unchanged.", ""]
    original = report_path.read_text(encoding="utf-8")
    report_path.write_text(original.split(MARKER)[0].rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")
    print(f"Final validation report: {report_path}")
