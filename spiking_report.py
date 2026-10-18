"""Build the requested technical report from archived stage measurements."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from spiking_config import load_spiking_config


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(rows, columns):
    def value(v):
        if v is None or v == "":
            return "—"
        try:
            return f"{float(v):.5g}"
        except (ValueError, TypeError):
            return str(v)
    return "| " + " | ".join(columns) + " |\n|"+"|".join("---" for _ in columns)+"|\n"+"\n".join(
        "| "+" | ".join(value(row.get(c)) for c in columns)+" |" for row in rows)+"\n"


def build(out):
    config=load_spiking_config(out/"selected_spiking.json")
    selected=json.loads((out/"selected_spiking.json").read_text())
    ab=read(out/"stage_ab.csv"); cd=read(out/"stage_cd_recorded.csv")
    replay=read(out/"target_replay.csv"); closed=read(out/"closed_loop_summary.csv")
    pairs=read(out/"closed_loop_pairs.csv")
    calibration=read(out/"derivative_calibration.csv"); synthetic=read(out/"stage_c_synthetic.csv")
    # Recompute metric labels from the unchanged actual spike-derived synthetic traces.
    from run_spiking_study import signal_metrics, write_rows
    synthetic_ticks=read(out/"derivative_synthetic_ticks.csv")
    for row in synthetic:
        group=[r for r in synthetic_ticks if r["sequence"]==row["sequence"] and r["joint"]==row["joint"]]
        reference=np.array([float(r["reference_d_deg_s"]) for r in group])
        actual=np.array([float(r["spiking_d_deg_s"]) for r in group])
        row.update(signal_metrics(reference,actual))
        row["tail_residual_std_deg_s"]=float(np.std((actual-reference)[-50:]))
        row.pop("stationary_ripple_rms_deg_s",None)
    write_rows(out/"stage_c_synthetic.csv",synthetic)
    d=[r for r in cd if r["stage"]=="D" and r["seed"] in ("23","41")]
    replay_summary=[]
    for name in ("pid","spiking-pid"):
        group=[r for r in replay if r["controller"]==name]
        for joint in ("shoulder","elbow"):
            time=sum(float(r[f"{joint}_tracking_time_s"]) for r in group)
            rmse=(sum(float(r[f"{joint}_tracking_rmse_deg"])**2*float(r[f"{joint}_tracking_time_s"]) for r in group)/time)**.5
            replay_summary.append({"controller":name,"joint":joint,"trials":len(group),"tracking_rmse_deg":rmse,
                "peak_error_deg":max(float(r[f"{joint}_peak_error_deg"]) for r in group),
                "saturation_fraction":sum(float(r[f"{joint}_saturation_fraction"])*float(r[f"{joint}_tracking_time_s"]) for r in group)/time,
                "joint_stops":sum(int(r[f"{joint}_joint_stop_count"]) for r in group)})
    commandrows=read(out/"target_replay_commands.csv")
    replay_commands=[]
    for signal in ("requested","applied"):
        for j in ("shoulder","elbow"):
            group=[r for r in commandrows if r["joint"]==j and r["signal"]==signal]
            count=sum(int(r["samples"]) for r in group)
            replay_commands.append({"signal":signal,"joint":j,"rmse_deg_s":
                (sum(float(r["rmse"])**2*int(r["samples"]) for r in group)/count)**.5,
                "max_difference_deg_s":max(float(r["max_error"]) for r in group)})
    delta=np.array([float(r["clearance_change_mm"]) for r in pairs])
    regression=json.loads((out/"regression_audit.json").read_text())
    text=["# NeuroReflex Lava NEF spiking PD report\n",
        "The implementation is a genuine spike-driven **CPU Lava realization** of the frozen PD structure. "
        "It approximates the conventional signals and does not constitute a compiled Loihi 2 controller. "
        "The conventional reference, default P controller, perception, IK, gate/release policy, limits and direct-velocity plant remain fixed.\n",
        "Identical-target tracking is close, but closed-loop trajectory equivalence fails in one sensitive "
        "case: shoulder paths separate by 163.9° and minimum clearance drops by 52.1 mm. "
        "Zero contacts and similar aggregate tracking therefore do not establish controller interchangeability. "
        "The measured failure is preserved without safety-based tuning.\n",
        "## 1. Files inspected\n\n"
        "`motor_controller.py`, `conventional_reference.json`, `simulation_core.py` controller hook/mode/clamps, "
        "`diagnostics.py`, `motor_metrics.py`, `run_pid_study.py` replay/profiles, `run_pid_final_validation.py` archive verifier, "
        "`run_dynamic_cases.py`, `run_static_cases.py` CLI/runner, `pid_baseline.py`, relevant PID tests, "
        "archived `benchmark_results/pid/*` traces/tracking/selected results, repository dependency metadata, "
        "installed Lava Process/Var/LIF/Dense/RunConfig/runtime APIs, and local `fnbot-15-631159.pdf` NEF/PID/Loihi sections.\n",
        "## 2. Files added or modified\n\n"
        "Added `spiking_config.py`, `spiking_reference.json`, `spiking_frozen_sources.json`, `nef_population.py`, "
        "`neuromorphic_lava/{__init__,graph}.py`, `lava_nef.py`, `spiking_controller.py`, `run_spiking_study.py`, "
        "`spiking_report.py`, `inspect_spiking_regression.py`, `export_spiking_topology.py`, `verify_spiking_environment.py`, `spiking_smoke.py`, `requirements-spiking.txt`, "
        "`tests/test_spiking.py` and all stage evidence in this directory. Modified only controller selection/hooks, "
        "resource cleanup and extra diagnostics in `motor_controller.py`, `simulation_core.py`, `run_dynamic_cases.py`, "
        "`run_static_cases.py`; also README and environment ignores. The PIDController class and P function bodies are unchanged.\n",
        "## 3–4. Lava backend and hardware availability\n\n"
        "Lava **0.10.0**, CPython **3.10.22**, NumPy **1.26.4**, SciPy **1.15.3**, Windows CPU; "
        "`Loihi2SimCfg(select_tag='floating_pt')`, actual `AbstractProcess`/`PyLoihiProcessModel`, "
        "`LoihiProtocol`, `RunSteps` and runtime actors. The simulator name does not imply hardware or bit-accurate Loihi emulation. "
        "The default Python 3.13 environment was unsuitable for Lava's declared Python >=3.10,<3.11 requirement, "
        "so dependencies were isolated in `.venv-neuromorphic`. "
        "No Loihi 2 board access or private compiler was available. No hardware latency, power or energy is reported. "
        "Windows actor pipes required an execution sandbox exception; all experiments ran locally. "
        "The first runtime smoke failed at pipe creation inside the sandbox, then succeeded outside it. "
        "A harmless networkx optional-pandas ImportWarning may appear during tests. "
        "[Lava release metadata](https://github.com/lava-nc/lava/blob/v0.10.0/pyproject.toml).\n",
        "## 5. Neural architecture\n\n```text\n"
        "existing IK target - joint angle (deg), host normalization only\n"
        "  -> independent shoulder / elbow error LIF populations\n"
        "  -> P synapse + fixed NEF decoder/encoder weights (5.25 or 2.5)\n"
        "  -> D short synapse -- (+0.1*scale) --+\n"
        "     D long synapse  -- (-0.1*scale) --+ sum -> output LIF populations\n"
        "     I scaffold (Ki=0, zero influence) +         -> actual output spikes\n"
        "  -> synaptic decoder + 20 ms mean readout (deg/s)\n"
        "  -> unchanged external velocity clamp -> unchanged simulated plant\n"
        "passive frozen PD diagnostics observe the same errors; no connection to actuation\n```\n",
        "## 6–10. NEF representation, neuron parameters and decoder fit\n\n"
        f"Each joint has {config.neurons_per_ensemble} error neurons and {config.neurons_per_ensemble} output neurons "
        f"(**{4*config.neurons_per_ensemble} total**). Scalars use 1-D ensembles. "
        "Encoders are paired +1/-1. Each pair shares intercept, gain, bias and initial voltage; the two joints "
        "have independently seeded state. Seed 2026, joint offset 101 and output-ensemble offset 10000. "
        "Intercepts are uniform [-0.9,0.9], specified rate targets uniform [120,360] Hz at normalized +1; "
        "actual discrete firing can differ from these continuous parameter targets. "
        "Gain=(Jmax−1)/(1−intercept), bias=1−gain*intercept, Jmax=1/(1−exp(−1/(rate*tau_mem))). "
        "Membrane tau=5 ms, threshold=1, hard reset=0, no separate refractory period. "
        "Negative subthreshold voltage is allowed in the floating CPU model. "
        "At h=1 ms, dv=1−exp(−h/5 ms); v_next=(1−dv)*v+dv*(gain*x+bias), spike if v>1. "
        "This is the installed Lava floating LIF voltage recurrence with instantaneous current (du=1) and scaled bias/activation.\n\n"
        "Offline decoder fitting uses 801 normalized samples [-1,1] and the actual discrete LIF steady firing-rate law "
        "(integer interspike intervals), not an online rate approximation. For paired rate differences A, "
        "d=(AᵀA+lambda*I)⁻¹Aᵀx, lambda=0.001*trace(AᵀA)/(N/2). "
        "Final motor values depend on binary spikes, not fitted rates. Exact +/- cancellation makes zero input "
        "decode to zero after a fresh paired reset without a host deadband, despite background spikes. After nonzero inputs, unequal neuron phases can leave small zero-input ripple. This pairing is an adaptation "
        "to reduce low-error noise, not a claim of identical neuron parameters to the paper. "
        "[Zaidel et al., NEF/PID methodology](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.631159/full).\n",
        "## 11. Representation ranges and invertible units\n\n"
        "Archived 324 selected-PD trials show max |shoulder error|=108.16616076796429°, "
        "max |elbow error|=76.88646721977756°. "
        f"Error radii={config.error_radius_deg}°: about 25% margin rounded up to 10°. "
        "Host x=clip(error/radius,−1,1), decoded error=radius*decoder·filtered spike rates. "
        "Input overflow is logged explicitly before clipping. "
        f"Output radii={config.output_radius_deg_s} deg/s, conservative bounds from "
        "1.25*(5.25+0.1/(0.04+0.02))*error_radius rounded up to 50 deg/s. "
        "Output population drive is not clipped; outside-range drive is logged. "
        "These radii are representational scales, not actuator limits. "
        "Only SimulationCore clips requested motor velocity at the original urgency-scaled limits.\n",
        "## 12. Timing and observation delay\n\n"
        "One Lava RunStep is explicitly mapped to **1 ms virtual neural time**, not wall time. "
        "Each input is held for 20 neural steps per original 20 ms control interval. "
        "All represented error/P/D/output values are means over the current interval of their synaptically "
        "filtered spike decoding. They are read after the 20th step and applied in that same simulator update. "
        "There is no queued extra 20 ms tick. LIF dynamics, 5 ms P synapse, dual D synapses, 10 ms output synapse "
        "and the mean observation window still cause phase lag; measured lag appears in stages B–E. "
        "This is a virtual-time software convention: running the network for 20 ms before applying its result "
        "does not prove a real-time hardware schedule. Fused CPU populations communicate in the same step; "
        "stock Dense's one-step buffer/partitioned hardware routing would add neural-step latency and needs explicit validation.\n",
        "## 13. P transformation\n\n"
        "With physical error decoder d_e=radius*d, weights are Kp*d_e. The population P synapse "
        "is s_p[t]=exp(−h/0.005)*s_p[t−1]+(1−exp(−h/0.005))*(spikes_positive−spikes_negative)/h. "
        "NEF projection into output-neuron current is W_P=outer(output_gain/output_radius,Kp*[d_e,−d_e]). "
        "The fused CPU model factors this matrix into a spike-weighted sum and encoder/gain multiplication; "
        "it never computes Kp*host error. Mode selects the frozen 5.25/2.5 weight set.\n",
        "## 14–16. Dual-synapse D and the frozen 40 ms reference\n\n"
        f"Selected **tau_short={config.tau_short_s:.17g} s**, **tau_long={config.tau_long_s:.17g} s**, "
        f"**scale={config.derivative_scale_per_s:.17g} /s**. Kd=0.1 throughout. "
        "D=0.1*scale*d_e·(s_short−s_long), with both synapses driven by the same actual population spikes. "
        "There is no online subtraction of consecutive host errors.\n\n"
        "For the conventional alpha=a=2/3 and T=0.020, D(z)=Kd*(1−a)/T*(1−z⁻¹)/(1−a*z⁻¹)*E(z). "
        "If H_long(z)=(1−a)/(1−a*z⁻¹) and the short branch were an identity, then "
        "D=Kd*(1−a)/(T*a)*(E−H_long*E), hence scale=25 /s. "
        "Matching an exponential neural long filter's 20-step retention to a gives "
        "tau_long=−T/log(a)=0.049326069247528636 s, which differs from the conventional backward-Euler tau=0.04 s. "
        "This derivation is the starting mapping, not a guarantee of equality with finite short filtering and "
        "LIF/readout dynamics. The small predeclared candidate bank short={2,5,10} ms and long={40,49.326,60} ms "
        "uses fixed error spikes, fits one shared positive scale (bounded [1,100]) by least squares to seed-17 D "
        "contributions, and selects minimum training RMSE. Held-out seeds never enter selection. "
        "Calibration's scalar filtering is an offline linear rearrangement of graph synapses; it is checked "
        "against actual Lava D output to <1e−8 deg/s. No collision or clearance objective was used.\n\n"+
        f"The selected continuous low-frequency gain scale*(tau_long−tau_short) is "
        f"{config.derivative_scale_per_s*(config.tau_long_s-config.tau_short_s):.9g}; "
        "it is close to an ideal differentiator's unit gain, while finite short filtering limits "
        "high-frequency response. This low-frequency observation does not remove discrete spike/readout errors.\n\n"+
        table(calibration,["tau_short_s","tau_long_s","derivative_scale_per_s","training_rmse_deg_s","validation_seed23_rmse_deg_s","validation_seed41_rmse_deg_s"]),
        "## 17–18. I scaffold and mode/reset policy\n\n"
        "Ki=0 and the two-joint integral scaffold and output weights remain zero. No recurrent integrator "
        "is active; adding the paper's recurrent dynamical-system integrator is a separate experiment. "
        "Every transition among NOMINAL/REFLEX_ACTIVE/PERCEPTION_HOLD resets voltages to deterministic paired "
        "initial phases and clears all synapse/readout/integral state. During the first interval after a reset, "
        "short and long D states follow the same actual P spike trace, so D=0 throughout the interval, "
        "matching the conventional no-previous-error semantics. Later intervals use their distinct time constants. "
        "Hold emits zero velocity with no neural time advancement in the online adapter. Exit resets again. "
        "Batched hold rows have zero diagnostics and no spikes; blank hold separators explicitly reset "
        "independent trials. Errors/targets are not fabricated on missing-data holds.\n",
        "## 19–20. Stages A and B\n\n"
        "Stage A uses −radius, −0.75radius, −0.25radius, zero, positive counterparts and an intentional "
        "1.1radius overflow. Stationary statistics exclude the first 200 ms of each 1 s plateau and compare "
        "to the explicitly clipped representable input. Stage B includes sinusoid, step plateau and reversal. "
        "RMSE/MAE/max/bias units are degrees for A and deg/s for B.\n\n"+
        table(ab,["stage","mode","joint","rmse","mae","max_error","bias","correlation","lag_ms","input_clipped_intervals"]),
        "The initial endpoint-only readout had A RMSE about 4.3° and P RMSE 16–19 deg/s "
        "(`initial_endpoint_stage_ab.csv`). Averaging the already spike-derived signals within the existing "
        "interval reduced ripple without changing conventional gains or neuron count. This failed initial "
        "measurement is retained.\n",
        "## 21. Stage C: derivative equivalence\n\n"
        "All steps, ramps, sinusoid, reversal and constant inputs are retained in "
        "`derivative_synthetic_ticks.csv`. A constant signal produces zero reset D and finite stationary "
        "ripple with approximately zero mean, rather than magically silent instantaneous decoding.\n\n"+
        table(synthetic,["sequence","joint","rmse","mae","max_error","correlation","target_peak","actual_peak","lag_ms","tail_residual_std_deg_s"])+"\n"+
        table([r for r in cd if r["stage"]=="C"],["seed","joint","rmse","mae","max_error","correlation","lag_ms"]),
        "## 22. Stage D: identical-error command equivalence\n\n"
        "All 324 frozen PD recorded error/mode/limit traces were replayed open-loop. "
        "`open_loop_ticks.csv` retains normalized/clipped inputs, actual neural contributions, spike statistics "
        "and passive reference command/residual. Sign disagreement includes exact zero and near-zero "
        "commands; it is not suppressed with a tolerance. Saturation status disagreement compares "
        "|request|>the same urgency limit on every tick; transition-count difference is supplementary. "
        "Lag is the minimum-MSE integer shift over ±100 ms and can hide sub-tick phase differences; both "
        "unaligned and aligned RMSE are saved. Trial-boundary resets are explicit.\n\n"+
        table([r for r in cd if r["stage"]=="D"],["seed","joint","rmse","mae","max_error","correlation","sign_disagreement_fraction","saturation_status_disagreement_fraction","lag_ms"]),
        "## 23. Stage E: identical-target replay\n\n"
        "The existing target replay update order, neutral initial angles (35°,55°), mode preparation, "
        "urgency limits, direct velocity integration and original joint stops are reused. Source targets "
        "are frozen-PD recorded angle+error, independently of the replayed arms. All 216 held-out trials "
        "(23/41 × three profiles × 36) use identical targets/modes/limits for both controllers; errors are "
        "recomputed against each arm. Holds preserve pose and omit tracking samples exactly as the existing "
        "replay. No neural parameter was changed after signal calibration.\n\n"+
        table(replay_summary,["controller","joint","trials","tracking_rmse_deg","peak_error_deg","saturation_fraction","joint_stops"])+"\n"+
        table(replay_commands,["signal","joint","rmse_deg_s","max_difference_deg_s"]),
        "## 24. Stage F: matched closed-loop safety\n\n"
        "Stage F follows reviewed A–E signal and target-replay results. It uses the original nine trajectories, "
        "all three profiles, held-out seeds 23/41, 36 trials/profile, speed range 0.3–2 m/s, 1 s post-encounter "
        "observation, 20 ms control and 2 ms geometry evaluation. **216 paired trials**. Actual true/perceived "
        "fields, availability, clock and urgency velocity limits are checked tick-by-tick with the existing "
        "matched-input verifier. Gate/IK targets can differ due to arm geometry in closed loop; no inputs "
        "are selected using controller motion. State activation false-trigger remains the primary non-threat "
        "metric, with stationary neutral baseline_contact=False and activation events as its definition. "
        "Legacy command-dependent false-trigger is archived only as supplementary provenance.\n\n"+
        table(closed,["controller","profile","trials","collisions","worst_clearance_mm","state_activation_false_trigger_trials","state_activation_non_threat_trials","missed_threats","late_threats","tracking_rmse_deg","peak_error_deg","reflex_saturation_ratio","request_applied_rms_deg_s","reflex_active_time_s"])+
        f"\nPaired clearance change (spiking−PD), mm: mean {delta.mean():.6g}, median {np.median(delta):.6g}, "
        f"minimum {delta.min():.6g}, maximum {delta.max():.6g}; "
        f"new contacts {sum(r['pd_contact']=='False' and r['spiking_contact']=='True' for r in pairs)}. "
        "All per-trial changes, including regressions, are in `closed_loop_pairs.csv`. These seeded simulation "
        "results are not a safety certification or proof about untested paths.\n\n"
        "**Largest fidelity regression:** seed 41 / hysteresis_corrupt / D0025, accelerating approach at "
        "0.3368301201313088 m/s. Clearance is 321.196889 mm (PD) versus 269.093736 mm (spiking). "
        "Reproducing this single case gives identical minima, matches all 296 actual input ticks, and "
        "has zero mode-disagreement ticks. The first reflex elbow command is −96.140 deg/s for PD and "
        "−39.652 deg/s for the spiking controller; output-population/filter startup leaves a 1.130° "
        "elbow difference after one interval. On the next tick the unchanged pose-dependent normalized "
        "escape directions become 48.364° versus 79.578°, with the end effector only 13.63/10.66 mm "
        "from the predicted impact point. Subsequently the IK targets diverge, reaching a 190.304° "
        "shoulder-target difference and 163.882° shoulder-angle difference; the neural arm reaches the "
        "existing +120° shoulder stop while PD moves toward negative angles. "
        "The traces support amplification of a neural startup mismatch by the unchanged escape geometry, "
        "rather than a difference in perception samples or mode policy. This is a closed-loop **trajectory "
        "equivalence failure**, even though both paths avoid contact in this case. "
        "`regression_audit.json` and `regression_{pid,spiking-pid}_ticks.csv` retain the diagnosis. "
        "No controller, IK, gate, release or safety parameter was adjusted in response.\n",
        "## 25. Spike, firing and synaptic-event statistics\n\n"+
        table([r for r in cd if r["stage"]=="D"],["seed","joint","error_spikes","output_spikes","mean_firing_rate_hz","peak_firing_rate_hz","conceptual_dense_synaptic_events","input_clipped_intervals"])+
        "\nMean firing rate counts both populations per joint over all archived intervals (zero on holds); peak "
        "rate is the maximum individual neuron's 20 ms spike-count rate. Synaptic events are a **conceptual "
        "dense-topology estimate**: error spikes ×250 destination neurons ×3 P/short/long projections. "
        "This does not count host readout, bias spikes or I (inactive), and is not the operation count of "
        "the factored CPU implementation or a measured on-chip event counter. The two P weight sets are "
        "mode alternatives, not both active. No energy/power estimate is inferred.\n",
        "## 26. Tests and baseline preservation\n\n"
        "20 new deterministic configuration/unit/integration tests cover fixed-seed actual spike output, "
        "zero cancellation with background spikes, signs/bounds/clipping, exact neural P transform scaling, "
        "constant/step/ramp D, dual-filter subtraction, resets, nominal gain, hold, joint independence, "
        "frozen PID/reference and P source hashes, and actuator dependence on actual output spikes. "
        "The decisive integration test zeros the output spike decoder: motor output becomes zero while "
        "passive conventional PD remains nonzero. No reference diagnostic can substitute for spike decoding. "
        "Python 3.13 full suite: **169 tests, OK, 16 optional Lava integrations skipped** (153 executed). "
        "Python 3.10/Lava new tests: **20/20 pass**. Full Python 3.10 suite runs all 169 methods but "
        "two historical baseline-hash tests have **176 failing subtests**; these are not suppressed. "
        "Unchanged Git HEAD in the same Python 3.10 environment reproduces the same failures, proving "
        "they predate this implementation. A sample fixed P trace differs in 10 geometry/rate diagnostic "
        "floats by a few last bits (about 1e−16), enough to change exact JSON hashes; true/perceived input "
        "sequences and command values are identical in that sample. The older Python-specific hash "
        "fixtures remain untouched. Full-suite, new-test, HEAD-isolation and legacy baseline evidence "
        "are in `tests_after.txt`, `tests_default_python.txt`, `tests_spiking.txt`, "
        "`tests_unmodified_head_lava_python.txt`, `environment_float_differences.json` and "
        "`baseline_preservation.txt`. All 324 reconstructed frozen PD trial metric/diagnostic/release records "
        "matched archived CSVs exactly; no conventional source or JSON changes occurred. "
        "The P source, 16 static results, 108 dynamic results and full tick hashes match the original capture.\n",
        "## 27. Remaining approximation errors and answer\n\n"
        f"On held-out identical-error traces, command RMSE spans {min(float(r['rmse']) for r in d):.4g}–"
        f"{max(float(r['rmse']) for r in d):.4g} deg/s; correlations span "
        f"{min(float(r['correlation']) for r in d):.6g}–{max(float(r['correlation']) for r in d):.6g}. "
        "The target-replay and closed-loop tables above quantify the practical impact. "
        "The spiking controller reproduces the frozen structure and gains, **not its exact waveform**. "
        "Differences come from finite discrete LIF populations and decoder fit, spike ripple, membrane/synapse "
        "filtering, the 20 ms observation mean, approximate dual-filter transfer function, fresh population "
        "transients on mode resets, and output-population re-encoding. Reset transients particularly affect "
        "the first command after a mode change. The isolated D0025 failure shows that these small early "
        "joint differences can be strongly amplified when the existing normalized escape vector passes "
        "near the predicted impact point: matching signals/target replay is insufficient to guarantee "
        "matching closed-loop trajectories. This CPU realization is an experimental baseline, not an "
        "interchangeable replacement for frozen PD. No online host numerical derivative, I activation, controller "
        "gain retuning or safety-based calibration is involved. Neural D initialization approximates "
        "conventional reset behavior using spikes; it cannot guarantee identical hidden filter state "
        "after the first interval. Input/output representation overflow and actuator saturation are distinct "
        "and independently logged. Fixed-point and hardware routing differences remain unmeasured.\n",
        "## 28. Loihi 2 mapping status and blockers\n\n"
        "The conceptual topology uses LIF neurons, linear NEF connection weights, signed exponential "
        "synapses and linear output decoding; full shoulder/elbow P, short-D, negative long-D, output "
        "decoder and inactive-I matrices are exported in `topology/nef_topology.npz`, alongside explicit "
        "primitive recurrences in `topology/topology.json`. These are structurally expressible using "
        "programmable Loihi compartments and Dense projections. **The validated fused CPU ProcessModel "
        "is not directly hardware deployable.** It has no NC ProcessModel or compiled LIF/Dense subprocess "
        "decomposition; float arrays, same-step matrix factorization, configurable mode/reset channels and "
        "host diagnostics are CPU components. Exporting matrices is not compilation or proof of hardware "
        "resource fit. Installed public stock LIF/Dense models also only supply CPU models in this environment.\n\n"
        "The CPU-only batch recorder writes final diagnostic arrays to an NPZ after the last neural "
        "step, avoiding Lava 0.10's scalar-per-message Var.get transfer overhead. The recording file "
        "is never read by the graph or connected to motor output; hardware requires a separate host logger. "
        "The initial signal run was interrupted after successful D calibration to replace that slow logging "
        "path; final open-loop results were rerun with identical neural parameters and recurrences.\n\n"
        "Next: obtain private compiler/Loihi 2 access and supported Linux environment; implement primitive "
        "subprocess/NC mappings and control/reset channels; quantize voltage/current/weights/decays with "
        "bounded overflow and paired +/- symmetry; preserve units and explicit neural routing latency; "
        "keep input normalization and 20 ms spike observation on host; remove history diagnostics from "
        "on-chip execution; rerun A–E with fixed settings; compile/run using Loihi2HwCfg only when those "
        "models and hardware exist; then run F and measure real hardware power/latency. "
        "No Python object callbacks or arbitrary functions are required by the conceptual computation; "
        "the current NumPy CPU realization must nevertheless be lowered to hardware primitives.\n",
        "## 29. Exact reproduction (PowerShell, workspace root)\n\n```powershell\n"
        "# Existing isolated environment: Python 3.10.22 / Lava 0.10.0\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe -m ensurepip\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe -m pip install -r requirements-spiking.txt\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe spiking_smoke.py\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe run_spiking_study.py --stage collect\n"
        "# Signals begins with the documented analytic mapping, then fits seed-17 D only.\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe run_spiking_study.py --stage signals\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe run_spiking_study.py --stage replay\n"
        "# Review A--E before closed-loop; do not calibrate from F outcomes.\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe run_spiking_study.py --stage closed-loop\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe inspect_spiking_regression.py\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe export_spiking_topology.py\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe -m unittest discover -s tests -v\n"
        "python -m unittest discover -s tests -v\n"
        "python pid_baseline.py --verify benchmark_results/pid/p_before.json\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe spiking_report.py\n"
        ".\\.venv-neuromorphic\\Scripts\\python.exe run_dynamic_cases.py --controller spiking-pid --spiking-config spiking_reference.json --num-trials 9 --csv spiking_metrics.csv --ticks-csv spiking_ticks.csv\n"
        "python run_dynamic_cases.py --controller pid --pid-preset conventional-reference\n```\n\n"
        "For a fresh environment, create a Python 3.10 venv and install `requirements-spiking.txt`; "
        "the workspace runtime was installed using uv. pip itself may need `python -m ensurepip` in uv-created "
        "environments, or use `uv pip install --python .venv-neuromorphic/Scripts/python.exe -r requirements-spiking.txt`. "
        "Optional count/seed/timing overrides: `--spiking-neurons`, `--spiking-seed`, `--neural-dt`; "
        "each changes the experiment and requires renewed calibration. "
        "Default Python suite skips Lava integration tests; the pinned environment runs them. "
        "Archived traces use valid compressed JSON with explicit string tags for infinite TTC diagnostics.\n"]
    (out/"SPIKING_REPORT.md").write_text("\n".join(text),encoding="utf-8")
    (out/"replay_summary.json").write_text(json.dumps({"tracking":replay_summary,"commands":replay_commands},indent=2),encoding="utf-8")


if __name__=="__main__":
    build(Path(sys.argv[1]) if len(sys.argv)>1 else Path("benchmark_results/spiking"))
