"""Host adapter: error input and spike output only; PD reference is passive."""

import math
from pathlib import Path
import tempfile
import numpy as np

from motor_controller import JOINTS, JOINT_BOUNDS, PIDController, load_conventional_reference
from spiking_config import load_spiking_config


SPIKING_FIELDS = ["spiking_backend", "neural_dt_s", "neural_steps"] + [
    f"{j}_{suffix}" for j in JOINTS for suffix in (
        "decoded_error_deg", "short_filter_error_deg", "long_filter_error_deg",
        "error_spikes", "output_spikes", "mean_firing_rate_hz", "peak_firing_rate_hz",
        "input_clipped", "normalized_error", "output_representation_out_of_range",
        "reference_p_deg_s", "reference_d_deg_s", "reference_requested_deg_s", "command_residual_deg_s")]


def run_spike_batch(errors, modes, config=None):
    """Replay known inputs in one actual Lava run, returning spikes' diagnostics."""
    from lava_nef import NEFPD
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    config = config or load_spiking_config()
    inputs = np.asarray(errors, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 2 or len(inputs) == 0 or len(modes) != len(inputs):
        raise ValueError("Replay requires nonempty Nx2 errors and N modes.")
    if not np.isfinite(inputs).all() or not np.isin(modes, (0, 1, 2)).all():
        raise ValueError("Finite errors and nominal=0, reflex=1, hold=2 modes required.")
    normalized = inputs / config.error_radius_deg
    runtime_dir = Path(__file__).resolve().parent/".spiking-runtime"
    runtime_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime_dir) as directory:
        archive = Path(directory)/"batch.npz"
        process = NEFPD(config, load_conventional_reference(), np.clip(normalized,-1,1), modes,
                        batch_output_path=str(archive))
        started = False
        try:
            process.run(RunSteps(num_steps=len(inputs)*round(.02/config.neural_dt_s)),
                        run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
            started = True
            with np.load(archive) as data:
                return {"history": data["history"], "raw_projection": data["raw_projection"],
                        "input_clipped": np.abs(normalized) > 1}
        finally:
            if started:
                process.stop()


class SpikingController:
    is_spiking_controller = True

    def __init__(self, config=None):
        try:
            from lava_nef import NEFPD
        except ImportError as error:
            raise ValueError("spiking-pid requires Python 3.10 / lava-nc 0.10.0; use .venv-neuromorphic/Scripts/python.exe.") from error
        self.config = config or load_spiking_config()
        self.reference = PIDController(load_conventional_reference())
        self.process = NEFPD(self.config, self.reference.config)
        self.mode = None
        self.velocity_limits = (98., 130.)
        self.started = False
        self.closed = False
        self.reset_pending = True

    def reset(self):
        self.reference.reset()
        self.reset_pending = True
        if self.started:
            self.process.reset_flag.set(np.ones(1))

    def prepare(self, mode, velocity_limits):
        if mode not in ("NOMINAL", "REFLEX_ACTIVE", "PERCEPTION_HOLD"):
            raise ValueError("Unknown motor mode.")
        if mode != self.mode:
            self.reset()
            self.mode = mode
        self.reference.prepare(mode, velocity_limits)
        self.velocity_limits = tuple(velocity_limits)

    def begin_trial(self):
        """Reuse a CPU actor while resetting every trial's neural/reference state."""
        self.reset()
        self.mode = None
        self.reference.mode = None

    def update(self, errors, dt, velocity_limits, angles=None):
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava_nef import DIAG
        if self.closed:
            raise RuntimeError("Spiking controller has been closed.")
        if dt != .02 or len(errors) != 2 or not all(math.isfinite(e) for e in errors):
            raise ValueError("Spiking controller requires finite joint errors and dt=0.020 s.")
        if len(velocity_limits) != 2 or not all(math.isfinite(v) and v >= 0 for v in velocity_limits):
            raise ValueError("Supply finite nonnegative joint velocity limits.")
        if angles is not None and (len(angles) != 2 or not all(math.isfinite(q) for q in angles)):
            raise ValueError("Supply two finite joint angles.")
        if self.mode == "PERCEPTION_HOLD":
            return [0., 0.], self.hold_diagnostics()
        normalized = np.asarray(errors) / self.config.error_radius_deg
        values = {"input_norm": np.clip(normalized, -1, 1),
                  "mode": np.array([0. if self.mode == "NOMINAL" else 1.]),
                  "reset_flag": np.array([float(self.reset_pending)])}
        for name, value in values.items():
            variable = getattr(self.process, name)
            if self.started:
                variable.set(value)
            else:
                variable.init = value
        self.process.run(RunSteps(num_steps=round(dt/self.config.neural_dt_s)),
                         run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
        self.started = True
        neural = self.process.diagnostics.get()
        # Reference outputs are written ONLY to diagnostic fields, never actuation.
        _, ref = self.reference.update(errors, dt, velocity_limits, angles)
        details = {"pid_mode": self.mode, "pid_reset": self.reset_pending,
                   "spiking_backend": "Lava 0.10.0 CPU floating_pt",
                   "neural_dt_s": self.config.neural_dt_s,
                   "neural_steps": round(dt/self.config.neural_dt_s)}
        requests = neural[:, DIAG.index("requested_velocity_deg_s")].tolist()
        for i, joint in enumerate(JOINTS):
            for k, name in enumerate(DIAG):
                details[f"{joint}_{name}"] = float(neural[i, k])
            details.update({f"{joint}_error_deg": float(errors[i]),
                f"requested_{joint}_velocity_deg_s": requests[i],
                f"{joint}_input_clipped": bool(abs(normalized[i]) > 1),
                f"{joint}_normalized_error": float(np.clip(normalized[i], -1, 1)),
                f"{joint}_reference_p_deg_s": ref[f"{joint}_p_deg_s"],
                f"{joint}_reference_d_deg_s": ref[f"{joint}_d_deg_s"],
                f"{joint}_reference_requested_deg_s": ref[f"requested_{joint}_velocity_deg_s"],
                f"{joint}_command_residual_deg_s": requests[i] - ref[f"requested_{joint}_velocity_deg_s"],
                f"{joint}_i_deg_s": 0., f"{joint}_integral_deg_s": 0.,
                f"{joint}_filtered_derivative_deg_s": neural[i, 4] / .1,
                f"{joint}_integration_blocked": False})
            for suffix in ("effective_lower_deg_s", "effective_upper_deg_s", "target_deg"):
                if f"{joint}_{suffix}" in ref:
                    details[f"{joint}_{suffix}"] = ref[f"{joint}_{suffix}"]
        self.reset_pending = False
        return requests, details

    def __call__(self, state, dt):
        from simulation_core import student_controller
        target = student_controller(state, dt)["diagnostics"]
        requests, details = self.update([target[f"{j}_error_deg"] for j in JOINTS], dt,
                                       self.velocity_limits, [state[f"{j}_angle_deg"] for j in JOINTS])
        return {"shoulder_velocity_deg_s": requests[0], "elbow_velocity_deg_s": requests[1],
                "diagnostics": details}

    def hold_diagnostics(self):
        details = self.reference.hold_diagnostics()
        details.update({"spiking_backend": "Lava 0.10.0 CPU floating_pt", "neural_dt_s": self.config.neural_dt_s,
                        "neural_steps": 0})
        for joint in JOINTS:
            for field in SPIKING_FIELDS:
                if field.startswith(joint + "_"):
                    details[field] = 0.
        self.reset_pending = False
        return details

    def close(self):
        if self.started and not self.closed:
            self.process.stop()
        self.closed = True
