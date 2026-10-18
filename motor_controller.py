"""Stateful joint-space PID for the direct-velocity plant (degrees throughout).

The legacy P function remains the authoritative escape/IK target generator.
Its passive error metadata lets PID reuse that logic without modifying it.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path


JOINTS = ("shoulder", "elbow")
JOINT_BOUNDS = ((-90.0, 120.0), (-150.0, 150.0))
CONVENTIONAL_REFERENCE_PATH = Path(__file__).with_name("conventional_reference.json")
if not CONVENTIONAL_REFERENCE_PATH.is_file():
    from importlib.resources import files
    CONVENTIONAL_REFERENCE_PATH = files("neuroreflex").joinpath("configs/conventional_reference.json")
PID_FIELDS = ["pid_reset", "pid_mode"] + [
    f"{joint}_{suffix}" for joint in JOINTS for suffix in (
        "target_deg", "p_deg_s", "i_deg_s", "d_deg_s", "integral_deg_s",
        "filtered_derivative_deg_s", "integration_blocked", "effective_lower_deg_s",
        "effective_upper_deg_s")]


@dataclass(frozen=True)
class JointGains:
    kp: float = 6.0  # 1/s
    ki: float = 0.0  # 1/s^2
    kd: float = 0.0  # dimensionless

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.kp, self.ki, self.kd)):
            raise ValueError("PID gains must be finite and nonnegative.")


@dataclass(frozen=True)
class PIDConfig:
    shoulder: JointGains = field(default_factory=JointGains)
    elbow: JointGains = field(default_factory=JointGains)
    derivative_tau_s: float = 0.04
    integral_bound_deg_s: float = 100.0
    recovery_kp: float = 2.5

    def __post_init__(self):
        for value in (self.derivative_tau_s, self.integral_bound_deg_s, self.recovery_kp):
            if not math.isfinite(value) or value < 0:
                raise ValueError("PID filter, integral bound and recovery gain must be finite/nonnegative.")


@dataclass
class JointPIDState:
    integral: float = 0.0  # deg*s
    previous_error: float | None = None  # deg
    derivative: float = 0.0  # deg/s


class PIDController:
    def __init__(self, config=None):
        self.config = config if config is not None else PIDConfig()
        self.mode = None
        self.velocity_limits = (98.0, 130.0)
        self.reset()

    def reset(self):
        self.joints = [JointPIDState(), JointPIDState()]
        self.reset_pending = True

    def prepare(self, mode, velocity_limits):
        """Reset on every mode transition, including entry/exit of perception hold."""
        if mode != self.mode:
            self.reset()
            self.mode = mode
        self.velocity_limits = tuple(velocity_limits)

    def update(self, errors, dt, velocity_limits, angles=None):
        """Return raw PID requests plus passive diagnostics; caller clips velocity.

        Conditional integration evaluates the trial integral's output. Reject
        its change only if it pushes farther beyond the reachable interval.
        Opposite-sign changes are accepted even while the output is saturated.
        Joint position stops narrow that interval for anti-windup only; they do
        not alter the existing velocity clamp or plant integration.
        """
        if len(errors) != 2 or len(velocity_limits) != 2 or (angles is not None and len(angles) != 2):
            raise ValueError("Supply two joint errors, limits and (optionally) angles.")
        if not math.isfinite(dt) or dt <= 0 or not all(math.isfinite(e) for e in errors):
            raise ValueError("PID requires finite errors and a positive finite dt.")
        if not all(math.isfinite(v) and v >= 0 for v in velocity_limits):
            raise ValueError("Velocity limits must be finite/nonnegative.")
        if angles is not None and not all(math.isfinite(q) for q in angles):
            raise ValueError("Joint angles must be finite.")
        alpha = self.config.derivative_tau_s / (self.config.derivative_tau_s + dt)
        details = {"pid_reset": self.reset_pending, "pid_mode": self.mode}
        requests = []
        for i, (joint, error, maximum) in enumerate(zip(JOINTS, errors, velocity_limits)):
            history = self.joints[i]
            gains = (self.config.shoulder, self.config.elbow)[i]
            kp = self.config.recovery_kp if self.mode == "NOMINAL" else gains.kp
            raw_d = 0.0 if history.previous_error is None else (error - history.previous_error) / dt
            history.derivative = alpha * history.derivative + (1 - alpha) * raw_d
            history.previous_error = error
            p, d = kp * error, gains.kd * history.derivative
            lower, upper = -maximum, maximum
            if angles is not None:
                q_min, q_max = JOINT_BOUNDS[i]
                lower = max(lower, min(0.0, (q_min - angles[i]) / dt))
                upper = min(upper, max(0.0, (q_max - angles[i]) / dt))
            trial = max(-self.config.integral_bound_deg_s, min(
                self.config.integral_bound_deg_s, history.integral + error * dt)) if gains.ki else 0.0
            change = gains.ki * (trial - history.integral)
            provisional = p + gains.ki * trial + d
            blocked = ((provisional > upper and change > 0)
                       or (provisional < lower and change < 0))
            if not blocked:
                history.integral = trial
            integral_contribution = gains.ki * history.integral
            request = p + integral_contribution + d
            if not all(math.isfinite(v) for v in (request, history.integral, history.derivative)):
                raise ValueError("PID arithmetic overflow.")
            requests.append(request)
            details.update({
                f"{joint}_error_deg": error, f"{joint}_p_deg_s": p,
                f"{joint}_i_deg_s": integral_contribution, f"{joint}_d_deg_s": d,
                f"requested_{joint}_velocity_deg_s": request,
                f"{joint}_integral_deg_s": history.integral,
                f"{joint}_filtered_derivative_deg_s": history.derivative,
                f"{joint}_integration_blocked": blocked,
                f"{joint}_effective_lower_deg_s": lower,
                f"{joint}_effective_upper_deg_s": upper,
            })
            if angles is not None:
                details[f"{joint}_target_deg"] = angles[i] + error
        self.reset_pending = False
        return requests, details

    def __call__(self, state, dt):
        from simulation_core import student_controller
        baseline = student_controller(state, dt)
        errors = [baseline["diagnostics"][f"{joint}_error_deg"] for joint in JOINTS]
        angles = [state[f"{joint}_angle_deg"] for joint in JOINTS]
        requests, details = self.update(errors, dt, self.velocity_limits, angles)
        return {"shoulder_velocity_deg_s": requests[0],
                "elbow_velocity_deg_s": requests[1], "diagnostics": details}

    def hold_diagnostics(self):
        """Record the reset/frozen state without evaluating a target or advancing PID."""
        details = {"pid_mode": self.mode, "pid_reset": self.reset_pending}
        for joint, history in zip(JOINTS, self.joints):
            details.update({f"{joint}_p_deg_s": 0.0, f"{joint}_i_deg_s": 0.0,
                            f"{joint}_d_deg_s": 0.0, f"{joint}_integral_deg_s": history.integral,
                            f"{joint}_filtered_derivative_deg_s": history.derivative,
                            f"{joint}_integration_blocked": False})
        self.reset_pending = False
        return details


def add_controller_arguments(parser):
    parser.add_argument("--controller", choices=("p", "pid", "spiking-pid"), default="p")
    from spiking_config import add_spiking_arguments
    add_spiking_arguments(parser)
    parser.add_argument("--pid-preset", choices=("conventional-reference",),
                        help="Use the frozen conventional reference; gain/filter overrides are rejected.")
    for name, default in (("kp", 6.0), ("ki", 0.0), ("kd", 0.0)):
        parser.add_argument(f"--pid-{name}", type=float, help=f"Shared avoidance gain (default: {default}).")
        for joint in JOINTS:
            parser.add_argument(f"--pid-{joint}-{name}", type=float, help="Override shared gain.")
    parser.add_argument("--pid-derivative-tau", type=float, help="Error derivative filter tau (s; default: 0.04).")
    parser.add_argument("--pid-integral-bound", type=float, help="Symmetric integral bound (deg*s; default: 100).")
    parser.add_argument("--pid-recovery-kp", type=float, help="Nominal proportional gain (default: 2.5).")


def load_conventional_reference():
    """Single canonical JSON preset; retain the integral implementation with Ki=0."""
    document = json.loads(CONVENTIONAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    from simulation_core import CONTROL_DT
    if document["control_dt_s"] != CONTROL_DT:
        raise ValueError("Frozen conventional reference requires dt=0.020 s.")
    config = document["config"]
    return PIDConfig(JointGains(**config["shoulder"]), JointGains(**config["elbow"]),
                     config["derivative_tau_s"], config["integral_bound_deg_s"], config["recovery_kp"])


def config_from_args(args):
    if args.pid_preset:
        if args.controller != "pid":
            raise ValueError("The conventional-reference preset requires --controller pid.")
        if any(v is not None for k, v in vars(args).items() if k.startswith("pid_") and k != "pid_preset"):
            raise ValueError("Frozen preset does not accept gain/filter overrides.")
        return load_conventional_reference()
    def value(name, default):
        selected = getattr(args, name)
        return default if selected is None else selected
    def gains(joint):
        values = []
        for name, default in (("kp", 6.0), ("ki", 0.0), ("kd", 0.0)):
            override = getattr(args, f"pid_{joint}_{name}")
            values.append(override if override is not None else value(f"pid_{name}", default))
        return JointGains(*values)
    return PIDConfig(gains("shoulder"), gains("elbow"), value("pid_derivative_tau", .04),
                     value("pid_integral_bound", 100.), value("pid_recovery_kp", 2.5))
