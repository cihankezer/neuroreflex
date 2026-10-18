"""Shared fixed-step arm simulation, independent of Qt and plotting."""

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from perception import PerceptionModel
from trajectories import TrueBallState
from threat_gate import ThreatGateConfig, evaluate_threat_gate
from reflex_release import ReflexStateMachine


# ============================================================
# PARAMETERS
# ============================================================

L1 = 0.45
L2 = 0.35

NEUTRAL_SHOULDER = 35.0
NEUTRAL_ELBOW = 55.0

CONTROL_DT = 0.02

BALL_RADIUS = 0.055
ARM_RADIUS = 0.035

SHOULDER_MAX = 98.0  # deg/s
ELBOW_MAX = 130.0  # deg/s
DEFAULT_BALL_SPEED = 1.2  # m/s


# ============================================================
# STUDENT CONTROLLER
# ============================================================
#
# Students only need to modify this function.
#
# Inputs available in state:
#
# state["looming_rate"]
# state["ttc"]
#
# state["ball_x"]
# state["ball_y"]
# state["ball_z"]
#
# state["impact_x"]
# state["impact_y"]
#
# state["bearing_x_deg"]
# state["bearing_y_deg"]
#
# state["shoulder_angle_deg"]
# state["elbow_angle_deg"]
#
# state["end_effector_x"]
# state["end_effector_y"]
#
# state["urgency"]  # 0..1; scales actuator speed limits
# state["danger"]
#
# Output:
#
# {
#     "shoulder_velocity_deg_s": ...,
#     "elbow_velocity_deg_s": ...
# }
#
# ============================================================

def nominal_controller(state, dt):
    """Normal control without the avoidance reflex."""
    return {
        "shoulder_velocity_deg_s": 2.5 * (NEUTRAL_SHOULDER - state["shoulder_angle_deg"]),
        "elbow_velocity_deg_s": 2.5 * (NEUTRAL_ELBOW - state["elbow_angle_deg"]),
        "diagnostics": {
            "shoulder_error_deg": NEUTRAL_SHOULDER - state["shoulder_angle_deg"],
            "elbow_error_deg": NEUTRAL_ELBOW - state["elbow_angle_deg"],
        },
    }


def student_controller(state, dt):

    shoulder = state["shoulder_angle_deg"]
    elbow = state["elbow_angle_deg"]

    # --------------------------------------------------------
    # No danger:
    # return toward neutral pose
    # --------------------------------------------------------

    if not state["danger"]:
        return nominal_controller(state, dt)

    # --------------------------------------------------------
    # Danger:
    # move end effector away from predicted impact point
    # --------------------------------------------------------

    impact_x = state["impact_x"]
    impact_y = state["impact_y"]

    ee_x = state["end_effector_x"]
    ee_y = state["end_effector_y"]

    dx = ee_x - impact_x
    dy = ee_y - impact_y

    norm = math.hypot(dx, dy)

    if norm < 1e-6:
        dx = -1.0
        dy = 0.5
        norm = math.hypot(dx, dy)

    dx /= norm
    dy /= norm

    looming = state["looming_rate"]
    ttc = state["ttc"]

    urgency = min(
        1.0,
        max(0.2, looming)
    )

    escape_distance = (
        0.16
        + 0.22 * urgency
    )

    target_x = (
        ee_x
        + dx * escape_distance
    )

    target_y = (
        ee_y
        + dy * escape_distance
    )

    target_x, target_y = clamp_workspace(
        target_x,
        target_y
    )

    target_shoulder, target_elbow = inverse_kinematics(
        target_x,
        target_y
    )

    shoulder_error = (
        target_shoulder
        - shoulder
    )

    elbow_error = (
        target_elbow
        - elbow
    )

    gain = 4.0

    if ttc < 0.7:
        gain = 6.0

    if ttc < 0.4:
        gain = 9.0

    if ttc < 0.2:
        gain = 12.0

    shoulder_velocity = (
        gain
        * shoulder_error
    )

    elbow_velocity = (
        gain
        * elbow_error
    )

    # Passive diagnostics retain the P request before the existing clamps.
    diagnostics = {
        "shoulder_error_deg": shoulder_error,
        "elbow_error_deg": elbow_error,
        "requested_shoulder_velocity_deg_s": shoulder_velocity,
        "requested_elbow_velocity_deg_s": elbow_velocity,
    }

    shoulder_velocity = clamp(
        shoulder_velocity,
        -state.get("urgency", 1.0) * SHOULDER_MAX,
        state.get("urgency", 1.0) * SHOULDER_MAX
    )

    elbow_velocity = clamp(
        elbow_velocity,
        -state.get("urgency", 1.0) * ELBOW_MAX,
        state.get("urgency", 1.0) * ELBOW_MAX
    )

    return {
        "shoulder_velocity_deg_s":
            shoulder_velocity,

        "elbow_velocity_deg_s":
            elbow_velocity,
        "diagnostics": diagnostics,
    }


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def forward_kinematics(
    shoulder_deg,
    elbow_deg
):

    q1 = math.radians(
        shoulder_deg
    )

    q2 = math.radians(
        elbow_deg
    )

    elbow_x = (
        L1 * math.cos(q1)
    )

    elbow_y = (
        L1 * math.sin(q1)
    )

    ee_x = (
        elbow_x
        + L2 * math.cos(q1 + q2)
    )

    ee_y = (
        elbow_y
        + L2 * math.sin(q1 + q2)
    )

    return (
        elbow_x,
        elbow_y,
        ee_x,
        ee_y
    )


def inverse_kinematics(x, y):

    r2 = (
        x * x
        + y * y
    )

    cos_q2 = (
        r2
        - L1 * L1
        - L2 * L2
    ) / (
        2.0 * L1 * L2
    )

    cos_q2 = clamp(
        cos_q2,
        -1.0,
        1.0
    )

    q2 = math.acos(
        cos_q2
    )

    q1 = (
        math.atan2(y, x)
        - math.atan2(
            L2 * math.sin(q2),
            L1 + L2 * math.cos(q2)
        )
    )

    return (
        math.degrees(q1),
        math.degrees(q2)
    )


def clamp_workspace(x, y):

    radius = math.hypot(
        x,
        y
    )

    max_radius = (
        L1
        + L2
        - 0.04
    )

    min_radius = (
        abs(L1 - L2)
        + 0.05
    )

    if radius > max_radius:

        scale = (
            max_radius
            / radius
        )

        x *= scale
        y *= scale

    elif radius < min_radius:

        if radius < 1e-6:
            x = min_radius
            y = 0.0

        else:

            scale = (
                min_radius
                / radius
            )

            x *= scale
            y *= scale

    return x, y


def point_segment_distance(
    p,
    a,
    b
):

    p = np.asarray(
        p,
        dtype=float
    )

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    ab = b - a

    denominator = np.dot(
        ab,
        ab
    )

    if denominator < 1e-12:
        return float(
            np.linalg.norm(
                p - a
            )
        )

    t = np.dot(
        p - a,
        ab
    ) / denominator

    t = clamp(
        t,
        0.0,
        1.0
    )

    closest = (
        a
        + t * ab
    )

    return float(
        np.linalg.norm(
            p - closest
        )
    )


# ============================================================
# STATE
# ============================================================

@dataclass
class RobotState:

    shoulder_deg: float = (
        NEUTRAL_SHOULDER
    )

    elbow_deg: float = (
        NEUTRAL_ELBOW
    )


@dataclass
class BallState:

    impact_x: float = 0.55
    impact_y: float = 0.42

    z: float = 4.0

    speed: float = DEFAULT_BALL_SPEED

    radius: float = BALL_RADIUS



def calculate_perception(ball):
    """Return looming rate, TTC, and bearings using the original model."""
    speed = max(ball.speed, 1e-6)
    distance = max(ball.z, 0.01)
    return (
        speed / distance,
        distance / speed,
        math.degrees(math.atan2(ball.impact_x, distance)),
        math.degrees(math.atan2(ball.impact_y, distance)),
    )


def collision_test(robot, ball):
    """Check both links at plane arrival; touching at exactly 0.09 m is safe."""
    elbow_x, elbow_y, ee_x, ee_y = forward_kinematics(
        robot.shoulder_deg, robot.elbow_deg
    )
    ball_point = (ball.impact_x, ball.impact_y, 0.0)
    shoulder_point = (0.0, 0.0, 0.0)
    elbow_point = (elbow_x, elbow_y, 0.0)
    ee_point = (ee_x, ee_y, 0.0)
    distance = min(
        point_segment_distance(ball_point, shoulder_point, elbow_point),
        point_segment_distance(ball_point, elbow_point, ee_point),
    )
    return distance < BALL_RADIUS + ARM_RADIUS


def surface_clearance(robot, ball):
    """Signed sphere-to-capsule clearance in metres and nearest link number.

    Both arm centre lines lie at Z=0, so the closest projection is in XY.
    This is equivalent to a 3D closest-point query for this planar arm.
    """
    elbow_x, elbow_y, ee_x, ee_y = forward_kinematics(
        robot.shoulder_deg, robot.elbow_deg
    )

    def distance(ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        fraction = clamp(
            ((ball.impact_x - ax) * dx + (ball.impact_y - ay) * dy)
            / (dx * dx + dy * dy), 0.0, 1.0,
        )
        return math.hypot(
            ball.impact_x - ax - fraction * dx,
            ball.impact_y - ay - fraction * dy,
            ball.z,
        )

    distances = (distance(0.0, 0.0, elbow_x, elbow_y),
                 distance(elbow_x, elbow_y, ee_x, ee_y))
    link = 1 if distances[0] <= distances[1] else 2
    return min(distances) - ball.radius - ARM_RADIUS, link


class SimulationCore:
    """One independent trial. Each step returns None, SAFE, or COLLISION.

    The GUI can reset the ball after a terminal result. Batch callers create a
    fresh core for each case so the arm always starts in its neutral pose.
    """

    def __init__(self, ball=None, urgency=1.0, ttc_threshold=1.0, controller=None,
                 trajectory=None, perception=None, threat_gate=None, reflex_release=None,
                 pid_config=None, spiking_config=None):
        if ball is not None and trajectory is not None:
            raise ValueError("Supply either a legacy ball or a trajectory.")
        self.trajectory = trajectory
        self.ball = (trajectory.state_at(0.0) if trajectory is not None
                     else ball if ball is not None else BallState())
        self.perception = perception if perception is not None else PerceptionModel()
        self.threat_gate = threat_gate if threat_gate is not None else ThreatGateConfig()
        self.last_gate_decision = None
        self.reflex_policy = ReflexStateMachine(self.threat_gate, reflex_release)
        self.last_release_decision = None
        self.release_metrics = None
        self.control_ticks = 0
        self.last_perceived_state = None
        self.control_diagnostics = []
        speed = (math.hypot(self.ball.velocity_x, self.ball.velocity_y,
                           self.ball.velocity_z) if trajectory is not None else self.ball.speed)
        values = (
            self.ball.impact_x, self.ball.impact_y, self.ball.z, speed,
            self.ball.radius,
            urgency, ttc_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Simulation inputs must be finite numbers.")
        if speed <= 0.0 or (trajectory is None and self.ball.z <= 0.0):
            raise ValueError("Ball speed and initial distance must be positive.")
        if self.ball.radius <= 0.0:
            raise ValueError("Ball radius must be positive.")
        if not 0.0 <= urgency <= 1.0:
            raise ValueError("Urgency must be between 0 and 1.")
        if ttc_threshold <= 0.0:
            raise ValueError("TTC threshold must be positive.")
        self.robot = RobotState()
        self.urgency = urgency
        self.ttc_threshold = ttc_threshold
        if controller == "pid":
            from motor_controller import PIDController
            self.controller = PIDController(pid_config)
        elif controller == "spiking-pid":
            from spiking_controller import SpikingController
            self.controller = SpikingController(spiking_config)
        elif controller is None or controller == "p":
            self.controller = student_controller
        elif callable(controller):
            self.controller = controller
        else:
            raise ValueError("Controller must be p, pid, spiking-pid or callable.")
        self.danger = False
        self.result = None
        self.last_control_state = None
        self.last_command = {
            "shoulder_velocity_deg_s": 0.0,
            "elbow_velocity_deg_s": 0.0,
        }

    @property
    def diagnostic_summary(self):
        """Passive trial statistics, also available to legacy static callers."""
        from diagnostics import summarize_diagnostics
        duration = self.trajectory.duration_s if self.trajectory is not None else None
        return summarize_diagnostics(self.control_diagnostics, duration_s=duration)

    def step(self, stop_at_plane=True):
        """Advance one control tick.

        Performance evaluation passes stop_at_plane=False and owns its longer
        encounter horizon and contact outcome. The manual UI keeps plane stops.
        """
        if self.result is not None:
            return self.result
        if stop_at_plane and self.trajectory is not None:
            raise ValueError("Dynamic trials must use the whole-encounter evaluator.")

        # Preserve the original update order and 20 ms integration step.
        self.control_ticks += 1
        time_s = self.control_ticks * CONTROL_DT
        if self.trajectory is None:
            self.ball.z -= self.ball.speed * CONTROL_DT
            true_state = TrueBallState(self.ball.impact_x, self.ball.impact_y,
                                       self.ball.z, 0, 0, -self.ball.speed, self.ball.radius)
        else:
            self.ball = self.trajectory.state_at(time_s)
            true_state = self.ball
        perceived = self.perception.update(true_state, time_s, self.urgency,
                                           legacy_static=self.trajectory is None)
        self.last_perceived_state = perceived
        elbow_x, elbow_y, ee_x, ee_y = forward_kinematics(
            self.robot.shoulder_deg, self.robot.elbow_deg)
        links = (((0.0, 0.0, 0.0), (elbow_x, elbow_y, 0.0)),
                 ((elbow_x, elbow_y, 0.0), (ee_x, ee_y, 0.0)))
        self.last_gate_decision = evaluate_threat_gate(
            perceived, links, self.ttc_threshold, self.threat_gate,
            obstacle_radius_m=BALL_RADIUS, arm_radius_m=ARM_RADIUS)
        self.last_release_decision = self.reflex_policy.update(
            self.last_gate_decision, perceived, links, time_s, CONTROL_DT)
        self.danger = self.last_release_decision.final_reflex_active
        self.last_gate_decision = replace(self.last_gate_decision, final_reflex_gate_pass=self.danger)
        # Missing threat fields stay explicit None. Hysteresis mode freezes the
        # pose after perception expires instead of calling nominal recovery.
        urgency = perceived.urgency if perceived is not None else 0.0
        _, _, ee_x, ee_y = forward_kinematics(
            self.robot.shoulder_deg, self.robot.elbow_deg
        )
        state = {
            "looming_rate": perceived.looming_rate if perceived else 0.0,
            "ttc": perceived.ttc_s if perceived else math.inf,
            "ball_x": perceived.x if perceived else None,
            "ball_y": perceived.y if perceived else None,
            "ball_z": perceived.z if perceived else None,
            "impact_x": perceived.impact_x if perceived else None,
            "impact_y": perceived.impact_y if perceived else None,
            "bearing_x_deg": perceived.bearing_x_deg if perceived else None,
            "bearing_y_deg": perceived.bearing_y_deg if perceived else None,
            "shoulder_angle_deg": self.robot.shoulder_deg,
            "elbow_angle_deg": self.robot.elbow_deg,
            "end_effector_x": ee_x,
            "end_effector_y": ee_y,
            "urgency": urgency,
            "danger": self.danger,
        }
        self.last_control_state = state
        # PID state is private to this trial. P follows its original path.
        from motor_controller import PIDController
        stateful_motor = isinstance(self.controller, PIDController) or getattr(self.controller, "is_spiking_controller", False)
        if stateful_motor:
            limit_urgency = urgency if perceived is not None else self.urgency
            self.controller.prepare(self.last_release_decision.reflex_state,
                                    (limit_urgency * SHOULDER_MAX, limit_urgency * ELBOW_MAX))
        command = ({} if self.last_release_decision.reflex_state == "PERCEPTION_HOLD"
                   else self.controller(state, CONTROL_DT))
        if self.last_release_decision.reflex_state == "PERCEPTION_HOLD" and stateful_motor:
            command = {"diagnostics": self.controller.hold_diagnostics()}
        shoulder_velocity = command.get("shoulder_velocity_deg_s", 0.0)
        elbow_velocity = command.get("elbow_velocity_deg_s", 0.0)
        if not all(math.isfinite(v) for v in (shoulder_velocity, elbow_velocity)):
            raise ValueError("Controller velocities must be finite numbers.")
        # Preserve configured limits on missing-data ticks. Stateful fallback
        # commands zero speed; legacy modes retain their nominal recovery.
        limit_urgency = urgency if perceived is not None else self.urgency
        shoulder_limit = limit_urgency * SHOULDER_MAX
        elbow_limit = limit_urgency * ELBOW_MAX
        details = command.get("diagnostics", {})
        requested_shoulder = details.get("requested_shoulder_velocity_deg_s", shoulder_velocity)
        requested_elbow = details.get("requested_elbow_velocity_deg_s", elbow_velocity)
        shoulder_velocity = clamp(
            shoulder_velocity, -shoulder_limit, shoulder_limit
        )
        elbow_velocity = clamp(elbow_velocity, -elbow_limit, elbow_limit)
        self.last_command = {
            "shoulder_velocity_deg_s": shoulder_velocity,
            "elbow_velocity_deg_s": elbow_velocity,
        }
        from diagnostics import control_tick_record
        self.control_diagnostics.append(control_tick_record(
            time_s, self.robot, true_state, perceived, self.danger, details,
            requested_shoulder, requested_elbow, shoulder_velocity, elbow_velocity,
            shoulder_limit, elbow_limit, legacy_static=self.trajectory is None,
        ))
        self.control_diagnostics[-1].update(asdict(self.last_gate_decision))
        self.control_diagnostics[-1].update(asdict(self.last_release_decision))
        if stateful_motor:
            from motor_controller import PID_FIELDS
            self.control_diagnostics[-1].update({k: details[k] for k in PID_FIELDS if k in details})
        if getattr(self.controller, "is_spiking_controller", False):
            from spiking_controller import SPIKING_FIELDS
            self.control_diagnostics[-1].update({k: details[k] for k in SPIKING_FIELDS if k in details})
        self.robot.shoulder_deg = clamp(
            self.robot.shoulder_deg + shoulder_velocity * CONTROL_DT,
            -90.0, 120.0,
        )
        self.robot.elbow_deg = clamp(
            self.robot.elbow_deg + elbow_velocity * CONTROL_DT,
            -150.0, 150.0,
        )
        if stop_at_plane and self.ball.z <= 0.0:
            self.result = (
                "COLLISION" if collision_test(self.robot, self.ball) else "SAFE"
            )
        return self.result

    def run_trial(self):
        if self.trajectory is not None:
            from dynamic_metrics import evaluate_dynamic_encounter
            evaluate_dynamic_encounter(self)
            return self.result
        while self.result is None:
            self.step()
        return self.result
