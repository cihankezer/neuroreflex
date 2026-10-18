"""Physical obstacle trajectories, in metres and seconds (no sensor noise)."""

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class TrueBallState:
    x: float
    y: float
    z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    radius: float = 0.055

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()) or self.radius <= 0:
            raise ValueError("True state must be finite with a positive radius.")

    # Geometry queries historically call the current XY coordinates impact_x/y.
    @property
    def impact_x(self):
        return self.x

    @property
    def impact_y(self):
        return self.y


class Trajectory(Protocol):
    duration_s: float

    def state_at(self, time_s: float) -> TrueBallState: ...


@dataclass(frozen=True)
class LinearTrajectory:
    name: str
    initial: TrueBallState
    duration_s: float

    def __post_init__(self):
        values = tuple(vars(self.initial).values()) + (self.duration_s,)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Trajectory inputs must be finite.")
        if self.duration_s <= 0 or self.initial.radius <= 0:
            raise ValueError("Trajectory duration and radius must be positive.")
        if math.hypot(self.initial.velocity_x, self.initial.velocity_y,
                      self.initial.velocity_z) <= 0:
            raise ValueError("Trajectory velocity must be nonzero.")

    def state_at(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Time must be finite and nonnegative.")
        b = self.initial
        return TrueBallState(b.x + b.velocity_x * time_s,
                             b.y + b.velocity_y * time_s,
                             b.z + b.velocity_z * time_s,
                             b.velocity_x, b.velocity_y, b.velocity_z, b.radius)


TRAJECTORY_TYPES = (
    "collision_course_center", "collision_course_left", "collision_course_right",
    "near_miss", "lateral_pass", "moving_away",
)
ROBUSTNESS_TRAJECTORY_TYPES = (
    "accelerating_approach", "decelerating_approach", "tangential_near_miss",
)
ALL_TRAJECTORY_TYPES = TRAJECTORY_TYPES + ROBUSTNESS_TRAJECTORY_TYPES


@dataclass(frozen=True)
class AcceleratedTrajectory:
    """Constant Z acceleration; deliberately small extension of the protocol."""
    name: str
    initial: TrueBallState
    acceleration_z: float
    duration_s: float

    def __post_init__(self):
        if (not math.isfinite(self.acceleration_z)
                or not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError("Acceleration must be finite and duration positive.")

    def state_at(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Time must be finite and nonnegative.")
        b = self.initial
        return TrueBallState(b.x + b.velocity_x * time_s, b.y + b.velocity_y * time_s,
            b.z + b.velocity_z * time_s + .5 * self.acceleration_z * time_s**2,
            b.velocity_x, b.velocity_y, b.velocity_z + self.acceleration_z * time_s, b.radius)


def make_trajectory(kind, speed=1.2):
    """Speed is initial full 3D magnitude, including oblique approaches.

    Left/right refer to world X relative to the neutral hand. Labels never
    determine safety classification; the evaluator queries physical geometry.
    """
    from simulation_core import forward_kinematics, NEUTRAL_SHOULDER, NEUTRAL_ELBOW

    if kind not in ALL_TRAJECTORY_TYPES:
        raise ValueError("Unknown trajectory type: " + kind)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be finite and positive.")
    _, _, hx, hy = forward_kinematics(NEUTRAL_SHOULDER, NEUTRAL_ELBOW)
    if kind in ("accelerating_approach", "decelerating_approach"):
        acceleration = (-.4 if kind == "accelerating_approach" else .15) * speed**2
        duration = 4.4 / (speed + math.sqrt(speed**2 - 4.4 * acceleration))
        return AcceleratedTrajectory(kind, TrueBallState(hx, hy, 2, 0, 0, -speed),
                                     acceleration, duration)
    if kind == "tangential_near_miss":
        return LinearTrajectory(kind, TrueBallState(hx - 1.2, hy + .13, .04,
                                                    speed, 0, 0), 2.4 / speed)
    if kind == "moving_away":
        return LinearTrajectory(kind, TrueBallState(hx, hy, 0.2, 0, 0, speed), 2 / speed)
    if kind == "lateral_pass":
        return LinearTrajectory(kind, TrueBallState(hx - 2, hy + 0.18, 0.12,
                                                    speed, 0, 0), 4 / speed)
    offset = {"collision_course_left": -0.04,
              "collision_course_right": 0.04}.get(kind, 0)
    target_x, target_y = hx + offset, hy
    if kind == "near_miss":
        target_x, target_y = hx + 0.15, hy + 0.10
    drift_x = {"collision_course_left": 0.25,
               "collision_course_right": -0.25}.get(kind, 0)
    vz = -speed / math.sqrt(1 + (drift_x / 2)**2)
    vx = -vz * drift_x / 2
    return LinearTrajectory(kind, TrueBallState(target_x - drift_x, target_y, 2,
                                                vx, 0, vz), 2.2 / -vz)
