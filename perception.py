"""Replaceable trajectory-to-controller adapter with seeded sensor corruption."""

from collections import deque
from dataclasses import dataclass, replace
import math
import random

from trajectories import TrueBallState


@dataclass(frozen=True)
class PerceptionConfig:
    position_sigma_m: float = 0.0
    ttc_relative_sigma: float = 0.0
    latency_s: float = 0.0
    dropout_probability: float = 0.0
    hold_timeout_s: float = 0.06
    outlier_probability: float = 0.0
    outlier_multiplier: float = 5.0
    velocity_sigma_m_s: float = 0.0
    velocity_scale: float = 1.0
    velocity_sign_flip_probability: float = 0.0
    burst_start_s: float = 0.0
    burst_duration_s: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Perception parameters must be finite.")
        for key in ("position_sigma_m", "ttc_relative_sigma", "latency_s", "hold_timeout_s",
                    "velocity_sigma_m_s", "burst_start_s", "burst_duration_s"):
            if getattr(self, key) < 0:
                raise ValueError(key + " must be nonnegative.")
        for key in ("dropout_probability", "outlier_probability", "velocity_sign_flip_probability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(key + " must be between 0 and 1.")
        if self.outlier_multiplier < 1:
            raise ValueError("Outlier multiplier must be at least 1.")
        if self.velocity_scale <= 0:
            raise ValueError("Velocity scale must be positive; use sign flips for wrong direction.")


@dataclass(frozen=True)
class PerceivedState:
    observed_time_s: float
    x: float
    y: float
    z: float
    impact_x: float
    impact_y: float
    ttc_s: float
    looming_rate: float
    bearing_x_deg: float
    bearing_y_deg: float
    urgency: float
    held: bool = False
    # Synthetic velocity estimates have optional corruption (off by default).
    # travel through the same latency/dropout buffer as all other measurements.
    # A real adapter can supply measured velocities or leave them unavailable.
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_z: float | None = None


def ideal_measurement(true_state, time_s, urgency=1.0, legacy_static=False):
    """Static mode retains the original clipped-Z TTC, including after arrival.

    Dynamic mode predicts a future Z=0 crossing when present. For a lateral
    path, TTC is range / radial closing speed about the camera at the origin;
    it is a looming estimate, not an oracle for arm collision. Receding and
    post-plane objects have infinite TTC and zero looming.
    """
    b = true_state
    if legacy_static:
        speed = max(-b.velocity_z, 1e-6)
        distance = max(b.z, 0.01)
        ttc, looming = distance / speed, speed / distance
        impact_x, impact_y = b.x, b.y
    elif b.velocity_z < 0 and b.z > 0:
        ttc = b.z / -b.velocity_z
        looming = 1 / ttc
        impact_x, impact_y = b.x + b.velocity_x * ttc, b.y + b.velocity_y * ttc
    else:
        distance = math.hypot(b.x, b.y, b.z)
        closing = -(b.x * b.velocity_x + b.y * b.velocity_y + b.z * b.velocity_z)
        closing /= max(distance, 1e-6)
        ttc = distance / closing if closing > 0 and b.velocity_z == 0 else math.inf
        looming = 1 / ttc if math.isfinite(ttc) and ttc > 0 else 0.0
        impact_x, impact_y = b.x, b.y
    distance = max(b.z, 0.01)
    return PerceivedState(time_s, b.x, b.y, b.z, impact_x, impact_y, ttc, looming,
                          math.degrees(math.atan2(b.x, distance)),
                          math.degrees(math.atan2(b.y, distance)), urgency,
                          velocity_x=b.velocity_x, velocity_y=b.velocity_y,
                          velocity_z=b.velocity_z)


class PerceptionModel:
    """Sample corruption -> acquisition-time dropout -> buffered transport delay.

    Outputs are held without extrapolation or TTC countdown. Once the last
    delivered observation is older than hold_timeout_s since delivery, None
    explicitly signals unavailability. No ground-truth fallback is supplied.
    """

    def __init__(self, config=None, seed=0):
        self.config = config if config is not None else PerceptionConfig()
        self.rng = random.Random(seed)
        # Keep the original corruption samples matched when only velocity varies.
        self.velocity_rng = random.Random(f"neuroreflex_velocity:{seed}")
        self.pending = deque()
        self.last_valid = None
        self.last_delivery_s = None
        self.last_update_s = None

    def update(self, true_state: TrueBallState, time_s, urgency=1.0, legacy_static=False):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Perception time must be finite and nonnegative.")
        if self.last_update_s is not None and time_s <= self.last_update_s:
            raise ValueError("Perception updates must have increasing times.")
        if not math.isfinite(urgency) or not 0 <= urgency <= 1:
            raise ValueError("Urgency must be between 0 and 1.")
        self.last_update_s = time_s
        c = self.config
        missing = c.dropout_probability > 0 and self.rng.random() < c.dropout_probability
        missing = missing or (c.burst_duration_s > 0
            and c.burst_start_s - 1e-12 <= time_s
            < c.burst_start_s + c.burst_duration_s - 1e-12)
        if not missing:
            estimate = ideal_measurement(true_state, time_s, urgency, legacy_static)
            scale = c.outlier_multiplier if (c.outlier_probability > 0 and
                    self.rng.random() < c.outlier_probability) else 1.0
            nx = self.rng.gauss(0, c.position_sigma_m * scale) if c.position_sigma_m else 0
            ny = self.rng.gauss(0, c.position_sigma_m * scale) if c.position_sigma_m else 0
            ttc = estimate.ttc_s
            if c.ttc_relative_sigma and math.isfinite(ttc):
                ttc = max(1e-6, ttc * (1 + self.rng.gauss(0, c.ttc_relative_sigma * scale)))
            # Preserve exact legacy looming arithmetic at zero TTC noise.
            looming = estimate.looming_rate if ttc == estimate.ttc_s else 1 / ttc
            estimate = replace(estimate, x=estimate.x + nx, y=estimate.y + ny,
                               impact_x=estimate.impact_x + nx, impact_y=estimate.impact_y + ny,
                               ttc_s=ttc, looming_rate=looming)
            if c.position_sigma_m:
                estimate = replace(estimate,
                    bearing_x_deg=math.degrees(math.atan2(estimate.x, max(estimate.z, 0.01))),
                    bearing_y_deg=math.degrees(math.atan2(estimate.y, max(estimate.z, 0.01))))
            if (c.velocity_sigma_m_s or c.velocity_scale != 1.0
                    or c.velocity_sign_flip_probability):
                velocity = [c.velocity_scale * getattr(estimate, "velocity_" + axis)
                    + (self.velocity_rng.gauss(0, c.velocity_sigma_m_s)
                       if c.velocity_sigma_m_s else 0.0) for axis in "xyz"]
                if (c.velocity_sign_flip_probability and
                        self.velocity_rng.random() < c.velocity_sign_flip_probability):
                    velocity = [-v for v in velocity]
                estimate = replace(estimate, **dict(zip(
                    ("velocity_x", "velocity_y", "velocity_z"), velocity)))
            self.pending.append((time_s + c.latency_s, estimate))
        delivered = False
        while self.pending and self.pending[0][0] <= time_s + 1e-12:
            delivery_s, self.last_valid = self.pending.popleft()
            self.last_delivery_s = delivery_s
            delivered = True
        if self.last_valid is None:
            return None
        if not delivered and time_s - self.last_delivery_s > c.hold_timeout_s + 1e-12:
            return None
        return replace(self.last_valid, held=not delivered)
