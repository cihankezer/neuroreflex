"""Lightweight perceived swept-sphere versus current-arm capsule prediction.

No true-state, trajectory label, nominal-pose reference, random sampling, or
future arm motion enters this module's decision.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThreatGateConfig:
    mode: str = "legacy"
    safety_margin_m: float = 0.03
    prediction_horizon_s: float = 1.0

    def __post_init__(self):
        if self.mode not in ("legacy", "geometric", "geometric_memoryless", "geometric_hysteresis"):
            raise ValueError("Unknown threat gate policy.")
        if not math.isfinite(self.safety_margin_m) or self.safety_margin_m < 0:
            raise ValueError("Safety margin must be finite and nonnegative.")
        if not math.isfinite(self.prediction_horizon_s) or self.prediction_horizon_s <= 0:
            raise ValueError("Prediction horizon must be finite and positive.")


@dataclass(frozen=True)
class GateDecision:
    geometric_gate_enabled: bool
    predicted_min_clearance_m: float | None
    predicted_min_clearance_link1_m: float | None
    predicted_min_clearance_link2_m: float | None
    gate_clearance_threshold_m: float
    prediction_horizon_s: float
    ttc_gate_pass: bool
    geometric_gate_pass: bool | None
    final_reflex_gate_pass: bool
    threat_gate_reason: str
    motion_estimate_source: str


def segment_segment_distance(start1, end1, start2, end2):
    """Exact minimum centre-line distance between two closed 3D segments.

    Minimize a convex quadratic on the parameter square [0,1]^2. The minimum
    lies at its unconstrained stationary point (if inside) or on one of four
    edges. Evaluating both handles parallel and zero-length segments cleanly.
    """
    points = (start1, end1, start2, end2)
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("Segment endpoints must have three finite coordinates.")
    dot = lambda p, q: sum(a * b for a, b in zip(p, q))
    u = tuple(b - a for a, b in zip(start1, end1))
    v = tuple(b - a for a, b in zip(start2, end2))
    r = tuple(a - b for a, b in zip(start1, start2))
    a, b, c = dot(u, u), dot(u, v), dot(v, v)
    d, e = dot(u, r), dot(v, r)
    bounded = lambda value: max(0.0, min(1.0, value))
    candidates = [(0.0, bounded(e / c) if c else 0.0),
                  (1.0, bounded((e + b) / c) if c else 0.0),
                  (bounded(-d / a) if a else 0.0, 0.0),
                  (bounded((b - d) / a) if a else 0.0, 1.0)]
    determinant = a * c - b * b
    if determinant > 1e-14 * a * c:
        s, t = (b * e - c * d) / determinant, (a * e - b * d) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    return min(math.sqrt(sum((ri + s * ui - t * vi)**2
               for ri, ui, vi in zip(r, u, v))) for s, t in candidates)


def perceived_velocity(perceived):
    """Use only sensor output, with a TTC/impact ray fallback for older adapters."""
    velocity = tuple(getattr(perceived, "velocity_" + axis, None) for axis in "xyz")
    if all(v is not None and math.isfinite(v) for v in velocity):
        return velocity, "perceived_velocity"
    if math.isfinite(perceived.ttc_s) and perceived.ttc_s > 0:
        return ((perceived.impact_x - perceived.x) / perceived.ttc_s,
                (perceived.impact_y - perceived.y) / perceived.ttc_s,
                -perceived.z / perceived.ttc_s), "impact_ttc_ray"
    return None, "unavailable"


def predict_clearances(position, velocity, links, horizon_s,
                       obstacle_radius_m=0.055, arm_radius_m=0.035):
    """Signed minimum surface clearance per link over 0 <= t <= horizon_s."""
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("Prediction horizon must be finite and positive.")
    if any(not math.isfinite(r) or r <= 0 for r in (obstacle_radius_m, arm_radius_m)):
        raise ValueError("Configured radii must be finite and positive.")
    if len(velocity) != 3 or not all(math.isfinite(v) for v in velocity):
        raise ValueError("Velocity must have three finite components.")
    if len(links) != 2:
        raise ValueError("Current arm geometry must contain two links.")
    end = tuple(p + v * horizon_s for p, v in zip(position, velocity))
    return tuple(segment_segment_distance(position, end, start, finish)
                 - obstacle_radius_m - arm_radius_m for start, finish in links)


def evaluate_threat_gate(perceived, links, ttc_threshold_s=1.0, config=None,
                         obstacle_radius_m=0.055, arm_radius_m=0.035):
    """TTC AND predicted surface clearance <= configured safety margin.

    Legacy mode calculates passive geometry but its decision is exactly the old
    TTC comparison. Missing motion conservatively falls back to TTC-only risk;
    missing perception fails this raw activation gate. The separate stateful
    release policy can retain an already active reflex.
    """
    config = config if config is not None else ThreatGateConfig()
    if not math.isfinite(ttc_threshold_s) or ttc_threshold_s <= 0:
        raise ValueError("TTC threshold must be finite and positive.")
    enabled = config.mode != "legacy"
    clearances = (None, None)
    geometric_pass = None
    ttc_pass = perceived is not None and perceived.ttc_s < ttc_threshold_s
    source, velocity = "unavailable", None
    if perceived is not None:
        velocity, source = perceived_velocity(perceived)
        if velocity is not None:
            clearances = predict_clearances((perceived.x, perceived.y, perceived.z),
                velocity, links, config.prediction_horizon_s, obstacle_radius_m, arm_radius_m)
            geometric_pass = min(clearances) <= config.safety_margin_m + 1e-12
    # Unknown motion cannot be used to prove safety or suppress a TTC threat.
    final = ttc_pass and (not enabled or geometric_pass is not False)
    if perceived is None:
        reason = "no_perception"
    elif not ttc_pass:
        reason = ("receding" if velocity is not None and perceived.z * velocity[2] > 0
                  else "ttc_too_large")
    elif not enabled:
        reason = "legacy_ttc_active"
    elif geometric_pass is None:
        reason = "motion_unavailable_ttc_fallback"
    elif not geometric_pass:
        reason = "geometrically_safe"
    else:
        reason = "predicted_collision_risk"
    return GateDecision(enabled, min(clearances) if clearances[0] is not None else None,
        clearances[0], clearances[1], config.safety_margin_m, config.prediction_horizon_s,
        ttc_pass, geometric_pass, final, reason, source)
