"""Perceived-only reflex release policy; activation remains the existing gate."""

from dataclasses import dataclass
import math

from threat_gate import perceived_velocity


@dataclass(frozen=True)
class ReflexReleaseConfig:
    release_margin_m: float = 0.06
    dwell_s: float = 0.04
    motion_condition: str = "receding"

    def __post_init__(self):
        if not math.isfinite(self.release_margin_m) or self.release_margin_m < 0:
            raise ValueError("Release margin must be finite and nonnegative.")
        if not math.isfinite(self.dwell_s) or self.dwell_s < 0:
            raise ValueError("Release dwell must be finite and nonnegative.")
        if self.motion_condition not in ("none", "receding"):
            raise ValueError("Release motion condition must be none or receding.")


@dataclass(frozen=True)
class ReleaseDecision:
    reflex_state: str
    reflex_ever_activated: bool
    reflex_activation_event: bool
    reflex_release_event: bool
    enter_threshold_m: float
    release_threshold_m: float
    clearance_delta_m: float | None
    release_condition_clearance: bool
    release_condition_motion: bool
    perceived_clearance_rate_m_s: float | None
    safe_release_timer_s: float
    release_dwell_required_s: float
    geometric_gate_raw: bool | None
    activation_gate_raw: bool
    final_reflex_active: bool
    recovery_active: bool
    recovery_started_event: bool
    state_transition_reason: str


def current_clearance_rate(perceived, links):
    """Obstacle distance derivative at the nearest CURRENT arm segment.

    dot(p - closest_point, perceived_velocity) / centre_distance, with the arm
    held fixed. This is obstacle motion evidence, not a predicted robot motion.
    """
    if perceived is None:
        return None
    velocity, _ = perceived_velocity(perceived)
    if velocity is None:
        return None
    position = (perceived.x, perceived.y, perceived.z)
    rates = []
    for start, end in links:
        vector = tuple(b - a for a, b in zip(start, end))
        relative = tuple(p - a for p, a in zip(position, start))
        length_sq = sum(v * v for v in vector)
        fraction = (max(0, min(1, sum(r * v for r, v in zip(relative, vector)) / length_sq))
                    if length_sq else 0.0)
        delta = tuple(r - fraction * v for r, v in zip(relative, vector))
        distance = math.sqrt(sum(d * d for d in delta))
        rate = sum(d * v for d, v in zip(delta, velocity)) / distance if distance else 0.0
        rates.append((distance, rate))
    minimum = min(d for d, _ in rates)
    # At equally close links use the least favourable derivative.
    return min(rate for d, rate in rates if d <= minimum + 1e-12)


class ReflexStateMachine:
    def __init__(self, gate_config, release_config=None):
        self.gate_config = gate_config
        self.config = release_config if release_config is not None else ReflexReleaseConfig()
        self.enabled = gate_config.mode == "geometric_hysteresis"
        if self.enabled and self.config.release_margin_m <= gate_config.safety_margin_m:
            raise ValueError("Release margin must exceed enter margin in hysteresis mode.")
        self.state = "NOMINAL"
        self.ever = False
        self.safe_timer = 0.0
        self.previous_clearance = None
        self.last_time_s = None

    def update(self, gate, perceived, links, time_s, dt=0.02):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Release update interval must be finite and positive.")
        if not math.isfinite(time_s) or time_s < 0 or (self.last_time_s is not None
                and abs(time_s - self.last_time_s - dt) > 1e-9):
            raise ValueError("Release updates must follow the configured control interval.")
        self.last_time_s = time_s
        previous_active = self.state != "NOMINAL"
        clearance = gate.predicted_min_clearance_m
        delta = (clearance - self.previous_clearance if clearance is not None
                 and self.previous_clearance is not None else None)
        self.previous_clearance = clearance
        rate = current_clearance_rate(perceived, links)
        fresh = perceived is not None and not perceived.held
        clearance_safe = clearance is not None and clearance > self.config.release_margin_m + 1e-12
        motion_safe = (perceived is not None and (self.config.motion_condition == "none"
                       or (rate is not None and rate >= -1e-12)))
        raw = gate.final_reflex_gate_pass
        if not self.enabled:
            self.state = "REFLEX_ACTIVE" if raw else "NOMINAL"
            self.safe_timer = 0.0
            reason = gate.threat_gate_reason
        elif not previous_active:
            self.safe_timer = 0.0
            self.state = "REFLEX_ACTIVE" if raw else "NOMINAL"
            reason = "threat_entered" if raw else "nominal"
        elif perceived is None:
            self.state = "PERCEPTION_HOLD"
            self.safe_timer = 0.0
            reason = "perception_temporarily_missing"
        else:
            self.state = "REFLEX_ACTIVE"
            if not fresh:
                self.safe_timer = 0.0
                reason = "held_observation"
            elif not clearance_safe:
                self.safe_timer = 0.0
                reason = "threat_persists" if gate.geometric_gate_pass else "hysteresis_hold"
            elif not motion_safe:
                self.safe_timer = 0.0
                reason = "obstacle_still_approaching"
            else:
                # N consecutive eligible tick intervals, matching the plant's
                # tick-end observation/held-command convention. Never count held
                # estimates or missing intervals as fresh safety evidence.
                self.safe_timer += dt
                if self.safe_timer + 1e-12 >= self.config.dwell_s:
                    self.state = "NOMINAL"
                    reason = "release_confirmed"
                else:
                    reason = "dwell_not_met"
        active = self.state != "NOMINAL"
        activation = active and not previous_active
        release = previous_active and not active
        self.ever = self.ever or active
        recovery = self.ever and not active
        return ReleaseDecision(self.state, self.ever, activation, release,
            self.gate_config.safety_margin_m, self.config.release_margin_m, delta,
            clearance_safe, motion_safe, rate, self.safe_timer, self.config.dwell_s,
            gate.geometric_gate_pass, raw, active, recovery, release, reason)
