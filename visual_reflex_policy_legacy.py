"""Unvalidated visual entry/release framework. No physical threat quantities."""
from dataclasses import dataclass
from enum import Enum
import math

from visual_mapping import VisualMapping, VisualZone


class VisualState(str, Enum):
    NOMINAL = "NOMINAL"
    ACTIVE = "VISUAL_REFLEX_ACTIVE"
    HOLD = "VISUAL_HOLD"


@dataclass(frozen=True)
class VisualPolicyConfig:
    entry_potential: float = 3.5
    release_potential: float = 1.0
    entry_requires_native_detection: bool = True
    entry_min_spikes: int = 1
    enter_persistence_us: int = 30_000
    release_persistence_us: int = 120_000
    minimum_active_us: int = 200_000
    max_observation_age_us: int = 30_000

    def __post_init__(self):
        if not isinstance(self.entry_requires_native_detection, bool):
            raise ValueError("entry_requires_native_detection must be boolean")
        if not all(math.isfinite(v) for v in (self.entry_potential, self.release_potential)):
            raise ValueError("Policy thresholds must be finite")
        if self.release_potential >= self.entry_potential:
            raise ValueError("Separate release threshold must be below entry threshold")
        for name in ("entry_min_spikes", "enter_persistence_us", "release_persistence_us",
                     "minimum_active_us", "max_observation_age_us"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.max_observation_age_us <= 0:
            raise ValueError("Observation age limit must be positive")


@dataclass(frozen=True)
class VisualDecision:
    event_time_us: int
    observation_timestamp_us: int | None
    state: str
    zone: str
    fresh: bool
    entry_evidence: bool
    release_evidence: bool
    transition: str | None
    reason: str
    enter_elapsed_us: int
    release_elapsed_us: int
    calibration_status: str


class VisualReflexPolicy:
    """Distinct observations accumulate persistence; polling cannot fabricate it.

    Missing/stale samples freeze movement in HOLD and reset persistence timers.
    Active intent is remembered explicitly through HOLD. Recovery requires a
    fresh localized sample. Release requires observed low activity, never motion.
    """
    def __init__(self, config=None, mapping=None):
        self.config = config or VisualPolicyConfig()
        self.mapping = mapping or VisualMapping()
        self.state = VisualState.NOMINAL
        self.active_intent = False
        self.enter_since = self.release_since = self.active_since = None
        self.last_observation = None
        self.last_event_time = None

    def update(self, now_us, observation=None, *, new_observation=False):
        if self.last_event_time is not None and now_us < self.last_event_time:
            raise ValueError("Policy event time regressed")
        self.last_event_time = now_us
        if new_observation:
            if observation is None or observation.event_timestamp_us != now_us:
                raise ValueError("New evidence must carry its actual event timestamp")
            if self.last_observation is not None and now_us <= self.last_observation.event_timestamp_us:
                raise ValueError("A distinct increasing observation is required")
            if (self.last_observation is not None and
                    now_us-self.last_observation.event_timestamp_us > self.config.max_observation_age_us):
                self.enter_since = self.release_since = None
            self.last_observation = observation
        obs = self.last_observation
        zone = self.mapping.zone(obs, now_us=now_us, max_age_us=self.config.max_observation_age_us)
        fresh = zone != VisualZone.UNKNOWN
        before = self.state
        enter = release = False
        reason = "awaiting_fresh_localization"
        if not fresh:
            self.state = VisualState.HOLD
            self.enter_since = self.release_since = None
        else:
            if self.state == VisualState.HOLD:
                self.state = VisualState.ACTIVE if self.active_intent else VisualState.NOMINAL
            reason = "fresh_sample" if new_observation else "within_explicit_freshness_window"
            potential = obs.lgmd_potential
            counts = (obs.on_spike_count or 0)+(obs.off_spike_count or 0)
            enter = (potential is not None and potential >= self.config.entry_potential
                     and counts >= self.config.entry_min_spikes
                     and (obs.native_detection is True or not self.config.entry_requires_native_detection))
            release = (potential is not None and potential <= self.config.release_potential
                       and obs.native_detection is False)
            if new_observation:
                if not self.active_intent:
                    self.release_since = None
                    if enter:
                        if self.enter_since is None:
                            self.enter_since = now_us
                        if now_us-self.enter_since >= self.config.enter_persistence_us:
                            self.active_intent = True
                            self.active_since = now_us
                            self.state = VisualState.ACTIVE
                            self.enter_since = None
                            reason = "persistent_visual_entry"
                    else:
                        self.enter_since = None
                else:
                    self.enter_since = None
                    if release:
                        if self.release_since is None:
                            self.release_since = now_us
                        if (now_us-self.release_since >= self.config.release_persistence_us
                                and now_us-self.active_since >= self.config.minimum_active_us):
                            self.active_intent = False
                            self.state = VisualState.NOMINAL
                            self.release_since = None
                            reason = "persistent_visual_release"
                    else:
                        self.release_since = None
        # Elapsed persistence uses the last DISTINCT evidence timestamp, not polls.
        evidence_time = obs.event_timestamp_us if obs is not None else now_us
        return VisualDecision(now_us, obs.event_timestamp_us if obs is not None else None,
                              self.state.value, zone.value, fresh, enter, release,
                              before.value+"->"+self.state.value if before != self.state else None,
                              reason, evidence_time-self.enter_since if self.enter_since is not None else 0,
                              evidence_time-self.release_since if self.release_since is not None else 0,
                              self.mapping.config.calibration_status)
