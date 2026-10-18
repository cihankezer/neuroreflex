"""Versioned downstream visual state machines; native evidence is read only."""
from dataclasses import dataclass

from visual_mapping import VisualZone
from visual_reflex_policy_legacy import (
    VisualDecision as LegacyDecision, VisualPolicyConfig as LegacyConfig,
    VisualReflexPolicy as LegacyPolicy, VisualState,
)

LEGACY_VERSION = "visual-policy-v1-legacy-lgmd"
GME_VERSION = "visual-policy-v2-gme-gated"


@dataclass(frozen=True)
class VisualPolicyConfig(LegacyConfig):
    policy_version: str = GME_VERSION
    receding_persistence_us: int = 30_000

    def __post_init__(self):
        super().__post_init__()
        if self.policy_version not in (LEGACY_VERSION, GME_VERSION):
            raise ValueError("Unknown visual policy version")
        if (isinstance(self.receding_persistence_us, bool) or
                not isinstance(self.receding_persistence_us, int) or
                self.receding_persistence_us <= 0):
            raise ValueError("Receding dwell must be a positive integer")
        if self.policy_version == GME_VERSION and (
                self.enter_persistence_us <= 0 or self.release_persistence_us <= 0):
            raise ValueError("Native entry and release require positive temporal persistence")


@dataclass(frozen=True)
class VisualDecision(LegacyDecision):
    policy_version: str
    native_detection: bool | None
    native_gme_state: int | None
    native_decision_potential: float | None
    lgmd_potential: float | None
    receding_evidence: bool
    receding_elapsed_us: int
    entry_zone: str | None
    entry_timestamp_us: int | None
    hold_entry_reason: str | None
    hold_exit_reason: str | None


class VisualReflexPolicy(LegacyPolicy):
    """v1 is preserved verbatim; v2 uses persistent semantic decisions.

    Localization is required to apply a configured target, never threat evidence.
    Missing or stale localization freezes the latched episode in HOLD. There is
    no blind HOLD release timeout: the existing 30 ms observation expiry is the
    deterministic timeout and fresh observations are required to leave HOLD.
    """
    def __init__(self, config=None, mapping=None):
        super().__init__(config or VisualPolicyConfig(), mapping)
        self.receding_since = None
        self.entry_zone = self.entry_timestamp = None

    def reset(self):
        self.__init__(self.config, self.mapping)

    def update(self, now_us, observation=None, *, new_observation=False):
        if getattr(self.config, "policy_version", LEGACY_VERSION) == LEGACY_VERSION:
            return super().update(now_us, observation, new_observation=new_observation)
        if self.last_event_time is not None and now_us < self.last_event_time:
            raise ValueError("Policy event time regressed")
        self.last_event_time = now_us
        if new_observation:
            if observation is None or observation.event_timestamp_us != now_us:
                raise ValueError("New evidence must carry its actual event timestamp")
            if self.last_observation is not None:
                gap = now_us-self.last_observation.event_timestamp_us
                if gap <= 0:
                    raise ValueError("A distinct increasing observation is required")
                if gap > self.config.max_observation_age_us:
                    self.enter_since = self.release_since = self.receding_since = None
            self.last_observation = observation
        obs = self.last_observation
        zone = self.mapping.zone(obs, now_us=now_us, max_age_us=self.config.max_observation_age_us)
        fresh = zone != VisualZone.UNKNOWN
        # Signal freshness and localization availability are distinct. A genuine
        # no-S-layer sample can still carry diagnostic false detection/direction.
        signal_fresh = (obs is not None and obs.source == "event_snn" and
                        obs.event_timestamp_us is not None and
                        0 <= now_us-obs.event_timestamp_us <= self.config.max_observation_age_us)
        detection = obs.native_detection if signal_fresh else None
        direction = obs.native_gme_state if signal_fresh else None
        before = self.state
        enter = release = receding = False
        hold_entry = hold_exit = None
        reason = "within_explicit_freshness_window"
        if not fresh:
            reason = ("observation_missing" if obs is None else
                      "observation_stale" if not signal_fresh else
                      obs.missing_reason or "localization_ineligible")
            if before != VisualState.HOLD:
                hold_entry = reason
            self.state = VisualState.HOLD
            self.enter_since = self.release_since = self.receding_since = None
        else:
            if before == VisualState.HOLD:
                self.state = VisualState.ACTIVE if self.active_intent else VisualState.NOMINAL
                hold_exit = "fresh_localization_resume_latched_intent" if self.active_intent else "fresh_localization_no_active_intent"
                reason = hold_exit
            elif new_observation:
                reason = "fresh_native_sample"
            enter = detection is True
            release = detection is False
            # Native decision is authoritative. Contradictory diagnostics cannot
            # override a true detection; direction is categorical, never velocity.
            receding = release and direction == -1
            if new_observation:
                if not self.active_intent:
                    self.release_since = self.receding_since = None
                    self.enter_since = (now_us if self.enter_since is None else self.enter_since) if enter else None
                    if self.enter_since is not None and now_us-self.enter_since >= self.config.enter_persistence_us:
                        self.active_intent = True
                        self.active_since = now_us
                        self.entry_zone, self.entry_timestamp = zone.value, now_us
                        self.state = VisualState.ACTIVE
                        self.enter_since = None
                        reason = "persistent_native_detection_entry"
                else:
                    self.enter_since = None
                    self.release_since = (now_us if self.release_since is None else self.release_since) if release else None
                    self.receding_since = (now_us if self.receding_since is None else self.receding_since) if receding else None
                    normal_ready = (self.release_since is not None and
                                    now_us-self.release_since >= self.config.release_persistence_us)
                    receding_ready = (self.receding_since is not None and
                                      now_us-self.receding_since >= self.config.receding_persistence_us)
                    if (now_us-self.active_since >= self.config.minimum_active_us and
                            (normal_ready or receding_ready)):
                        self.active_intent = False
                        self.state = VisualState.NOMINAL
                        self.release_since = self.receding_since = None
                        self.entry_zone = self.entry_timestamp = self.active_since = None
                        reason = "persistent_receding_release" if receding_ready else "persistent_native_detection_false_release"
        evidence_time = obs.event_timestamp_us if obs is not None else now_us
        elapsed = lambda since: evidence_time-since if since is not None else 0
        return VisualDecision(
            now_us, obs.event_timestamp_us if obs is not None else None,
            self.state.value, zone.value, fresh, enter, release,
            before.value+"->"+self.state.value if before != self.state else None,
            reason, elapsed(self.enter_since), elapsed(self.release_since),
            self.mapping.config.calibration_status, self.config.policy_version,
            detection, direction, obs.native_decision_potential if signal_fresh else None,
            obs.lgmd_potential if signal_fresh else None, receding, elapsed(self.receding_since),
            self.entry_zone, self.entry_timestamp, hold_entry, hold_exit,
        )
