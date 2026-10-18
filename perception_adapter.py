"""Version 1 perception boundary: image evidence is never metric geometry.

No imports of the event runtime or motor controller are needed here. See
PERCEPTION_INTEGRATION.md for units, clock domains and capability constraints.
"""
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from perception import PerceivedState

EVENT_UNAVAILABLE_FIELDS = (
    "metric_position", "metric_impact", "ttc_s", "metric_velocity",
    "physical_looming_rate", "urgency", "confidence", "camera_to_robot_mapping",
)


@dataclass(frozen=True)
class ReflexPerceptionObservation:
    schema_version: int
    source: str
    clock: str
    timestamp_s: float | None
    event_timestamp_us: int | None
    available: bool
    missing_reason: str | None = None
    held: bool = False
    resolution_wh: tuple[int, int] | None = None
    center_px: tuple[float, float] | None = None
    center_normalized: tuple[float, float] | None = None
    spatial_weights: tuple[tuple[float, ...], ...] | None = None
    on_spike_count: int | None = None
    off_spike_count: int | None = None
    lgmd_potential: float | None = None
    native_detection: bool | None = None
    unavailable_fields: tuple[str, ...] = EVENT_UNAVAILABLE_FIELDS
    # Genuine synthetic measurements only; event adapter leaves this absent.
    metric_state: "PerceivedState | None" = None

    def at_time(self, now_s, max_age_s, *, clock):
        """Explicit expiry in the SAME clock domain; never silently hold events."""
        if clock != self.clock:
            raise ValueError("Cannot compare different perception clock domains")
        if self.timestamp_s is None:
            raise ValueError("No event timestamp is available for this empty native window")
        if not all(math.isfinite(v) for v in (now_s, max_age_s)) or max_age_s < 0:
            raise ValueError("Expiry times must be finite and max_age_s nonnegative")
        if now_s < self.timestamp_s:
            raise ValueError("Observation is from the future in this clock domain")
        if now_s - self.timestamp_s <= max_age_s:
            return self
        return replace(self, available=False, missing_reason="stale", held=False,
                       center_px=None, center_normalized=None, spatial_weights=None,
                       on_spike_count=None, off_spike_count=None, lgmd_potential=None,
                       native_detection=None, metric_state=None,
                       **({"native_gme_state": None, "native_decision_potential": None}
                          if isinstance(self, EventReflexPerceptionObservation) else {}))


@dataclass(frozen=True)
class EventReflexPerceptionObservation(ReflexPerceptionObservation):
    """Expose audited native diagnostics only on the event boundary.

    Synthetic observations retain their exact historical serialized schema.
    """
    native_gme_state: int | None = None
    native_decision_potential: float | None = None


class EventSNNAdapter:
    """Wrap native snapshots, rejecting malformed data and timestamp regressions.

    One adapter per recording/session. No center smoothing, holding, classification,
    TTC derivation, confidence estimate or robot transform is performed.
    """
    def __init__(self):
        self._last_timestamp_us = None

    def adapt(self, *, timestamp_us, resolution_wh, center_x, center_y,
              spatial_weights, on_spikes, off_spikes, lgmd_potential,
              native_detection, native_gme_state=None, native_decision_potential=None):
        if timestamp_us is not None and (isinstance(timestamp_us, bool) or not isinstance(timestamp_us, Integral)):
            raise ValueError("Event timestamp must be integer microseconds")
        timestamp_us = int(timestamp_us) if timestamp_us is not None else None
        if timestamp_us is not None and self._last_timestamp_us is not None and timestamp_us <= self._last_timestamp_us:
            raise ValueError("Duplicate or regressing event timestamp; start a new session adapter")
        if len(resolution_wh) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral)
                                        or v <= 0 for v in resolution_wh):
            raise ValueError("Resolution must be positive integer (width, height)")
        width, height = map(int, resolution_wh)
        arrays = []
        for name, value, allowed in (("spatial_weights", spatial_weights, (0, 1, 2)),
                                     ("on_spikes", on_spikes, (0, 1)),
                                     ("off_spikes", off_spikes, (0, 1))):
            a = np.asarray(value)
            if a.shape != (3, 3) or a.dtype.kind not in "iuf" or not np.isin(a, allowed).all():
                raise ValueError(name + " must be a numeric finite 3x3 native activity map")
            arrays.append(a.astype(float, copy=True))
        weights, on, off = arrays
        if not np.array_equal(weights, on + off):
            raise ValueError("Native spatial map must equal ON + OFF spikes")
        if not isinstance(lgmd_potential, Real) or not math.isfinite(lgmd_potential):
            raise ValueError("LGMD potential must be finite")
        if not isinstance(native_detection, (bool, np.bool_)):
            raise ValueError("Native detection must be boolean")
        if native_gme_state is not None and (isinstance(native_gme_state, (bool, np.bool_)) or
                not isinstance(native_gme_state, Integral) or native_gme_state not in (-1, 0, 1)):
            raise ValueError("Native GME state must be categorical -1, 0, +1 or unavailable")
        if native_decision_potential is not None and (
                not isinstance(native_decision_potential, Real) or not math.isfinite(native_decision_potential)):
            raise ValueError("Native decision potential must be finite or unavailable")
        total = float(weights.sum())
        if total == 0:
            if center_x is not None or center_y is not None:
                raise ValueError("Empty native activity requires both centers to be None")
            center = normalized = None
        else:
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                   for v in (center_x, center_y)):
                raise ValueError("Active native activity requires two finite pixel centers")
            center = float(center_x), float(center_y)
            if not (width/6 <= center[0] <= 5*width/6 and height/6 <= center[1] <= 5*height/6):
                raise ValueError("Center outside native 3x3 cell-centroid range")
            normalized = center[0]/width, center[1]/height
        if timestamp_us is not None:
            self._last_timestamp_us = timestamp_us
        return EventReflexPerceptionObservation(
            schema_version=1, source="event_snn", clock="recorded_event",
            timestamp_s=timestamp_us/1_000_000 if timestamp_us is not None else None,
            event_timestamp_us=timestamp_us,
            available=center is not None and timestamp_us is not None,
            missing_reason="event_timestamp_unavailable" if timestamp_us is None else
                           (None if center else "no_s_layer_activity"),
            resolution_wh=(width, height), center_px=center, center_normalized=normalized,
            spatial_weights=tuple(tuple(float(v) for v in row) for row in weights),
            on_spike_count=int(on.sum()), off_spike_count=int(off.sum()),
            lgmd_potential=float(lgmd_potential), native_detection=bool(native_detection),
            native_gme_state=int(native_gme_state) if native_gme_state is not None else None,
            native_decision_potential=float(native_decision_potential) if native_decision_potential is not None else None,
            unavailable_fields=EVENT_UNAVAILABLE_FIELDS +
                               (("event_timestamp_us",) if timestamp_us is None else ()))


def wrap_synthetic(state, *, timestamp_s):
    """Keep exact synthetic measurements, including intentional held semantics."""
    if not math.isfinite(timestamp_s):
        raise ValueError("Synthetic update time must be finite")
    observed_time = state.observed_time_s if state is not None else timestamp_s
    return ReflexPerceptionObservation(
        schema_version=1, source="synthetic", clock="simulation", timestamp_s=observed_time,
        event_timestamp_us=None, available=state is not None,
        missing_reason=None if state is not None else "synthetic_unavailable",
        held=state.held if state is not None else False, metric_state=state,
        unavailable_fields=("visual_center", "spatial_weights", "native_detection", "confidence",
                            "camera_to_robot_mapping") + (() if state is not None else
                            ("metric_position", "metric_impact", "ttc_s", "urgency")))


class SyntheticObservationSource:
    """Additive wrapper; all corruption, latency and dropout remain upstream."""
    def __init__(self, model):
        self.model = model

    def update(self, true_state, time_s, **kwargs):
        return wrap_synthetic(self.model.update(true_state, time_s, **kwargs), timestamp_s=time_s)


class PerceptionInterfaceMismatch(ValueError):
    pass


def require_legacy_state(observation):
    """Explicit boundary to the existing geometric gate/release/IK interface.

    A missing synthetic sample preserves the old None behavior. Image evidence
    never enters geometric collision logic, even when localization is missing.
    """
    if observation.source != "synthetic":
        raise PerceptionInterfaceMismatch(
            "Visual observation lacks metric position/impact, TTC and urgency; "
            "the current release also uses metric clearance/motion. A visual reflex "
            "policy and calibrated escape mapping are required before actuation.")
    return observation.metric_state if observation.available else None
