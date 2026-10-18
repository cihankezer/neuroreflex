"""Experimental raster zones and explicit robot TARGET templates, not sensing."""
from dataclasses import dataclass
from enum import Enum
import math

from simulation_core import clamp_workspace, inverse_kinematics


class VisualZone(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class VisualMappingConfig:
    left_max_x: float = 1/3
    right_min_x: float = 2/3
    y_range: tuple | None = None
    calibration_status: str = "uncalibrated"
    calibration_id: str = "illustrative-simulation-templates-v1"
    calibration_evidence: str = "No controlled camera-to-robot calibration performed"
    left_target_xy_m: tuple = (.55, .32)
    center_target_xy_m: tuple = (.30, .32)
    right_target_xy_m: tuple = (.14, .58)

    def __post_init__(self):
        if not 0 < self.left_max_x < self.right_min_x < 1:
            raise ValueError("Zone thresholds require 0 < left < right < 1")
        if self.calibration_status not in ("uncalibrated", "calibrated"):
            raise ValueError("Unknown calibration status")
        if not self.calibration_id or not self.calibration_evidence:
            raise ValueError("Explicit calibration metadata is required")
        if self.y_range is not None:
            if len(self.y_range) != 2 or not 0 <= self.y_range[0] < self.y_range[1] <= 1:
                raise ValueError("Optional y eligibility range must lie in [0,1]")
            object.__setattr__(self, "y_range", tuple(self.y_range))
        for name in ("left_target_xy_m", "center_target_xy_m", "right_target_xy_m"):
            target = tuple(getattr(self, name))
            if len(target) != 2 or not all(math.isfinite(v) for v in target):
                raise ValueError("Supply finite XY robot target templates")
            # Reject inaccessible templates instead of silently inventing geometry.
            if not all(math.isclose(a, b, abs_tol=1e-12) for a, b in
                       zip(target, clamp_workspace(*target))):
                raise ValueError("Robot target template lies outside the existing workspace")
            q = inverse_kinematics(*target)
            if not -90 <= q[0] <= 120 or not -150 <= q[1] <= 150:
                raise ValueError("IK template lies outside existing joint bounds")
            object.__setattr__(self, name, target)


class VisualMapping:
    def __init__(self, config=None):
        self.config = config or VisualMappingConfig()

    def zone(self, observation, *, now_us, max_age_us):
        if observation is None or observation.source != "event_snn" or not observation.available:
            return VisualZone.UNKNOWN
        timestamp = observation.event_timestamp_us
        if timestamp is None or not 0 <= now_us-timestamp <= max_age_us:
            return VisualZone.UNKNOWN
        center = observation.center_normalized
        if center is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in center):
            return VisualZone.UNKNOWN
        x, y = center
        if self.config.y_range is not None and not self.config.y_range[0] <= y <= self.config.y_range[1]:
            return VisualZone.UNKNOWN
        if x < self.config.left_max_x:
            return VisualZone.LEFT
        if x > self.config.right_min_x:
            return VisualZone.RIGHT
        return VisualZone.CENTER

    def target(self, zone):
        return {VisualZone.LEFT: self.config.left_target_xy_m,
                VisualZone.CENTER: self.config.center_target_xy_m,
                VisualZone.RIGHT: self.config.right_target_xy_m}.get(zone)
