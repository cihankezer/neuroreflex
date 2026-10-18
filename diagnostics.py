"""Passive controller request/clamp diagnostics; never modifies commands."""

from dataclasses import asdict, dataclass
import math


def control_tick_record(time_s, robot, true_state, perceived, reflex_active, details,
                        requested_shoulder, requested_elbow, applied_shoulder,
                        applied_elbow, shoulder_limit, elbow_limit, legacy_static=False):
    # For the nominal controller its position error is neutral minus current.
    # Arbitrary external controllers without target metadata have unknown error.
    row = {
        "time_s": time_s,
        "reflex_active": reflex_active,
        "perception_available": perceived is not None,
        "shoulder_angle_deg": robot.shoulder_deg,
        "elbow_angle_deg": robot.elbow_deg,
        "shoulder_error_deg": details.get("shoulder_error_deg"),
        "elbow_error_deg": details.get("elbow_error_deg"),
        "requested_shoulder_velocity_deg_s": requested_shoulder,
        "requested_elbow_velocity_deg_s": requested_elbow,
        "applied_shoulder_velocity_deg_s": applied_shoulder,
        "applied_elbow_velocity_deg_s": applied_elbow,
        "shoulder_velocity_limit_deg_s": shoulder_limit,
        "elbow_velocity_limit_deg_s": elbow_limit,
        "shoulder_saturated": abs(requested_shoulder) > shoulder_limit,
        "elbow_saturated": abs(requested_elbow) > elbow_limit,
    }
    row.update({"true_" + k: v for k, v in asdict(true_state).items()})
    # Include ideal TTC for interpretation only; it is never sent to the controller.
    from perception import ideal_measurement
    row["true_ttc_s"] = ideal_measurement(true_state, time_s, legacy_static=legacy_static).ttc_s
    row.update({"perceived_" + k: v for k, v in
                asdict(perceived).items()} if perceived else {})
    return row


@dataclass(frozen=True)
class TrialDiagnostics:
    reflex_active_time_s: float
    saturated_reflex_time_s: float
    saturation_ratio: float | None
    shoulder_velocity_residual_rms_deg_s: float | None
    elbow_velocity_residual_rms_deg_s: float | None
    perception_unavailable_ticks: int
    perception_held_ticks: int


def summarize_diagnostics(ticks, control_dt=0.02, duration_s=None):
    """Time-weighted reflex-only saturation and request-minus-clamp RMS.

    Tick commands apply over [tick_end-dt, tick_end], matching the plant.
    A final partial dynamic interval is clipped to the encounter horizon.
    No reflex time yields undefined ratio/RMS (None, exported as blank).
    Joint position stops remain a separate plant constraint, not velocity clips.
    """
    active_time = saturated_time = 0.0
    residuals = [0.0, 0.0]
    for tick in ticks:
        weight = control_dt
        if duration_s is not None:
            weight = max(0.0, min(control_dt, duration_s - (tick["time_s"] - control_dt)))
        if not tick["reflex_active"]:
            continue
        active_time += weight
        if tick["shoulder_saturated"] or tick["elbow_saturated"]:
            saturated_time += weight
        for i, joint in enumerate(("shoulder", "elbow")):
            residuals[i] += weight * (tick[f"requested_{joint}_velocity_deg_s"]
                                      - tick[f"applied_{joint}_velocity_deg_s"])**2
    return TrialDiagnostics(active_time, saturated_time,
                            saturated_time / active_time if active_time else None,
                            math.sqrt(residuals[0] / active_time) if active_time else None,
                            math.sqrt(residuals[1] / active_time) if active_time else None,
                            sum(not t["perception_available"] for t in ticks),
                            sum(bool(t.get("perceived_held")) for t in ticks))
