"""Passive tracking statistics, separate from unchanged safety metric schemas."""

import math

from motor_controller import JOINTS, JOINT_BOUNDS


def tracking_metrics(ticks, duration_s, dt=.02):
    result = {}
    for joint, bounds in zip(JOINTS, JOINT_BOUNDS):
        weighted = [(row, max(0.0, min(dt, duration_s - (row["time_s"] - dt))))
                    for row in ticks if row.get(f"{joint}_error_deg") is not None]
        total = sum(w for _, w in weighted)
        if not total:
            continue
        def rms(field):
            samples = [(row[field], w) for row, w in weighted if row.get(field) is not None]
            denominator = sum(w for _, w in samples)
            return math.sqrt(sum(v * v * w for v, w in samples) / denominator) if denominator else None
        result.update({
            f"{joint}_tracking_rmse_deg": rms(f"{joint}_error_deg"),
            f"{joint}_peak_error_deg": max(abs(row[f"{joint}_error_deg"]) for row, _ in weighted),
            f"{joint}_saturation_fraction": sum(w for row, w in weighted if row[f"{joint}_saturated"]) / total,
            f"{joint}_i_rms_deg_s": rms(f"{joint}_i_deg_s"),
            f"{joint}_d_rms_deg_s": rms(f"{joint}_d_deg_s"),
            f"{joint}_tracking_time_s": total,
            f"{joint}_joint_stop_count": sum(
                row[f"{joint}_angle_deg"] + row[f"applied_{joint}_velocity_deg_s"] * w < bounds[0] - 1e-10
                or row[f"{joint}_angle_deg"] + row[f"applied_{joint}_velocity_deg_s"] * w > bounds[1] + 1e-10
                for row, w in weighted),
        })
    return result


def fixed_segment_metrics(angles, target, initial, dt=.02, tolerance=.5):
    """Post-update samples; settling = first tick after the LAST tolerance exit.

    A segment never settled by its horizon returns None. Overshoot is angular
    excursion beyond the fixed target in the commanded direction, in degrees.
    """
    result = {}
    for i, joint in enumerate(JOINTS):
        errors = [target[i] - q[i] for q in angles]
        outside = [k for k, e in enumerate(errors) if abs(e) > tolerance]
        settling = (outside[-1] + 2) * dt if outside else dt
        if settling > len(angles) * dt:
            settling = None
        direction = 1 if target[i] >= initial[i] else -1
        result[f"{joint}_settling_s"] = settling
        result[f"{joint}_overshoot_deg"] = max(0.0, max(direction * (q[i] - target[i]) for q in angles))
        signs = [1 if e > 0 else -1 for e in errors if abs(e) > tolerance]
        result[f"{joint}_target_crossings_outside_tolerance"] = sum(
            a != b for a, b in zip(signs, signs[1:]))
    return result
