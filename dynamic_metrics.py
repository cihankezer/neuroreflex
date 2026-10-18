"""Whole-encounter true geometry for 3D paths; legacy observer stays intact."""

from dataclasses import replace
import math

from performance_metrics import (
    CONTACT_TOLERANCE, EncounterMetrics, GEOMETRY_DT, INTRUSION_CLEARANCE,
    REACTION_THRESHOLD,
)
from simulation_core import (
    ARM_RADIUS, CONTROL_DT, NEUTRAL_ELBOW, NEUTRAL_SHOULDER,
    RobotState, clamp, forward_kinematics, nominal_controller,
    surface_clearance,
)
from trajectories import LinearTrajectory


def _reference_geometry(trajectory, geometry_dt, intrusion_clearance):
    """Neutral-arm safety is label-free. Linear reference minima are exact.

    Point-to-segment projection is piecewise linear; in each region squared
    distance is quadratic. Include its minimizer and projection breakpoints.
    Contact/intrusion onset is then found per link by bisection. Generic paths
    use fine sampling; supply a smaller geometry_dt for grazing encounters.
    """
    duration = trajectory.duration_s
    neutral = RobotState()
    if not isinstance(trajectory, LinearTrajectory):
        count = math.ceil(duration / geometry_dt)
        times = [duration * i / count for i in range(count + 1)]
        clearances = [(surface_clearance(neutral, trajectory.state_at(t))[0], t)
                      for t in times]
        minimum, min_time = min(clearances)
        contact = next((t for d, t in clearances if d <= CONTACT_TOLERANCE), None)
        intrusion = next((t for d, t in clearances if d <= intrusion_clearance), None)
        return minimum, contact, intrusion, [min_time]

    b = trajectory.initial
    ex, ey, hx, hy = forward_kinematics(NEUTRAL_SHOULDER, NEUTRAL_ELBOW)
    segments = [((0, 0, 0), (ex, ey, 0)), ((ex, ey, 0), (hx, hy, 0))]
    velocity = (b.velocity_x, b.velocity_y, b.velocity_z)
    dot = lambda a, c: sum(x * y for x, y in zip(a, c))
    minima, contacts, intrusions, critical = [], [], [], []
    for start, end in segments:
        delta = tuple(c - a for a, c in zip(start, end))
        relative = tuple(c - a for a, c in zip(start, (b.x, b.y, b.z)))
        length_sq = dot(delta, delta)
        f0, fv = dot(relative, delta) / length_sq, dot(velocity, delta) / length_sq
        breaks = [0.0, duration]
        if fv:
            breaks += [t for t in (-f0 / fv, (1 - f0) / fv) if 0 < t < duration]
        breaks.sort()
        candidates = list(breaks)
        for lo, hi in zip(breaks, breaks[1:]):
            projection = f0 + fv * (lo + hi) / 2
            if projection <= 0:
                offset, rate = relative, velocity
            elif projection >= 1:
                offset = tuple(r - d for r, d in zip(relative, delta))
                rate = velocity
            else:
                offset = tuple(r - f0 * d for r, d in zip(relative, delta))
                rate = tuple(v - fv * d for v, d in zip(velocity, delta))
            if dot(rate, rate) > 0:
                candidates.append(clamp(-dot(offset, rate) / dot(rate, rate), lo, hi))

        def clearance(t):
            position = tuple(r + v * t for r, v in zip(relative, velocity))
            fraction = clamp(dot(position, delta) / length_sq, 0, 1)
            separation = tuple(r - fraction * d for r, d in zip(position, delta))
            return math.sqrt(dot(separation, separation)) - b.radius - ARM_RADIUS

        minimum, min_time = min((clearance(t), t) for t in candidates)
        minima.append(minimum)
        critical.extend(candidates)
        for threshold, entries in ((CONTACT_TOLERANCE, contacts),
                                   (intrusion_clearance, intrusions)):
            if clearance(0) <= threshold:
                entries.append(0.0)
            elif minimum <= threshold:
                lo, hi = 0.0, min_time
                for _ in range(60):
                    middle = (lo + hi) / 2
                    if clearance(middle) <= threshold:
                        hi = middle
                    else:
                        lo = middle
                entries.append(hi)
    # Include exact plane crossing for moving-arm geometry too.
    if b.velocity_z:
        plane = -b.z / b.velocity_z
        if 0 <= plane <= duration:
            critical.append(plane)
    return min(minima), min(contacts) if contacts else None, \
        min(intrusions) if intrusions else None, sorted(set(critical))


def evaluate_dynamic_encounter(simulation, geometry_dt=GEOMETRY_DT,
                               intrusion_clearance=INTRUSION_CLEARANCE,
                               reaction_threshold=REACTION_THRESHOLD):
    trajectory = simulation.trajectory
    if trajectory is None:
        raise ValueError("Dynamic evaluation requires a trajectory.")
    for name, value in (("Geometry step", geometry_dt),
                        ("Intrusion clearance", intrusion_clearance),
                        ("Reaction threshold", reaction_threshold),
                        ("Duration", trajectory.duration_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive.")
    if geometry_dt > CONTROL_DT:
        raise ValueError("Geometry step cannot exceed the controller step.")
    if simulation.last_control_state is not None or simulation.result is not None \
            or simulation.robot != RobotState():
        raise ValueError("Metrics require a fresh neutral simulation.")
    duration = trajectory.duration_s
    baseline_minimum, baseline_first_contact, intrusion, critical = _reference_geometry(
        trajectory, geometry_dt, intrusion_clearance)
    baseline_contact = baseline_minimum <= CONTACT_TOLERANCE
    minimum, nearest = surface_clearance(simulation.robot, trajectory.state_at(0))
    minimum_time = 0.0
    geometry_samples = [(0.0, minimum, False)]
    contact = 0.0 if minimum <= CONTACT_TOLERANCE else None
    reflex = None
    total_motion = max_deviation = 0.0
    subdivisions = math.ceil(CONTROL_DT / geometry_dt)
    for tick in range(math.ceil(duration / CONTROL_DT)):
        start = tick * CONTROL_DT
        end = min((tick + 1) * CONTROL_DT, duration)
        old = replace(simulation.robot)
        simulation.step(stop_at_plane=False)
        state = simulation.last_control_state
        nominal = nominal_controller(state, CONTROL_DT)
        limits = simulation.control_diagnostics[-1]
        differences = []
        for joint in ("shoulder", "elbow"):
            limit = limits[f"{joint}_velocity_limit_deg_s"]
            differences.append(abs(simulation.last_command[f"{joint}_velocity_deg_s"] -
                clamp(nominal[f"{joint}_velocity_deg_s"], -limit, limit)))
        if reflex is None and max(differences) >= reaction_threshold:
            # Retain existing control-tick-end timestamps, including last tick.
            reflex = (tick + 1) * CONTROL_DT
        times = [start + i * CONTROL_DT / subdivisions
                 for i in range(1, subdivisions + 1)
                 if start + i * CONTROL_DT / subdivisions <= end]
        times += [t for t in critical if start < t <= end]
        times.append(end)
        for time_s in sorted(set(times)):
            elapsed = time_s - start
            pose = RobotState(
                clamp(old.shoulder_deg + simulation.last_command["shoulder_velocity_deg_s"]
                      * elapsed, -90, 120),
                clamp(old.elbow_deg + simulation.last_command["elbow_velocity_deg_s"]
                      * elapsed, -150, 150))
            clearance, link = surface_clearance(pose, trajectory.state_at(time_s))
            geometry_samples.append((time_s, clearance, limits["recovery_active"]))
            if clearance < minimum:
                minimum, minimum_time, nearest = clearance, time_s, link
            if contact is None and clearance <= CONTACT_TOLERANCE:
                contact = time_s
            max_deviation = max(max_deviation, math.hypot(
                pose.shoulder_deg - NEUTRAL_SHOULDER, pose.elbow_deg - NEUTRAL_ELBOW))
        # Account for a partial last interval at the exact physical horizon.
        simulation.robot = pose
        total_motion += abs(pose.shoulder_deg - old.shoulder_deg)
        total_motion += abs(pose.elbow_deg - old.elbow_deg)
    simulation.ball = trajectory.state_at(duration)
    latency = None
    if reflex is None:
        status = "no_response" if baseline_contact else "not_triggered"
    elif intrusion is None:
        status = "trigger_without_intrusion"
    else:
        latency = reflex - intrusion
        status = "pre_intrusion" if latency < 0 else "responded"
    simulation.result = "COLLISION" if contact is not None else "SAFE"
    from release_metrics import calculate_release_metrics
    simulation.release_metrics = calculate_release_metrics(
        simulation.control_diagnostics, geometry_samples, duration)
    return EncounterMetrics(
        contact is not None, minimum, minimum_time, nearest, contact, intrusion,
        reflex, latency, status, baseline_contact, baseline_minimum,
        baseline_first_contact, not baseline_contact and reflex is not None,
        (reflex is not None and reflex < contact) if contact is not None else None,
        (reflex is not None and reflex < baseline_first_contact) if baseline_contact else None,
        total_motion, max_deviation, duration,
    )
