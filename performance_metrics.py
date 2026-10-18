"""Passive whole-encounter metrics for the current planar arm and Z approach."""

from dataclasses import dataclass, replace
import math

from simulation_core import (
    ARM_RADIUS,
    CONTROL_DT,
    ELBOW_MAX,
    NEUTRAL_ELBOW,
    NEUTRAL_SHOULDER,
    RobotState,
    SHOULDER_MAX,
    clamp,
    nominal_controller,
    surface_clearance,
)


GEOMETRY_DT = 0.002  # s; controller still runs at 20 ms
INTRUSION_CLEARANCE = 1.5  # m from the nominal arm surfaces
REACTION_THRESHOLD = 1.0  # deg/s deviation from nominal applied command
CONTACT_TOLERANCE = 1e-12  # m; count numerical tangency as contact


@dataclass(frozen=True)
class EncounterMetrics:
    contact: bool
    min_clearance_m: float
    min_clearance_time_s: float
    nearest_link: int
    first_contact_time_s: float | None
    intrusion_time_s: float | None
    first_reflex_time_s: float | None
    reaction_latency_s: float | None
    reaction_status: str
    baseline_contact: bool
    baseline_min_clearance_m: float
    baseline_first_contact_time_s: float | None
    false_trigger: bool
    reflex_before_actual_contact: bool | None
    reflex_before_baseline_contact: bool | None
    total_joint_motion_deg: float
    max_joint_deviation_deg: float
    duration_s: float


def evaluate_encounter(simulation, geometry_dt=GEOMETRY_DT,
                       intrusion_clearance=INTRUSION_CLEARANCE,
                       reaction_threshold=REACTION_THRESHOLD):
    """Observe a fresh neutral-pose trial until Z < -(ball radius + arm radius).

    Reference ground truth is analytical: without avoidance the nominal arm
    stays neutral, and a fixed-XY ball has its closest approach at Z=0.
    Commands are timestamped at control-tick ends (20 ms resolution).
    Geometry is sampled within ticks using held velocities and joint limits.
    """
    for label, value in (("Geometry step", geometry_dt),
                         ("Intrusion clearance", intrusion_clearance),
                         ("Reaction threshold", reaction_threshold)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(label + " must be finite and positive.")
    if geometry_dt > CONTROL_DT:
        raise ValueError("Geometry step cannot exceed the 20 ms control step.")
    if (simulation.result is not None or simulation.last_control_state is not None
            or simulation.robot != RobotState()):
        raise ValueError("Metrics require a fresh simulation at its neutral pose.")

    initial_ball = replace(simulation.ball)
    radius_sum = initial_ball.radius + ARM_RADIUS
    baseline_clearance, _ = surface_clearance(
        RobotState(), replace(initial_ball, z=0.0)
    )
    xy_distance = baseline_clearance + radius_sum

    def entry_time(surface_distance):
        centre_distance = surface_distance + radius_sum
        if xy_distance > centre_distance + CONTACT_TOLERANCE:
            return None
        entry_z = math.sqrt(max(0.0, centre_distance**2 - xy_distance**2))
        return max(0.0, (initial_ball.z - entry_z) / initial_ball.speed)

    baseline_contact = baseline_clearance <= CONTACT_TOLERANCE
    baseline_contact_time = entry_time(0.0) if baseline_contact else None
    intrusion_time = entry_time(intrusion_clearance)
    minimum, nearest_link = surface_clearance(simulation.robot, initial_ball)
    minimum_time = 0.0
    first_contact = 0.0 if minimum <= CONTACT_TOLERANCE else None
    first_reflex = None
    total_motion = 0.0
    max_deviation = 0.0
    steps = 0
    subdivisions = math.ceil(CONTROL_DT / geometry_dt)

    while simulation.ball.z >= -radius_sum:
        old_robot = replace(simulation.robot)
        start_time = steps * CONTROL_DT
        simulation.step(stop_at_plane=False)
        steps += 1
        end_time = steps * CONTROL_DT

        nominal = nominal_controller(simulation.last_control_state, CONTROL_DT)
        differences = []
        for key, maximum in (("shoulder_velocity_deg_s", SHOULDER_MAX),
                             ("elbow_velocity_deg_s", ELBOW_MAX)):
            limit = simulation.urgency * maximum
            normal_velocity = clamp(nominal[key], -limit, limit)
            differences.append(abs(simulation.last_command[key] - normal_velocity))
        if first_reflex is None and max(differences) >= reaction_threshold:
            first_reflex = end_time

        fractions = [i / subdivisions for i in range(1, subdivisions + 1)]
        plane_fraction = (initial_ball.z / initial_ball.speed - start_time) / CONTROL_DT
        if 0.0 < plane_fraction < 1.0:
            # Include exact plane crossing; important for tangency and clearance.
            fractions.append(plane_fraction)
        for fraction in sorted(fractions):
            time = start_time + fraction * CONTROL_DT
            pose = RobotState(
                clamp(old_robot.shoulder_deg + simulation.last_command[
                    "shoulder_velocity_deg_s"] * fraction * CONTROL_DT, -90.0, 120.0),
                clamp(old_robot.elbow_deg + simulation.last_command[
                    "elbow_velocity_deg_s"] * fraction * CONTROL_DT, -150.0, 150.0),
            )
            ball = replace(initial_ball, z=initial_ball.z - initial_ball.speed * time)
            clearance, link = surface_clearance(pose, ball)
            if clearance < minimum:
                minimum, minimum_time, nearest_link = clearance, time, link
            if first_contact is None and clearance <= CONTACT_TOLERANCE:
                first_contact = time
            max_deviation = max(max_deviation, math.hypot(
                pose.shoulder_deg - NEUTRAL_SHOULDER,
                pose.elbow_deg - NEUTRAL_ELBOW,
            ))
        total_motion += abs(simulation.robot.shoulder_deg - old_robot.shoulder_deg)
        total_motion += abs(simulation.robot.elbow_deg - old_robot.elbow_deg)

    latency = None
    if first_reflex is None:
        status = "no_response" if baseline_contact else "not_triggered"
    elif intrusion_time is None:
        status = "trigger_without_intrusion"
    else:
        latency = first_reflex - intrusion_time
        status = "pre_intrusion" if latency < 0.0 else "responded"
    contact = first_contact is not None
    simulation.result = "COLLISION" if contact else "SAFE"
    return EncounterMetrics(
        contact=contact,
        min_clearance_m=minimum,
        min_clearance_time_s=minimum_time,
        nearest_link=nearest_link,
        first_contact_time_s=first_contact,
        intrusion_time_s=intrusion_time,
        first_reflex_time_s=first_reflex,
        reaction_latency_s=latency,
        reaction_status=status,
        baseline_contact=baseline_contact,
        baseline_min_clearance_m=baseline_clearance,
        baseline_first_contact_time_s=baseline_contact_time,
        false_trigger=not baseline_contact and first_reflex is not None,
        reflex_before_actual_contact=(first_reflex is not None and first_reflex < first_contact)
        if contact else None,
        reflex_before_baseline_contact=(first_reflex is not None and first_reflex < baseline_contact_time)
        if baseline_contact else None,
        total_joint_motion_deg=total_motion,
        max_joint_deviation_deg=max_deviation,
        duration_s=steps * CONTROL_DT,
    )
