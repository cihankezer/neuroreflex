import math
import unittest

from performance_metrics import evaluate_encounter
from run_static_cases import StaticCase, TrialResult, summarize_results
from simulation_core import (
    ARM_RADIUS,
    BallState,
    RobotState,
    SimulationCore,
    forward_kinematics,
    nominal_controller,
    surface_clearance,
)


def passive(state, dt):
    return {}


def fixed_reflex(state, dt):
    return {"shoulder_velocity_deg_s": 5.0, "elbow_velocity_deg_s": -5.0}


class PerformanceTests(unittest.TestCase):
    def centered_ball(self, z=0.12, speed=1.2):
        _, _, x, y = forward_kinematics(35.0, 55.0)
        return BallState(impact_x=x, impact_y=y, z=z, speed=speed)

    def test_surface_distance_uses_actual_z_and_ball_radius(self):
        ball = self.centered_ball(z=0.2)
        clearance, link = surface_clearance(RobotState(), ball)
        self.assertAlmostEqual(clearance, 0.2 - ball.radius - ARM_RADIUS)
        self.assertEqual(link, 2)
        ball.radius = 0.1
        self.assertAlmostEqual(surface_clearance(RobotState(), ball)[0], 0.065)

    def test_whole_encounter_contact_and_clearance_match_analytic_solution(self):
        simulation = SimulationCore(ball=self.centered_ball(), controller=passive)
        m = evaluate_encounter(simulation)
        self.assertTrue(m.contact)
        self.assertTrue(m.baseline_contact)
        self.assertEqual(simulation.result, "COLLISION")
        self.assertAlmostEqual(m.min_clearance_m, -0.09)
        self.assertAlmostEqual(m.min_clearance_time_s, 0.1)
        self.assertAlmostEqual(m.baseline_first_contact_time_s, 0.025)
        self.assertLessEqual(abs(m.first_contact_time_s - 0.025), 0.002 + 1e-12)
        self.assertGreater(m.duration_s, 0.1)
        self.assertLess(simulation.ball.z, -0.09)
        self.assertEqual(m.reaction_status, "no_response")
        self.assertIsNone(m.reaction_latency_s)
        self.assertFalse(m.reflex_before_actual_contact)
        self.assertEqual(m.total_joint_motion_deg, 0.0)

    def test_contact_present_at_initial_state_is_not_missed(self):
        m = evaluate_encounter(SimulationCore(ball=self.centered_ball(z=0.06),
                                             controller=passive))
        self.assertEqual(m.first_contact_time_s, 0.0)
        self.assertEqual(m.baseline_first_contact_time_s, 0.0)

    def test_exact_tangency_between_control_ticks_counts_as_contact(self):
        ball = self.centered_ball(z=0.1, speed=8.0)
        ball.impact_y += ball.radius + ARM_RADIUS
        m = evaluate_encounter(SimulationCore(ball=ball, controller=passive))
        self.assertTrue(m.contact)
        self.assertTrue(m.baseline_contact)
        self.assertAlmostEqual(m.min_clearance_m, 0.0, places=12)
        self.assertAlmostEqual(m.first_contact_time_s, 0.0125)

    def test_fast_encounter_is_observed_between_control_ticks(self):
        m = evaluate_encounter(SimulationCore(ball=self.centered_ball(speed=30.0),
                                             controller=passive))
        self.assertTrue(m.contact)
        self.assertGreater(m.first_contact_time_s, 0.0)
        self.assertLess(m.first_contact_time_s, 0.02)
        self.assertAlmostEqual(m.min_clearance_m, -0.09)

    def test_reflex_latency_and_motion_are_measured_from_applied_command(self):
        m = evaluate_encounter(SimulationCore(ball=self.centered_ball(),
                                             controller=fixed_reflex))
        self.assertEqual(m.intrusion_time_s, 0.0)
        self.assertEqual(m.first_reflex_time_s, 0.02)
        self.assertEqual(m.reaction_latency_s, 0.02)
        self.assertEqual(m.reaction_status, "responded")
        self.assertTrue(m.reflex_before_baseline_contact)
        self.assertAlmostEqual(m.total_joint_motion_deg, 10.0 * m.duration_s)
        self.assertAlmostEqual(m.max_joint_deviation_deg,
                               math.sqrt(50.0) * m.duration_s)
        zero_urgency = evaluate_encounter(SimulationCore(ball=self.centered_ball(),
                                         urgency=0.0, controller=fixed_reflex))
        self.assertIsNone(zero_urgency.first_reflex_time_s)
        self.assertEqual(zero_urgency.total_joint_motion_deg, 0.0)

    def test_false_trigger_ground_truth_is_independent_of_controller(self):
        ball = self.centered_ball()
        ball.impact_x += 0.3
        active = evaluate_encounter(SimulationCore(ball=ball, controller=fixed_reflex))
        quiet_ball = self.centered_ball()
        quiet_ball.impact_x += 0.3
        quiet = evaluate_encounter(SimulationCore(ball=quiet_ball, controller=nominal_controller))
        self.assertFalse(active.baseline_contact)
        self.assertFalse(quiet.baseline_contact)
        self.assertAlmostEqual(active.baseline_min_clearance_m,
                               quiet.baseline_min_clearance_m)
        self.assertTrue(active.false_trigger)
        self.assertFalse(quiet.false_trigger)
        self.assertEqual(quiet.reaction_status, "not_triggered")

    def test_no_intrusion_and_anticipatory_reactions_are_explicit(self):
        far = BallState(impact_x=3.0, impact_y=3.0, z=0.12)
        m = evaluate_encounter(SimulationCore(ball=far, controller=fixed_reflex))
        self.assertIsNone(m.intrusion_time_s)
        self.assertIsNone(m.reaction_latency_s)
        self.assertEqual(m.reaction_status, "trigger_without_intrusion")
        early = evaluate_encounter(SimulationCore(ball=self.centered_ball(z=1.2),
                                                  controller=fixed_reflex),
                                   intrusion_clearance=0.1)
        self.assertEqual(early.reaction_status, "pre_intrusion")
        self.assertLess(early.reaction_latency_s, 0.0)

    def test_undefined_rates_and_no_response_latencies_are_not_zero(self):
        m = evaluate_encounter(SimulationCore(ball=self.centered_ball(), controller=passive))
        result = TrialResult("center", "COLLISION", m, StaticCase("center", 0, 0, 0, 0.1))
        rows = {r["Metric"]: r for r in summarize_results([result])}
        self.assertEqual(rows["false_trigger_rate"]["Value"], "NA")
        self.assertEqual(rows["false_trigger_rate"]["Denominator"], 0)
        self.assertEqual(rows["reaction_latency_median"]["Value"], "NA")
        self.assertEqual(rows["reaction_latency_samples"]["Value"], 0)
        self.assertEqual(rows["threat_no_response_rate"]["Value"], 100.0)

    def test_geometry_refinement_preserves_control_timing(self):
        coarse = evaluate_encounter(SimulationCore(ball=self.centered_ball(),
                                                   controller=fixed_reflex), geometry_dt=0.002)
        fine = evaluate_encounter(SimulationCore(ball=self.centered_ball(),
                                                 controller=fixed_reflex), geometry_dt=0.001)
        self.assertEqual(coarse.contact, fine.contact)
        self.assertEqual(coarse.first_reflex_time_s, fine.first_reflex_time_s)
        self.assertEqual(coarse.duration_s, fine.duration_s)
        self.assertLess(abs(coarse.min_clearance_m - fine.min_clearance_m), 0.0001)

    def test_invalid_observer_settings_are_rejected(self):
        for settings in ({"geometry_dt": 0.0}, {"geometry_dt": 0.1},
                         {"intrusion_clearance": float("nan")},
                         {"reaction_threshold": -1.0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                evaluate_encounter(SimulationCore(), **settings)


if __name__ == "__main__":
    unittest.main()
