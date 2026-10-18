import argparse
import hashlib
import inspect
import json
from pathlib import Path
import unittest

from motor_controller import (JointGains, PIDConfig, PIDController,
                              add_controller_arguments, config_from_args)
from perception import PerceptionConfig, PerceptionModel
from simulation_core import SimulationCore, student_controller, nominal_controller
from threat_gate import ThreatGateConfig
from trajectories import make_trajectory


def pid(kp=0, ki=0, kd=0, tau=0, elbow=None):
    gains = JointGains(kp, ki, kd)
    return PIDController(PIDConfig(gains, elbow or gains, tau))


class PIDTests(unittest.TestCase):
    def test_zero_error_zero_output(self):
        self.assertEqual(pid(6, 1, .1).update((0, 0), .02, (98, 130))[0], [0, 0])

    def test_signs_and_proportional_only(self):
        controller = pid(4)
        self.assertEqual(controller.update((3, -5), .02, (98, 130))[0], [12, -20])
        for _ in range(10):
            self.assertEqual(controller.update((3, -5), .02, (98, 130))[0], [12, -20])
        self.assertEqual(controller.joints[0].integral, 0)

    def test_integral_accumulation(self):
        controller = pid(0, 2)
        controller.update((5, -5), .02, (98, 130))
        self.assertEqual(controller.update((5, -5), .02, (98, 130))[0], [.4, -.4])

    def test_positive_negative_antiwindup(self):
        for sign in (-1, 1):
            controller = pid(4, 1)
            for _ in range(100):
                requests, diagnostics = controller.update((sign * 10, sign * 10), .02, (5, 5))
                self.assertEqual(requests, [sign * 40, sign * 40])
                self.assertTrue(diagnostics["shoulder_integration_blocked"])
            self.assertEqual(controller.joints[0].integral, 0)

    def test_integral_only_checks_provisional_output(self):
        controller = pid(0, 1)
        controller.update((10, 0), .1, (1, 130))
        self.assertEqual(controller.joints[0].integral, 1)
        controller.update((10, 0), .1, (1, 130))
        self.assertEqual(controller.joints[0].integral, 1)

    def test_reverse_error_unwinds_even_while_saturated(self):
        for sign in (-1, 1):
            controller = pid(0, 1)
            for _ in range(10):
                controller.update((sign * 10, 0), .1, (100, 130))
            requests, details = controller.update((-sign, 0), .1, (1, 130))
            self.assertAlmostEqual(requests[0], sign * 9.9)
            self.assertFalse(details["shoulder_integration_blocked"])

    def test_derivative_response_and_initialization(self):
        controller = pid(0, 0, .1)
        self.assertEqual(controller.update((10, -10), .02, (98, 130))[0], [0, 0])
        self.assertEqual(controller.update((12, -12), .02, (98, 130))[0], [10, -10])

    def test_derivative_filter(self):
        controller = pid(0, 0, 1, tau=.02)
        controller.update((0, 0), .02, (98, 130))
        self.assertEqual(controller.update((2, 0), .02, (98, 130))[0], [50, 0])
        self.assertEqual(controller.update((2, 0), .02, (98, 130))[0], [25, 0])

    def test_reset_clears_integral_and_derivative(self):
        controller = pid(1, 1, .1)
        controller.update((1, 1), .02, (98, 130))
        controller.update((2, 2), .02, (98, 130))
        controller.reset()
        requests, details = controller.update((20, -20), .02, (98, 130))
        self.assertEqual(details["shoulder_d_deg_s"], 0)
        self.assertEqual(details["shoulder_integral_deg_s"], .4)
        self.assertTrue(details["pid_reset"])

    def test_per_joint_independence(self):
        controller = pid(2, 1, .1, elbow=JointGains(3, 0, 0))
        self.assertEqual(controller.update((1, 2), .02, (98, 130))[0], [2.02, 6])
        controller.update((10, 2), .02, (1, 130))
        self.assertEqual(controller.joints[1].integral, 0)
        self.assertEqual(controller.joints[0].integral, .02)

    def test_integral_numerical_bound(self):
        controller = PIDController(PIDConfig(JointGains(0, 1, 0), integral_bound_deg_s=.1))
        for _ in range(100):
            controller.update((10, 0), .02, (98, 130))
        self.assertEqual(controller.joints[0].integral, .1)

    def test_joint_stop_blocks_outward_integral_only(self):
        controller = pid(1, 1)
        _, details = controller.update((1, -1), .02, (98, 130), (120, -150))
        self.assertTrue(details["shoulder_integration_blocked"])
        self.assertTrue(details["elbow_integration_blocked"])
        controller.update((-1, 1), .02, (98, 130), (120, -150))
        self.assertEqual(controller.joints[0].integral, -.02)

    def test_all_mode_transitions_reset(self):
        controller = pid(1, 1, .1)
        for mode in ("REFLEX_ACTIVE", "NOMINAL", "REFLEX_ACTIVE", "PERCEPTION_HOLD", "NOMINAL"):
            controller.prepare(mode, (98, 130))
            self.assertIsNone(controller.joints[0].previous_error)
            self.assertEqual(controller.joints[0].integral, 0)
            controller.update((1, 1), .02, (98, 130))
        controller.prepare("NOMINAL", (98, 130))
        self.assertEqual(controller.joints[0].integral, .02)

    def test_perception_hold_freezes_without_windup(self):
        simulation = SimulationCore(trajectory=make_trajectory("collision_course_center", 1),
            controller="pid", pid_config=PIDConfig(JointGains(6, 1, .1)),
            threat_gate=ThreatGateConfig(mode="geometric_hysteresis"),
            perception=PerceptionModel(PerceptionConfig(burst_start_s=1.3, burst_duration_s=1), 0))
        held = 0
        for _ in range(150):
            old = (simulation.robot.shoulder_deg, simulation.robot.elbow_deg)
            simulation.step(stop_at_plane=False)
            if simulation.last_release_decision.reflex_state == "PERCEPTION_HOLD":
                held += 1
                self.assertEqual(old, (simulation.robot.shoulder_deg, simulation.robot.elbow_deg))
                self.assertEqual(simulation.controller.joints[0].integral, 0)
                self.assertIsNone(simulation.controller.joints[0].previous_error)
        self.assertGreater(held, 0)

    def test_reuses_exact_legacy_tracking_errors_and_raw_request(self):
        simulation = SimulationCore(controller="pid", pid_config=PIDConfig(JointGains(6, 0, 0)))
        simulation.step()
        expected = student_controller(simulation.last_control_state, .02)["diagnostics"]
        tick = simulation.control_diagnostics[-1]
        for joint in ("shoulder", "elbow"):
            self.assertEqual(tick[f"{joint}_error_deg"], expected[f"{joint}_error_deg"])
            self.assertEqual(tick[f"requested_{joint}_velocity_deg_s"], 6 * expected[f"{joint}_error_deg"])
            limit = tick[f"{joint}_velocity_limit_deg_s"]
            self.assertLessEqual(abs(tick[f"applied_{joint}_velocity_deg_s"]), limit)

    def test_baseline_functions_unchanged_and_default_equals_explicit_p(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / "benchmark_results/pid/p_before.json").read_text())
        source = inspect.getsource(student_controller) + inspect.getsource(nominal_controller)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), baseline["p_source_sha256"])
        default, explicit = SimulationCore(), SimulationCore(controller="p")
        for _ in range(30):
            default.step()
            explicit.step()
        self.assertEqual(default.control_diagnostics, explicit.control_diagnostics)
        self.assertEqual(default.robot, explicit.robot)

    def test_invalid_inputs_and_config(self):
        for v in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                JointGains(v, 0, 0)
            with self.assertRaises(ValueError):
                PIDConfig(derivative_tau_s=v)
        for dt in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                pid().update((1, 0), dt, (98, 130))

    def test_cli_shared_and_per_joint_gains(self):
        parser = argparse.ArgumentParser()
        add_controller_arguments(parser)
        args = parser.parse_args(["--controller", "pid", "--pid-kp", "9", "--pid-elbow-kp", "4"])
        config = config_from_args(args)
        self.assertEqual(config.shoulder.kp, 9)
        self.assertEqual(config.elbow.kp, 4)


if __name__ == "__main__":
    unittest.main()
