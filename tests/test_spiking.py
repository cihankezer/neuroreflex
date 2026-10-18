import argparse
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
from motor_controller import PIDController, add_controller_arguments, load_conventional_reference
from simulation_core import nominal_controller, student_controller
from spiking_config import SpikingConfig, load_spiking_config, spiking_config_from_args

ROOT = Path(__file__).resolve().parents[1]
HAS_LAVA = importlib.util.find_spec("lava") is not None


class SpikingConfigurationTests(unittest.TestCase):
    def test_default_controller_preserved_and_spiking_explicit(self):
        parser = argparse.ArgumentParser()
        add_controller_arguments(parser)
        self.assertEqual(parser.parse_args([]).controller, "p")
        args = parser.parse_args(["--controller", "spiking-pid", "--spiking-neurons", "100"])
        self.assertEqual(spiking_config_from_args(args).neurons_per_ensemble, 100)
        with self.assertRaises(ValueError):
            spiking_config_from_args(parser.parse_args(["--controller", "spiking-pid", "--pid-kd", ".2"]))

    def test_invalid_neural_timing_and_population(self):
        for kwargs in ({"neural_dt_s": 0}, {"neural_dt_s": .003}, {"neurons_per_ensemble": 9},
                       {"tau_short_s": .08}, {"error_radius_deg": (0, 100)}):
            with self.assertRaises(ValueError):
                SpikingConfig(**kwargs)

    def test_frozen_conventional_source_and_preset_unchanged(self):
        expected = json.loads((ROOT/"spiking_frozen_sources.json").read_text())
        self.assertEqual(hashlib.sha256(inspect.getsource(PIDController).encode()).hexdigest(), expected["pid_class_sha256"])
        self.assertEqual(hashlib.sha256((ROOT/"conventional_reference.json").read_bytes()).hexdigest(), expected["conventional_reference_sha256"])

    def test_legacy_p_source_unchanged(self):
        expected = json.loads((ROOT/"benchmark_results/pid/p_before.json").read_text())
        actual = hashlib.sha256((inspect.getsource(student_controller)+inspect.getsource(nominal_controller)).encode()).hexdigest()
        self.assertEqual(actual, expected["p_source_sha256"])


@unittest.skipUnless(HAS_LAVA, "Requires optional pinned Python 3.10/Lava environment.")
class SpikingGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from spiking_controller import run_spike_batch
        cls.config = load_spiking_config()
        # Known steps, ramp, zero, reset and paired joints; 3.2 s simulated time.
        cls.errors = np.zeros((320, 2))
        cls.errors[20:100] = (20, -20)
        cls.errors[100:180] = (-20, 20)
        cls.errors[180:240, 0] = np.arange(60)*.2
        cls.errors[180:240, 1] = -np.arange(60)*.2
        cls.errors[260:280] = cls.config.error_radius_deg
        cls.errors[280:300] = np.array(cls.config.error_radius_deg)*1.2
        cls.modes = np.ones(320)
        cls.modes[240:260] = 2
        cls.modes[300:] = 0
        cls.first = run_spike_batch(cls.errors, cls.modes, cls.config)
        cls.second = run_spike_batch(cls.errors, cls.modes, cls.config)
        cls.history = cls.first["history"]

    def test_deterministic_actual_spike_activity(self):
        np.testing.assert_array_equal(self.first["raw_projection"], self.second["raw_projection"])
        np.testing.assert_array_equal(self.history, self.second["history"])
        self.assertGreater(self.history[:, :, 6].sum(), 0)
        self.assertGreater(self.history[:, :, 7].sum(), 0)

    def test_zero_error_is_zero_without_host_deadband(self):
        np.testing.assert_array_equal(self.history[:20, :, :6], 0)
        self.assertGreater(self.history[:20, :, 6].sum(), 0)  # Background pair spikes cancel.

    def test_positive_negative_representation(self):
        represented = self.history[40:100, :, 0].mean(axis=0)
        np.testing.assert_allclose(represented, [20, -20], atol=2)

    def test_radius_and_logged_input_clipping(self):
        self.assertFalse(self.first["input_clipped"][260:280].any())
        self.assertTrue(self.first["input_clipped"][280:300].all())
        np.testing.assert_allclose(self.history[290:300, :, 0].mean(axis=0), self.config.error_radius_deg, rtol=.08)

    def test_proportional_sign(self):
        self.assertGreater(self.history[40:100, 0, 1].mean(), 0)
        self.assertLess(self.history[40:100, 1, 1].mean(), 0)

    def test_neural_proportional_transform_exact_scaling(self):
        np.testing.assert_allclose(self.history[:240, :, 1], 5.25*self.history[:240, :, 0], atol=1e-10)

    def test_constant_error_derivative_has_no_systematic_drift(self):
        # Residual spike ripple is measured rather than falsely requiring instantaneous zero.
        self.assertLess(np.max(np.abs(self.history[70:100, :, 4].mean(axis=0))), 3.)

    def test_derivative_step_response(self):
        self.assertGreater(self.history[20:24, 0, 4].mean(), 0)
        self.assertLess(self.history[20:24, 1, 4].mean(), 0)

    def test_derivative_ramp_response(self):
        self.assertGreater(self.history[210:240, 0, 4].mean(), 0)
        self.assertLess(self.history[210:240, 1, 4].mean(), 0)

    def test_dual_filter_subtraction_is_actual_derivative(self):
        np.testing.assert_allclose(self.history[:, :, 4], .1*self.config.derivative_scale_per_s*(
            self.history[:, :, 2]-self.history[:, :, 3]), atol=1e-10)

    def test_transition_resets_derivative_and_integral_influence(self):
        np.testing.assert_array_equal(self.history[260, :, 4], 0)
        np.testing.assert_array_equal(self.history[300, :, 4], 0)

    def test_nominal_gain_is_frozen(self):
        np.testing.assert_allclose(self.history[300:, :, 1], 2.5*self.history[300:, :, 0], atol=1e-10)

    def test_perception_hold_zero_and_no_spikes(self):
        np.testing.assert_array_equal(self.history[240:260], 0)

    def test_joint_independence(self):
        from spiking_controller import run_spike_batch
        changed = self.errors.copy(); changed[:, 1] *= .37
        altered = run_spike_batch(changed, self.modes, self.config)
        np.testing.assert_array_equal(altered["history"][:, 0], self.history[:, 0])

    def test_online_adapter_hold_and_reference_cannot_drive_actuator(self):
        from spiking_controller import SpikingController
        controller = SpikingController(self.config)
        try:
            controller.prepare("REFLEX_ACTIVE", (1, 1))
            request, details = controller.update((20, -20), .02, (1, 1))
            self.assertGreater(abs(request[0]), 1)  # Existing actuator performs clipping.
            self.assertEqual(details["shoulder_i_deg_s"], 0)
            initial_request = list(request)
            controller.update((60, -60), .02, (1,1))
            controller.begin_trial()
            controller.prepare("REFLEX_ACTIVE", (1,1))
            request, _ = controller.update((20,-20), .02, (1,1))
            np.testing.assert_array_equal(request, initial_request)
            # Mute ACTUAL output-spike decoder. Passive PD remains nonzero.
            controller.process.output_decoder.set(np.zeros((2, self.config.neurons_per_ensemble//2)))
            request, details = controller.update((20, -20), .02, (1, 1))
            np.testing.assert_array_equal(request, 0)
            self.assertNotEqual(details["shoulder_reference_requested_deg_s"], 0)
            controller.prepare("PERCEPTION_HOLD", (1,1))
            request, details = controller.update((20,-20), .02, (1,1))
            np.testing.assert_array_equal(request, 0)
            self.assertEqual(details["neural_steps"], 0)
        finally:
            controller.close()

    def test_dynamic_runtime_reuse_preserves_trial_ticks(self):
        from run_dynamic_cases import run_dynamic_cases
        common=dict(num_trials=2,seed=23,speed_min=6,speed_max=6,
                    controller="spiking-pid",spiking_config=self.config)
        fresh=run_dynamic_cases(**common)
        reused=run_dynamic_cases(**common,reuse_spiking_runtime=True)
        self.assertEqual(fresh,reused)


if __name__ == "__main__":
    unittest.main()
