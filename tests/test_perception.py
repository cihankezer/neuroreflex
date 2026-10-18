import math
import unittest

from perception import PerceptionConfig, PerceptionModel, ideal_measurement
from simulation_core import BallState, SimulationCore, calculate_perception
from trajectories import TrueBallState, make_trajectory


class PerceptionTests(unittest.TestCase):
    def observation(self, time_s):
        return TrueBallState(time_s, -time_s, 2 - time_s, 1, -1, -1)

    def test_fixed_seed_reproduces_noise_latency_dropout_and_outliers(self):
        config = PerceptionConfig(0.02, 0.1, 0.04, 0.3, 0.02, 0.2, 5)
        def sample(seed):
            model = PerceptionModel(config, seed)
            return [model.update(self.observation(i * 0.02), i * 0.02)
                    for i in range(100)]
        self.assertEqual(sample(45), sample(45))
        self.assertNotEqual(sample(45), sample(46))
        valid = [p for p in sample(45) if p is not None]
        self.assertTrue(any(p.held for p in valid))
        self.assertTrue(any(p is None for p in sample(45)))

    def test_zero_corruption_exactly_preserves_measurements(self):
        model = PerceptionModel()
        for i in range(10):
            t = i * 0.02
            true = self.observation(t)
            self.assertEqual(model.update(true, t, urgency=0.37),
                             ideal_measurement(true, t, urgency=0.37))

    def test_legacy_static_measurement_matches_original_arithmetic(self):
        for z in (3.2, 0.001, -0.1):
            ball = BallState(z=z)
            p = ideal_measurement(TrueBallState(ball.impact_x, ball.impact_y, z,
                                                0, 0, -ball.speed), 0, legacy_static=True)
            self.assertEqual((p.looming_rate, p.ttc_s, p.bearing_x_deg, p.bearing_y_deg),
                             calculate_perception(ball))

    def test_latency_returns_buffered_historical_state(self):
        model = PerceptionModel(PerceptionConfig(latency_s=0.04))
        self.assertIsNone(model.update(self.observation(0), 0))
        self.assertIsNone(model.update(self.observation(0.02), 0.02))
        old = model.update(self.observation(0.04), 0.04)
        self.assertEqual(old, ideal_measurement(self.observation(0), 0))
        self.assertEqual(model.update(self.observation(0.06), 0.06),
                         ideal_measurement(self.observation(0.02), 0.02))

    def test_sub_tick_latency_waits_until_next_control_sample(self):
        model = PerceptionModel(PerceptionConfig(latency_s=0.01))
        self.assertIsNone(model.update(self.observation(0), 0))
        p = model.update(self.observation(0.02), 0.02)
        self.assertEqual(p.observed_time_s, 0)
        self.assertEqual(p.x, 0)

    def test_total_dropout_has_no_ground_truth_fallback(self):
        model = PerceptionModel(PerceptionConfig(dropout_probability=1))
        for t in (0, 0.02, 0.04):
            self.assertIsNone(model.update(self.observation(t), t))
        sim = SimulationCore(trajectory=make_trajectory("collision_course_center"),
                             perception=PerceptionModel(PerceptionConfig(dropout_probability=1)))
        sim.step(stop_at_plane=False)
        self.assertFalse(sim.danger)
        self.assertIsNone(sim.last_control_state["impact_x"])
        self.assertEqual(sim.robot.shoulder_deg, 35)

    def test_dropout_holds_then_expires_without_extrapolation(self):
        model = PerceptionModel(PerceptionConfig(hold_timeout_s=0.04))
        first = model.update(self.observation(0), 0)
        model.config = PerceptionConfig(dropout_probability=1, hold_timeout_s=0.04)
        for t in (0.02, 0.04):
            held = model.update(self.observation(t), t)
            self.assertTrue(held.held)
            self.assertEqual((held.x, held.ttc_s, held.observed_time_s),
                             (first.x, first.ttc_s, 0))
        self.assertIsNone(model.update(self.observation(0.06), 0.06))

    def test_dropout_does_not_discard_observations_already_in_transit(self):
        model = PerceptionModel(PerceptionConfig(latency_s=0.04, hold_timeout_s=0.02))
        model.update(self.observation(0), 0)
        model.config = PerceptionConfig(latency_s=0.04, dropout_probability=1,
                                        hold_timeout_s=0.02)
        self.assertIsNone(model.update(self.observation(0.02), 0.02))
        self.assertFalse(model.update(self.observation(0.04), 0.04).held)
        self.assertTrue(model.update(self.observation(0.06), 0.06).held)
        self.assertIsNone(model.update(self.observation(0.08), 0.08))

    def test_noise_is_independent_xy_and_ttc_is_positive(self):
        model = PerceptionModel(PerceptionConfig(position_sigma_m=0.04,
                                                ttc_relative_sigma=5), seed=3)
        source = self.observation(0)
        p = model.update(source, 0)
        self.assertNotEqual(p.x - source.x, p.y - source.y)
        self.assertEqual(p.z, source.z)
        self.assertGreaterEqual(p.ttc_s, 1e-6)
        self.assertAlmostEqual(p.looming_rate, 1 / p.ttc_s)

    def test_outliers_scale_existing_noise_without_changing_truth(self):
        source = self.observation(0)
        normal = PerceptionModel(PerceptionConfig(position_sigma_m=0.01), 20).update(source, 0)
        outlier = PerceptionModel(PerceptionConfig(position_sigma_m=0.01,
            outlier_probability=1, outlier_multiplier=10), 20).update(source, 0)
        self.assertNotEqual(normal.x, outlier.x)
        self.assertEqual(source, self.observation(0))

    def test_receding_and_post_plane_objects_have_no_dynamic_looming(self):
        for source in (TrueBallState(0.2, 0.3, 0.2, 0, 0, 1),
                       TrueBallState(0.2, 0.3, -0.2, 0, 0, -1)):
            p = PerceptionModel(PerceptionConfig(ttc_relative_sigma=1)).update(source, 0)
            self.assertEqual(p.ttc_s, math.inf)
            self.assertEqual(p.looming_rate, 0)

    def test_invalid_configuration_and_nonmonotonic_updates(self):
        for kwargs in ({"latency_s": -1}, {"position_sigma_m": float("nan")},
                       {"dropout_probability": 1.1}, {"ttc_relative_sigma": -0.2},
                       {"hold_timeout_s": -1}, {"outlier_multiplier": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PerceptionConfig(**kwargs)
        model = PerceptionModel()
        model.update(self.observation(0), 0)
        with self.assertRaises(ValueError):
            model.update(self.observation(0), 0)


if __name__ == "__main__":
    unittest.main()
