import unittest
from types import SimpleNamespace

from motor_metrics import fixed_segment_metrics, tracking_metrics
from run_pid_study import verify_matched_inputs


class MotorMetricsTests(unittest.TestCase):
    def test_input_verifier_checks_sensor_channels_not_arm_relative_metrics(self):
        baseline = [SimpleNamespace(ticks=[{"perceived_x": 1, "perceived_clearance_rate_m_s": 2}])]
        changed_arm = [SimpleNamespace(ticks=[{"perceived_x": 1, "perceived_clearance_rate_m_s": 3}])]
        verify_matched_inputs(changed_arm, baseline)
        changed_sensor = [SimpleNamespace(ticks=[{"perceived_x": 2}])]
        with self.assertRaises(AssertionError):
            verify_matched_inputs(changed_sensor, baseline)
    def test_fixed_segment_overshoot_direction_and_last_exit_settling(self):
        rows = [[9.8, -9.8], [11, -11], [10.1, -10.1], [10, -10]]
        metrics = fixed_segment_metrics(rows, [10, -10], [0, 0])
        self.assertEqual(metrics["shoulder_overshoot_deg"], 1)
        self.assertEqual(metrics["elbow_overshoot_deg"], 1)
        self.assertEqual(metrics["shoulder_settling_s"], .06)

    def test_unsettled_segment_is_undefined(self):
        metrics = fixed_segment_metrics([[0, 0], [1, 1]], [10, 10], [0, 0])
        self.assertIsNone(metrics["shoulder_settling_s"])
        self.assertEqual(metrics["shoulder_overshoot_deg"], 0)

    def test_tracking_weights_partial_tick_and_excludes_hold(self):
        def row(time, error):
            return {"time_s": time, **{f"{j}_{k}": v for j in ("shoulder", "elbow")
                    for k, v in (("error_deg", error), ("angle_deg", 0),
                                 ("saturated", True))},
                    "applied_shoulder_velocity_deg_s": 0, "applied_elbow_velocity_deg_s": 0}
        metrics = tracking_metrics([row(.02, 3), row(.04, 0), row(.06, None)], .03)
        self.assertAlmostEqual(metrics["shoulder_tracking_rmse_deg"], 6 ** .5)
        self.assertAlmostEqual(metrics["shoulder_tracking_time_s"], .03)
        self.assertIsNone(metrics["shoulder_i_rms_deg_s"])


if __name__ == "__main__":
    unittest.main()
