import csv
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics import summarize_diagnostics
from dynamic_metrics import evaluate_dynamic_encounter
from performance_metrics import evaluate_encounter
from perception import PerceptionConfig, PerceptionModel, ideal_measurement
from run_dynamic_cases import main, run_dynamic_cases
from run_static_cases import DEFAULT_WORKBOOK, case_to_ball, load_static_cases
from simulation_core import SimulationCore, surface_clearance, RobotState
from trajectories import LinearTrajectory, TRAJECTORY_TYPES, TrueBallState, make_trajectory


ROOT = Path(__file__).resolve().parents[1]


def passive(state, dt):
    return {}


def forced_reflex(state, dt):
    return {"shoulder_velocity_deg_s": 5, "elbow_velocity_deg_s": -5}


class TrajectoryTests(unittest.TestCase):
    def test_collision_courses_intersect_neutral_arm_and_preserve_speed(self):
        for kind in TRAJECTORY_TYPES[:3]:
            trajectory = make_trajectory(kind, 1.6)
            source = trajectory.state_at(0)
            self.assertAlmostEqual(math.hypot(source.velocity_x, source.velocity_y,
                                              source.velocity_z), 1.6)
            at_plane = trajectory.state_at(-source.z / source.velocity_z)
            self.assertLess(surface_clearance(RobotState(), at_plane)[0], 0)
            m = evaluate_dynamic_encounter(SimulationCore(trajectory=trajectory, controller=passive))
            self.assertTrue(m.baseline_contact)
            self.assertTrue(m.contact)

    def test_moving_away_increases_distance_and_has_no_reflex(self):
        path = make_trajectory("moving_away", 2)
        self.assertGreater(path.state_at(0.1).z, path.state_at(0).z)
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=path))
        self.assertFalse(m.contact)
        self.assertFalse(m.baseline_contact)
        self.assertIsNone(m.first_reflex_time_s)

    def test_lateral_pass_moves_in_x_without_changing_z(self):
        path = make_trajectory("lateral_pass", 2)
        self.assertGreater(path.state_at(0.1).x, path.state_at(0).x)
        self.assertEqual(path.state_at(0.1).z, path.state_at(0).z)
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=path, controller=passive))
        self.assertFalse(m.baseline_contact)
        self.assertFalse(m.contact)
        self.assertGreater(m.baseline_min_clearance_m, 0)

    def test_near_miss_has_positive_neutral_geometry_clearance(self):
        path = make_trajectory("near_miss", 2)
        self.assertLess(path.state_at(0).velocity_z, 0)
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=path, controller=passive))
        self.assertFalse(m.baseline_contact)
        self.assertFalse(m.contact)
        self.assertAlmostEqual(m.baseline_min_clearance_m, math.hypot(0.15, 0.10) - 0.09)

    def test_general_3d_velocity_and_projected_impact(self):
        path = LinearTrajectory("custom", TrueBallState(1, 2, 2, 1, -1, -2), 2)
        self.assertEqual(path.state_at(0.5), TrueBallState(1.5, 1.5, 1, 1, -1, -2))
        p = ideal_measurement(path.state_at(0), 0)
        self.assertEqual((p.impact_x, p.impact_y, p.ttc_s), (2, 1, 1))

    def test_invalid_trajectory_parameters(self):
        for speed in (0, -1, math.inf):
            with self.assertRaises(ValueError):
                make_trajectory("near_miss", speed)
        with self.assertRaises(ValueError):
            make_trajectory("unknown")
        with self.assertRaises(ValueError):
            TrueBallState(math.nan, 0, 1, 0, 0, -1)


class DynamicMetricsTests(unittest.TestCase):
    def test_collision_geometry_is_independent_of_noisy_perception(self):
        trajectory = make_trajectory("collision_course_center", 6)
        clean = SimulationCore(trajectory=trajectory, controller=passive)
        noisy = SimulationCore(trajectory=trajectory, controller=passive,
            perception=PerceptionModel(PerceptionConfig(position_sigma_m=100, ttc_relative_sigma=0.2), 5))
        self.assertEqual(evaluate_dynamic_encounter(clean), evaluate_dynamic_encounter(noisy))
        self.assertTrue(noisy.result == "COLLISION")
        self.assertNotEqual(noisy.control_diagnostics[0]["true_x"],
                            noisy.control_diagnostics[0]["perceived_x"])

    def test_false_triggers_use_true_trajectory_geometry_not_labels(self):
        for kind in ("near_miss", "lateral_pass", "moving_away"):
            with self.subTest(kind=kind):
                trajectory = make_trajectory(kind, 6)
                active = evaluate_dynamic_encounter(SimulationCore(trajectory=trajectory,
                                                                   controller=forced_reflex))
                quiet = evaluate_dynamic_encounter(SimulationCore(trajectory=trajectory,
                                                                  controller=passive))
                self.assertFalse(active.baseline_contact)
                self.assertTrue(active.false_trigger)
                self.assertFalse(quiet.false_trigger)
        mislabeled = replace(make_trajectory("collision_course_center", 6), name="moving_away")
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=mislabeled, controller=forced_reflex))
        self.assertTrue(m.baseline_contact)
        self.assertFalse(m.false_trigger)

    def test_exact_lateral_tangency_between_ticks(self):
        path = LinearTrajectory("lateral_pass", TrueBallState(-0.1, -0.09, 0, 8, 0, 0), 0.1)
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=path, controller=passive))
        self.assertTrue(m.contact)
        self.assertTrue(m.baseline_contact)
        self.assertAlmostEqual(m.min_clearance_m, 0, places=12)
        self.assertAlmostEqual(m.min_clearance_time_s, 0.0125)

    def test_generic_trajectory_protocol_uses_true_state_sampling(self):
        class CurvedPath:
            duration_s = 0.1
            def state_at(self, t):
                return TrueBallState(0, 0, 0.2 - 3 * t, math.sin(t), 0, -3)
        # Not a LinearTrajectory: arbitrary state_at implementations remain usable.
        m = evaluate_dynamic_encounter(SimulationCore(trajectory=CurvedPath(), controller=passive))
        self.assertTrue(m.contact)
        self.assertTrue(m.baseline_contact)

    def test_geometry_refinement_keeps_control_timing_and_baseline(self):
        path = make_trajectory("collision_course_left", 6)
        a = evaluate_dynamic_encounter(SimulationCore(trajectory=path), geometry_dt=0.002)
        b = evaluate_dynamic_encounter(SimulationCore(trajectory=path), geometry_dt=0.001)
        self.assertEqual(a.first_reflex_time_s, b.first_reflex_time_s)
        self.assertEqual(a.baseline_min_clearance_m, b.baseline_min_clearance_m)
        self.assertEqual(a.baseline_first_contact_time_s, b.baseline_first_contact_time_s)
        self.assertEqual(a.contact, b.contact)
        self.assertLess(abs(a.min_clearance_m - b.min_clearance_m), 0.001)

    def test_partial_final_tick_ends_at_exact_duration(self):
        path = LinearTrajectory("custom", TrueBallState(3, 3, 0.2, 0, 0, 1), 0.025)
        sim = SimulationCore(trajectory=path, controller=forced_reflex)
        m = evaluate_dynamic_encounter(sim)
        self.assertAlmostEqual(m.duration_s, 0.025)
        self.assertAlmostEqual(sim.robot.shoulder_deg, 35 + 5 * 0.025)
        self.assertEqual(sim.ball, path.state_at(0.025))
        self.assertAlmostEqual(m.total_joint_motion_deg, 10 * 0.025)

    def test_full_dropout_reports_threat_without_response(self):
        sim = SimulationCore(trajectory=make_trajectory("collision_course_center", 6),
                             perception=PerceptionModel(PerceptionConfig(dropout_probability=1)))
        m = evaluate_dynamic_encounter(sim)
        self.assertTrue(m.contact)
        self.assertEqual(m.reaction_status, "no_response")
        self.assertIsNone(m.reaction_latency_s)


class DiagnosticsTests(unittest.TestCase):
    def tick(self, time_s, active=True, shoulder=0, elbow=0):
        return {"time_s": time_s, "reflex_active": active, "perception_available": True,
                "shoulder_saturated": shoulder != 0, "elbow_saturated": elbow != 0,
                "requested_shoulder_velocity_deg_s": shoulder,
                "requested_elbow_velocity_deg_s": elbow,
                "applied_shoulder_velocity_deg_s": 0,
                "applied_elbow_velocity_deg_s": 0}

    def test_saturation_ratio_counts_either_joint_once_over_active_time(self):
        ticks = [self.tick(0.02, shoulder=3), self.tick(0.04),
                 self.tick(0.06, shoulder=3, elbow=4), self.tick(0.08, active=False, shoulder=10)]
        d = summarize_diagnostics(ticks)
        self.assertAlmostEqual(d.reflex_active_time_s, 0.06)
        self.assertAlmostEqual(d.saturated_reflex_time_s, 0.04)
        self.assertAlmostEqual(d.saturation_ratio, 2 / 3)
        self.assertAlmostEqual(d.shoulder_velocity_residual_rms_deg_s, math.sqrt(6))
        self.assertAlmostEqual(d.elbow_velocity_residual_rms_deg_s, math.sqrt(16 / 3))

    def test_undefined_saturation_and_partial_interval_weight(self):
        empty = summarize_diagnostics([self.tick(0.02, active=False)])
        self.assertIsNone(empty.saturation_ratio)
        self.assertIsNone(empty.shoulder_velocity_residual_rms_deg_s)
        d = summarize_diagnostics([self.tick(0.02), self.tick(0.04, shoulder=3)], duration_s=0.025)
        self.assertAlmostEqual(d.saturation_ratio, 0.2)
        self.assertAlmostEqual(d.reflex_active_time_s, 0.025)

    def test_student_requests_are_logged_before_internal_clipping(self):
        sim = SimulationCore(trajectory=make_trajectory("collision_course_center", 6))
        sim.step(stop_at_plane=False)
        tick = sim.control_diagnostics[-1]
        self.assertTrue(tick["reflex_active"])
        self.assertTrue(tick["shoulder_saturated"] or tick["elbow_saturated"])
        self.assertIsNotNone(tick["shoulder_error_deg"])
        self.assertEqual(tick["applied_shoulder_velocity_deg_s"],
                         sim.last_command["shoulder_velocity_deg_s"])
        self.assertGreater(max(abs(tick["requested_shoulder_velocity_deg_s"]),
                               abs(tick["requested_elbow_velocity_deg_s"])), 130)

    def test_legacy_static_core_exposes_trial_diagnostics(self):
        sim = SimulationCore(ball=case_to_ball(load_static_cases(DEFAULT_WORKBOOK)[2]))
        evaluate_encounter(sim)
        self.assertGreater(sim.diagnostic_summary.reflex_active_time_s, 0)
        self.assertIsNotNone(sim.diagnostic_summary.saturation_ratio)


class BenchmarkTests(unittest.TestCase):
    def test_seeded_trials_are_reproducible_and_corruption_keeps_same_paths(self):
        settings = dict(num_trials=6, speed_min=5, speed_max=6, seed=99)
        config = PerceptionConfig(0.01, 0.1, 0.02, 0.05)
        a = run_dynamic_cases(**settings, perception_config=config)
        b = run_dynamic_cases(**settings, perception_config=config)
        self.assertEqual(a, b)
        clean = run_dynamic_cases(**settings)
        self.assertEqual([(r.speed_m_s, r.perception_seed) for r in a],
                         [(r.speed_m_s, r.perception_seed) for r in clean])

    def test_cli_exports_trial_metrics_configuration_and_ticks(self):
        # Existing Windows test setup uses files rather than temporary directories
        # whose restrictive ACL can prevent the sandbox from reopening them.
        with tempfile.NamedTemporaryFile(dir=ROOT / "tests", suffix=".csv", delete=False) as f:
            path = Path(f.name)
        with tempfile.NamedTemporaryFile(dir=ROOT / "tests", suffix=".csv", delete=False) as f:
            ticks = Path(f.name)
        try:
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(main(["--num-trials", "6", "--speed-min", "6", "--speed-max", "6",
                    "--csv", str(path), "--ticks-csv", str(ticks), "--position-noise", "0.01",
                    "--latency", "0.02", "--dropout", "0.05", "--ttc-noise", "0.10"]), 0)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(r["Row_Type"] == "trial" for r in rows), 6)
            self.assertTrue(any(r["Metric"] == "latency_s" for r in rows))
            self.assertTrue(any(r["Metric"] == "false_trigger_rate" for r in rows))
            with ticks.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertTrue(records)
            self.assertIn("true_velocity_x", records[0])
            self.assertIn("perceived_ttc_s", records[0])
            self.assertIn("shoulder_saturated", records[0])
        finally:
            path.unlink(missing_ok=True)
            ticks.unlink(missing_ok=True)

    def test_invalid_cli_and_observer_settings_are_rejected(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--num-trials", "0"])
        self.assertEqual(error.exception.code, 2)
        for kwargs in ({"geometry_dt": 0.1}, {"geometry_dt": 0},
                       {"intrusion_clearance": math.nan}, {"reaction_threshold": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_dynamic_encounter(SimulationCore(trajectory=make_trajectory("near_miss")), **kwargs)

    def test_all_static_control_traces_and_metrics_match_prechange_baseline(self):
        expected = json.loads((ROOT / "benchmark_results/static_control_trace_before.json").read_text())
        with (ROOT / "benchmark_results/static_before.csv").open(encoding="utf-8-sig", newline="") as stream:
            golden = {r["Case_ID"]: r for r in csv.DictReader(stream) if r["Row_Type"] == "trial"}
        for case in load_static_cases(DEFAULT_WORKBOOK):
            sim = SimulationCore(ball=case_to_ball(case), urgency=case.urgency)
            original = sim.step
            digest = hashlib.sha256()
            def step(*args, **kwargs):
                result = original(*args, **kwargs)
                digest.update(json.dumps([sim.robot.shoulder_deg, sim.robot.elbow_deg,
                    sim.last_command, sim.last_control_state], sort_keys=True).encode())
                return result
            sim.step = step
            m = evaluate_encounter(sim)
            self.assertEqual(digest.hexdigest(), expected[case.case_id], case.case_id)
            row = golden[case.case_id]
            for key, value in asdict(m).items():
                column, factor = {"min_clearance_m": ("min_clearance_mm", 1000),
                    "baseline_min_clearance_m": ("baseline_min_clearance_mm", 1000),
                    "reaction_latency_s": ("reaction_latency_ms", 1000)}.get(key, (key, 1))
                if value is None:
                    self.assertEqual(row[column], "")
                elif isinstance(value, bool):
                    self.assertEqual(int(row[column]), int(value))
                elif isinstance(value, str):
                    self.assertEqual(row[column], value)
                else:
                    self.assertEqual(float(row[column]), value * factor)


if __name__ == "__main__":
    unittest.main()
