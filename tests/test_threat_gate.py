from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from dynamic_metrics import evaluate_dynamic_encounter
from perception import PerceptionConfig, PerceptionModel, ideal_measurement
from run_dynamic_cases import main, run_dynamic_cases
from run_gate_ablation import compare_matched, summarize_ablation
from simulation_core import BALL_RADIUS, ARM_RADIUS, SimulationCore, forward_kinematics
from threat_gate import (ThreatGateConfig, evaluate_threat_gate, predict_clearances,
                         segment_segment_distance)
from trajectories import make_trajectory


ROOT = Path(__file__).resolve().parents[1]


def links_at(shoulder=35, elbow=55):
    ex, ey, hx, hy = forward_kinematics(shoulder, elbow)
    return (((0, 0, 0), (ex, ey, 0)), ((ex, ey, 0), (hx, hy, 0)))


def passive(state, dt):
    return {}


class SegmentGeometryTests(unittest.TestCase):
    def test_intersection_skew_parallel_and_degenerate_segments(self):
        queries = [(((0, 0, 0), (1, 0, 0), (.5, -1, 0), (.5, 1, 0)), 0),
                   (((0, 0, 0), (1, 0, 0), (.5, -1, 2), (.5, 1, 2)), 2),
                   (((0, 0, 0), (1, 0, 0), (2, 1, 0), (-1, 1, 0)), 1),
                   (((0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)), 1),
                   (((0, 0, 0), (0, 0, 0), (-1, 2, 0), (1, 2, 0)), 2)]
        for points, expected in queries:
            with self.subTest(points=points):
                self.assertAlmostEqual(segment_segment_distance(*points), expected)
                self.assertAlmostEqual(segment_segment_distance(*points[2:], *points[:2]), expected)

    def test_capsule_surface_clearance_subtracts_both_radii(self):
        links = (((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0)))
        clearances = predict_clearances((.5, 0, 1), (0, 0, -2), links, 1)
        self.assertAlmostEqual(clearances[0], -BALL_RADIUS - ARM_RADIUS)
        self.assertAlmostEqual(clearances[1], .5 - BALL_RADIUS - ARM_RADIUS)

    def test_prediction_includes_initial_state_and_finite_horizon(self):
        links = (((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 0, 0)))
        short = predict_clearances((.5, 0, 1), (0, 0, -1), links, .1)
        long = predict_clearances((.5, 0, 1), (0, 0, -1), links, 1)
        receding = predict_clearances((.5, 0, .2), (0, 0, 1), links, 1)
        self.assertAlmostEqual(short[0], .81)
        self.assertAlmostEqual(long[0], -.09)
        self.assertAlmostEqual(receding[0], .11)

    def test_invalid_geometry_configuration(self):
        with self.assertRaises(ValueError):
            segment_segment_distance((math.nan, 0, 0), (1, 0, 0), (0, 0, 0), (1, 1, 1))
        for kwargs in ({"mode": "unknown"}, {"safety_margin_m": -.01},
                       {"prediction_horizon_s": 0}, {"prediction_horizon_s": math.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ThreatGateConfig(**kwargs)


class GateDecisionTests(unittest.TestCase):
    def estimate(self, kind, time_s=.8, speed=1.2):
        return ideal_measurement(make_trajectory(kind, speed).state_at(time_s), time_s)

    def decide(self, p, **kwargs):
        return evaluate_threat_gate(p, links_at(), config=ThreatGateConfig("geometric", **kwargs))

    def test_center_collision_course_passes(self):
        d = self.decide(self.estimate("collision_course_center"))
        self.assertTrue(d.ttc_gate_pass)
        self.assertTrue(d.geometric_gate_pass)
        self.assertTrue(d.final_reflex_gate_pass)
        self.assertAlmostEqual(d.predicted_min_clearance_m, -.09)
        self.assertEqual(d.threat_gate_reason, "predicted_collision_risk")

    def test_left_collision_course_passes(self):
        d = self.decide(self.estimate("collision_course_left"))
        self.assertTrue(d.final_reflex_gate_pass)
        self.assertLess(d.predicted_min_clearance_link2_m, 0)

    def test_right_collision_course_passes(self):
        d = self.decide(self.estimate("collision_course_right"))
        self.assertTrue(d.final_reflex_gate_pass)
        self.assertLess(d.predicted_min_clearance_link2_m, 0)

    def test_near_miss_rejected_despite_small_ttc(self):
        d = self.decide(self.estimate("near_miss"))
        self.assertTrue(d.ttc_gate_pass)
        self.assertFalse(d.final_reflex_gate_pass)
        self.assertAlmostEqual(d.predicted_min_clearance_m, math.hypot(.15, .1) - .09)
        self.assertEqual(d.threat_gate_reason, "geometrically_safe")

    def test_fast_lateral_pass_rejected_despite_small_ttc(self):
        p = self.estimate("lateral_pass", time_s=.2, speed=6)
        d = self.decide(p)
        self.assertTrue(d.ttc_gate_pass)
        self.assertFalse(d.geometric_gate_pass)
        self.assertFalse(d.final_reflex_gate_pass)
        self.assertAlmostEqual(d.predicted_min_clearance_m, math.hypot(.18, .12) - .09)

    def test_moving_away_rejected(self):
        d = self.decide(self.estimate("moving_away"))
        self.assertFalse(d.ttc_gate_pass)
        self.assertFalse(d.final_reflex_gate_pass)
        self.assertEqual(d.threat_gate_reason, "receding")

    def test_marginal_close_pass_on_both_sides_of_safety_margin(self):
        p = self.estimate("collision_course_center")
        for offset, expected in ((.119999, True), (.120001, False)):
            altered = replace(p, x=p.x + offset, impact_x=p.impact_x + offset)
            d = self.decide(altered)
            self.assertEqual(d.final_reflex_gate_pass, expected)
            self.assertAlmostEqual(d.gate_clearance_threshold_m, .03)
        at_threshold = replace(p, x=p.x + .12, impact_x=p.impact_x + .12)
        self.assertTrue(self.decide(at_threshold).geometric_gate_pass)

    def test_current_arm_geometry_not_neutral_reference_is_used(self):
        p = self.estimate("collision_course_center")
        self.assertTrue(self.decide(p).geometric_gate_pass)
        rotated = evaluate_threat_gate(p, links_at(0, 0), config=ThreatGateConfig("geometric"))
        self.assertFalse(rotated.geometric_gate_pass)

    def test_prediction_horizon_is_configurable(self):
        p = self.estimate("collision_course_center")
        self.assertTrue(self.decide(p).final_reflex_gate_pass)
        self.assertFalse(self.decide(p, prediction_horizon_s=.1).final_reflex_gate_pass)

    def test_no_perception_releases_gate(self):
        d = self.decide(None)
        self.assertFalse(d.final_reflex_gate_pass)
        self.assertIsNone(d.predicted_min_clearance_m)
        self.assertEqual(d.threat_gate_reason, "no_perception")

    def test_older_adapter_uses_perceived_ttc_impact_ray(self):
        p = replace(self.estimate("collision_course_left"),
                    velocity_x=None, velocity_y=None, velocity_z=None)
        d = self.decide(p)
        self.assertEqual(d.motion_estimate_source, "impact_ttc_ray")
        self.assertTrue(d.final_reflex_gate_pass)

    def test_legacy_gate_ignores_safe_geometry(self):
        p = self.estimate("near_miss")
        d = evaluate_threat_gate(p, links_at(), config=ThreatGateConfig("legacy"))
        self.assertTrue(d.final_reflex_gate_pass)
        self.assertFalse(d.geometric_gate_pass)
        self.assertFalse(d.geometric_gate_enabled)

    def test_severe_perception_noise_changes_decision_without_truth_fallback(self):
        source = make_trajectory("collision_course_center").state_at(.8)
        clean = PerceptionModel().update(source, .8)
        noisy = PerceptionModel(PerceptionConfig(position_sigma_m=10), 5).update(source, .8)
        self.assertTrue(self.decide(clean).final_reflex_gate_pass)
        self.assertFalse(self.decide(noisy).final_reflex_gate_pass)
        self.assertEqual(source, make_trajectory("collision_course_center").state_at(.8))

    def test_gate_accepts_perceived_collision_estimate_for_true_near_miss(self):
        true_path = make_trajectory("near_miss", speed=6)
        estimate = self.estimate("collision_course_center", time_s=.02, speed=6)
        class FixedPerception:
            def update(self, true_state, time_s, urgency, legacy_static=False):
                return replace(estimate, observed_time_s=time_s)
        sim = SimulationCore(trajectory=true_path, perception=FixedPerception(),
                             threat_gate=ThreatGateConfig("geometric"))
        sim.step(stop_at_plane=False)
        self.assertTrue(sim.danger)
        self.assertNotEqual(sim.ball.x, sim.last_perceived_state.x)

    def test_true_collision_metrics_survive_falsely_safe_perception(self):
        config = PerceptionConfig(position_sigma_m=10)
        def run(noise):
            sim = SimulationCore(trajectory=make_trajectory("collision_course_center", 6),
                perception=PerceptionModel(noise, 5), controller=passive,
                threat_gate=ThreatGateConfig("geometric"))
            return evaluate_dynamic_encounter(sim)
        self.assertEqual(run(config), run(PerceptionConfig()))
        self.assertTrue(run(config).contact)

    def test_geometric_gate_needs_both_ttc_and_geometry(self):
        p = self.estimate("collision_course_center", time_s=0)
        d = self.decide(p, prediction_horizon_s=3)
        self.assertTrue(d.geometric_gate_pass)
        self.assertFalse(d.ttc_gate_pass)
        self.assertFalse(d.final_reflex_gate_pass)


class GateIntegrationTests(unittest.TestCase):
    def test_geometric_seed_and_complete_ticks_are_deterministic(self):
        args = dict(num_trials=6, speed_min=6, speed_max=6, seed=77,
                    perception_config=PerceptionConfig(.01, .1, .02, .05),
                    threat_gate=ThreatGateConfig("geometric"))
        self.assertEqual(run_dynamic_cases(**args), run_dynamic_cases(**args))

    def test_legacy_reproduces_every_prechange_dynamic_trial_and_tick(self):
        fixture = json.loads((ROOT / "benchmark_results/dynamic_gate_baseline.json").read_text())
        for label, count, lo, hi in (("fixed", 6, 1.2, 1.2), ("intrusion", 24, .3, 2),
                                    ("stress", 24, 2, 6)):
            for noisy in (False, True):
                config = PerceptionConfig(.01, .1, .02, .05) if noisy else PerceptionConfig()
                results = run_dynamic_cases(count, 42, config, lo, hi,
                                            threat_gate=ThreatGateConfig("legacy"))
                old = fixture["configurations"][label + ("_noisy" if noisy else "_clean")]
                for r, expected in zip(results, old):
                    with self.subTest(configuration=label, noisy=noisy, case=r.case_id):
                        self.assertEqual(asdict(r.metrics), expected["metrics"])
                        self.assertEqual(asdict(r.diagnostics), expected["diagnostics"])
                        ticks = [{k: v for k, v in t.items() if k in fixture["tick_fields"]} for t in r.ticks]
                        digest = hashlib.sha256(json.dumps(ticks, sort_keys=True,
                                                          separators=(',', ':')).encode()).hexdigest()
                        self.assertEqual(digest, expected["ticks_sha256"])

    def test_default_margin_six_cases_have_no_new_misses_or_collisions(self):
        legacy = run_dynamic_cases(6, 42, speed_min=1.2, speed_max=1.2,
                                   threat_gate=ThreatGateConfig("legacy"))
        geometric = run_dynamic_cases(6, 42, speed_min=1.2, speed_max=1.2,
                                      threat_gate=ThreatGateConfig("geometric"))
        tradeoff = compare_matched(legacy, geometric)
        self.assertEqual(tradeoff["false_triggers_prevented"], 1)
        self.assertEqual(tradeoff["new_collisions"], 0)
        self.assertEqual(tradeoff["new_missed_threats"], 0)
        self.assertEqual(tradeoff["new_late_threat_responses"], 0)
        for r in geometric[3:]:
            self.assertFalse(r.metrics.false_trigger)
            self.assertEqual(r.diagnostics.reflex_active_time_s, 0)
            self.assertIsNone(r.diagnostics.saturation_ratio)

    def test_too_small_margin_exposes_slow_approach_recovery_collision(self):
        args = dict(num_trials=1, speed_min=.3, speed_max=.3,
                    trajectory_types=("collision_course_center",), seed=42)
        small = run_dynamic_cases(**args, threat_gate=ThreatGateConfig("geometric", .01))[0]
        default = run_dynamic_cases(**args, threat_gate=ThreatGateConfig("geometric"))[0]
        self.assertTrue(small.metrics.contact)
        self.assertFalse(default.metrics.contact)
        self.assertIsNotNone(small.metrics.first_reflex_time_s)

    def test_ablation_summary_pools_actuator_time_and_rms(self):
        results = run_dynamic_cases(6, 42, speed_min=6, speed_max=6)
        summary = summarize_ablation(results)
        time = sum(r.diagnostics.reflex_active_time_s for r in results)
        self.assertEqual(summary["reflex_active_time_s"], time)
        self.assertAlmostEqual(summary["saturation_ratio"],
            sum(r.diagnostics.saturated_reflex_time_s for r in results) / time)
        shoulder = math.sqrt(sum(r.diagnostics.reflex_active_time_s *
            (r.diagnostics.shoulder_velocity_residual_rms_deg_s or 0)**2 for r in results) / time)
        self.assertAlmostEqual(summary["shoulder_velocity_residual_rms_deg_s"], shoulder)

    def test_invalid_cli_margin_is_rejected_before_running(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--threat-gate", "geometric", "--gate-safety-margin", "-1"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
