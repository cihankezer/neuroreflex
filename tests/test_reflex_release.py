from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import unittest

from dynamic_metrics import evaluate_dynamic_encounter
from perception import PerceptionConfig, PerceptionModel, ideal_measurement
from reflex_release import ReflexReleaseConfig, ReflexStateMachine
from release_metrics import calculate_release_metrics
from run_dynamic_cases import run_dynamic_cases
from simulation_core import SimulationCore, forward_kinematics
from threat_gate import GateDecision, ThreatGateConfig, evaluate_threat_gate
from trajectories import make_trajectory


ROOT = Path(__file__).resolve().parents[1]
ex, ey, hx, hy = forward_kinematics(35, 55)
LINKS = (((0, 0, 0), (ex, ey, 0)), ((ex, ey, 0), (hx, hy, 0)))


def estimate(z=.2, velocity_z=-1):
    source = make_trajectory("collision_course_center").state_at(0)
    return ideal_measurement(replace(source, z=z, velocity_z=velocity_z), 0)


def gate(clearance, ttc=.5, available=True):
    ttc_pass = available and ttc < 1
    geo = clearance is not None and clearance <= .03
    return GateDecision(True, clearance, clearance, clearance, .03, 1, ttc_pass,
                        geo, ttc_pass and geo, "test", "perceived_velocity")


class StateMachineTests(unittest.TestCase):
    def machine(self, dwell=0, margin=.06, motion="none"):
        return ReflexStateMachine(ThreatGateConfig("geometric_hysteresis"),
                                  ReflexReleaseConfig(margin, dwell, motion))

    def step(self, machine, clearance, ttc=.5, p=None, missing=False):
        time_s = (machine.last_time_s or 0) + .02
        observed = None if missing else p if p is not None else estimate()
        return machine.update(gate(clearance, ttc, not missing), observed, LINKS, time_s)

    def test_direct_collision_activates(self):
        machine = self.machine()
        d = self.step(machine, -.09)
        self.assertEqual(d.reflex_state, "REFLEX_ACTIVE")
        self.assertTrue(d.reflex_activation_event)
        self.assertFalse(d.recovery_active)

    def test_near_miss_never_activates(self):
        path = make_trajectory("near_miss", 6)
        sim = SimulationCore(trajectory=path, threat_gate=ThreatGateConfig("geometric_hysteresis"))
        evaluate_dynamic_encounter(sim)
        self.assertEqual(sim.release_metrics.activation_count, 0)
        self.assertFalse(any(t["final_reflex_active"] for t in sim.control_diagnostics))

    def test_hysteresis_holds_just_above_enter_threshold(self):
        machine = self.machine()
        self.step(machine, .02)
        d = self.step(machine, .031)
        self.assertTrue(d.final_reflex_active)
        self.assertFalse(d.activation_gate_raw)
        self.assertFalse(d.release_condition_clearance)
        self.assertEqual(d.state_transition_reason, "hysteresis_hold")

    def test_release_threshold_must_be_strictly_exceeded(self):
        machine = self.machine()
        self.step(machine, .02)
        self.assertTrue(self.step(machine, .06).final_reflex_active)
        d = self.step(machine, .060001)
        self.assertTrue(d.reflex_release_event)
        self.assertTrue(d.recovery_active)
        self.assertTrue(d.recovery_started_event)

    def test_dwell_requires_consecutive_safe_ticks(self):
        machine = self.machine(dwell=.08)
        self.step(machine, .02)
        for i in range(3):
            d = self.step(machine, .10)
            self.assertTrue(d.final_reflex_active)
            self.assertAlmostEqual(d.safe_release_timer_s, (i + 1) * .02)
        self.assertTrue(self.step(machine, .10).reflex_release_event)

    def test_dwell_resets_when_clearance_drops(self):
        machine = self.machine(dwell=.04)
        self.step(machine, .02)
        self.step(machine, .1)
        d = self.step(machine, .05)
        self.assertEqual(d.safe_release_timer_s, 0)
        self.assertTrue(self.step(machine, .1).final_reflex_active)
        self.assertTrue(self.step(machine, .1).reflex_release_event)

    def test_missing_observation_does_not_release(self):
        machine = self.machine(dwell=.04)
        self.step(machine, .02)
        self.step(machine, .1)
        d = self.step(machine, None, missing=True)
        self.assertEqual(d.reflex_state, "PERCEPTION_HOLD")
        self.assertTrue(d.final_reflex_active)
        self.assertFalse(d.recovery_active)
        self.assertEqual(d.safe_release_timer_s, 0)

    def test_held_estimates_cannot_complete_dwell(self):
        machine = self.machine(dwell=.04)
        self.step(machine, .02)
        self.step(machine, .1)
        d = self.step(machine, .1, p=replace(estimate(), held=True))
        self.assertTrue(d.final_reflex_active)
        self.assertEqual(d.safe_release_timer_s, 0)

    def test_infinite_ttc_does_not_release_unsafe_geometry(self):
        machine = self.machine()
        self.step(machine, .02)
        d = self.step(machine, .04, ttc=math.inf, p=estimate(z=-.02))
        self.assertTrue(d.final_reflex_active)
        self.assertFalse(d.activation_gate_raw)
        self.assertFalse(d.recovery_active)

    def test_motion_requires_perceived_receding_obstacle(self):
        machine = self.machine(motion="receding")
        self.step(machine, .02)
        approaching = self.step(machine, .1, p=estimate(z=.2))
        self.assertFalse(approaching.release_condition_motion)
        self.assertTrue(approaching.final_reflex_active)
        receding = self.step(machine, .1, ttc=math.inf, p=estimate(z=-.2))
        self.assertTrue(receding.release_condition_motion)
        self.assertTrue(receding.reflex_release_event)

    def test_reentry_after_release_reactivates(self):
        machine = self.machine()
        self.step(machine, .02)
        self.step(machine, .1)
        d = self.step(machine, .02)
        self.assertTrue(d.reflex_activation_event)
        self.assertEqual(d.reflex_state, "REFLEX_ACTIVE")
        self.assertFalse(d.recovery_active)

    def test_neutral_recovery_only_after_release(self):
        machine = self.machine(dwell=.04)
        self.step(machine, .02)
        held = self.step(machine, .1)
        self.assertFalse(held.recovery_active)
        released = self.step(machine, .1)
        self.assertTrue(released.recovery_active)
        self.assertEqual(released.state_transition_reason, "release_confirmed")

    def test_invalid_release_configuration(self):
        for kwargs in ({"release_margin_m": -.1}, {"dwell_s": -.1},
                       {"dwell_s": math.inf}, {"motion_condition": "oracle"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReflexReleaseConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.machine(margin=.03)


class ReleaseIntegrationTests(unittest.TestCase):
    def test_geometric_alias_preserves_memoryless_behavior(self):
        kwargs = dict(num_trials=6, speed_min=6, speed_max=6, seed=42)
        self.assertEqual(run_dynamic_cases(**kwargs, threat_gate=ThreatGateConfig("geometric")),
                         run_dynamic_cases(**kwargs, threat_gate=ThreatGateConfig("geometric_memoryless")))

    def test_all_prechange_legacy_and_memoryless_traces_match(self):
        baseline = json.loads((ROOT / "benchmark_results/release_baseline.json").read_text())
        for label, count, lo, hi in (("fixed", 6, 1.2, 1.2), ("intrusion", 24, .3, 2),
                                    ("stress", 24, 2, 6)):
            for noisy in (False, True):
                for mode in ("legacy", "geometric"):
                    config = PerceptionConfig(.01, .1, .02, .05) if noisy else PerceptionConfig()
                    results = run_dynamic_cases(count, 42, config, lo, hi, threat_gate=ThreatGateConfig(mode))
                    expected = baseline[f'{label}_{"noisy" if noisy else "clean"}_{mode}']
                    for r, old in zip(results, expected):
                        with self.subTest(configuration=label, noisy=noisy, mode=mode, case=r.case_id):
                            self.assertEqual(asdict(r.metrics), old["metrics"])
                            self.assertEqual(asdict(r.diagnostics), old["diagnostics"])
                            ticks = [{k: v for k, v in t.items() if k in old["tick_fields"]} for t in r.ticks]
                            digest = hashlib.sha256(json.dumps(ticks, sort_keys=True,
                                                              separators=(',', ':')).encode()).hexdigest()
                            self.assertEqual(digest, old["ticks_sha256"])

    def test_first_activation_timing_unchanged_clean_and_noisy(self):
        for noisy in (False, True):
            config = PerceptionConfig(.01, .1, .02, .05) if noisy else PerceptionConfig()
            common = dict(num_trials=6, seed=42, perception_config=config, speed_min=6, speed_max=6)
            old = run_dynamic_cases(**common, threat_gate=ThreatGateConfig("geometric_memoryless"))
            new = run_dynamic_cases(**common, threat_gate=ThreatGateConfig("geometric_hysteresis"))
            for a, b in zip(old, new):
                self.assertEqual(a.release_metrics.first_activation_time_s,
                                 b.release_metrics.first_activation_time_s)

    def test_fixed_seed_stateful_runs_are_deterministic(self):
        kwargs = dict(num_trials=6, seed=7, perception_config=PerceptionConfig(.01, .1, .02, .05),
            speed_min=6, speed_max=6, threat_gate=ThreatGateConfig("geometric_hysteresis"),
            post_encounter_time_s=.5)
        self.assertEqual(run_dynamic_cases(**kwargs), run_dynamic_cases(**kwargs))

    def test_expired_perception_freezes_pose_until_real_estimate_returns(self):
        class MissingAfterFirst:
            def update(self, true_state, time_s, urgency, legacy_static=False):
                return ideal_measurement(true_state, time_s) if time_s <= .02 else None
        sim = SimulationCore(trajectory=make_trajectory("collision_course_center", 6),
                             perception=MissingAfterFirst(), threat_gate=ThreatGateConfig("geometric_hysteresis"))
        sim.step(stop_at_plane=False)
        pose = replace(sim.robot)
        for _ in range(10):
            sim.step(stop_at_plane=False)
            self.assertEqual(sim.robot, pose)
            self.assertTrue(sim.danger)
            self.assertFalse(sim.last_release_decision.recovery_active)
            self.assertEqual(sim.last_command["shoulder_velocity_deg_s"], 0)
            self.assertIsNone(sim.last_control_state["impact_x"])

    def test_true_geometry_independent_of_perceived_gate_with_passive_controller(self):
        def run(noise):
            sim = SimulationCore(trajectory=make_trajectory("collision_course_center", 6),
                perception=PerceptionModel(noise, 5), controller=lambda state, dt: {},
                threat_gate=ThreatGateConfig("geometric_hysteresis"))
            return evaluate_dynamic_encounter(sim)
        self.assertEqual(run(PerceptionConfig()), run(PerceptionConfig(position_sigma_m=100)))
        self.assertTrue(run(PerceptionConfig(position_sigma_m=100)).contact)

    def test_policy_has_no_trajectory_name_input(self):
        a = SimulationCore(trajectory=make_trajectory("collision_course_center", 6),
                           threat_gate=ThreatGateConfig("geometric_hysteresis"))
        b = SimulationCore(trajectory=replace(make_trajectory("collision_course_center", 6), name="near_miss"),
                           threat_gate=ThreatGateConfig("geometric_hysteresis"))
        self.assertEqual(evaluate_dynamic_encounter(a), evaluate_dynamic_encounter(b))
        self.assertEqual(a.control_diagnostics, b.control_diagnostics)

    def test_release_is_identical_for_same_perception_different_true_paths(self):
        class IdenticalEstimates:
            def update(self, true_state, time_s, urgency, legacy_static=False):
                # A deliberately wrong but identical sensor stream for both paths.
                return estimate(z=.02 if time_s <= .04 else -.3)
        decisions = []
        outcomes = []
        for kind in ("collision_course_center", "near_miss"):
            sim = SimulationCore(trajectory=make_trajectory(kind, 6),
                perception=IdenticalEstimates(), controller=lambda state, dt: {},
                threat_gate=ThreatGateConfig("geometric_hysteresis"))
            outcomes.append(evaluate_dynamic_encounter(sim))
            decisions.append([(t["reflex_state"], t["safe_release_timer_s"],
                               t["reflex_release_event"], t["recovery_active"])
                              for t in sim.control_diagnostics])
        self.assertEqual(decisions[0], decisions[1])
        self.assertTrue(decisions[0][3][2])
        self.assertTrue(outcomes[0].contact)
        self.assertFalse(outcomes[1].contact)

    def test_slow_reactivation_remains_counted_without_chatter_label(self):
        ticks = [dict(time_s=time, final_reflex_active=active,
                      reflex_activation_event=activation, reflex_release_event=release)
                 for time, active, activation, release in
                 ((.02, True, True, False), (.04, False, False, True),
                  (.24, True, True, False), (.26, False, False, True))]
        m = calculate_release_metrics(ticks, [], .26)
        self.assertEqual(m.reactivation_count, 1)
        self.assertEqual(m.number_of_gate_chatter_events, 0)

    def test_known_small_margin_failure_prevented_without_gain_changes(self):
        kwargs = dict(num_trials=1, seed=42, speed_min=.34511114846, speed_max=.34511114846,
                      trajectory_types=("collision_course_center",), post_encounter_time_s=.5)
        old = run_dynamic_cases(**kwargs, threat_gate=ThreatGateConfig("geometric_memoryless", .01))[0]
        new = run_dynamic_cases(**kwargs, threat_gate=ThreatGateConfig("geometric_hysteresis", .01),
                                reflex_release=ReflexReleaseConfig(.06, .04, "receding"))[0]
        self.assertTrue(old.metrics.contact)
        self.assertFalse(new.metrics.contact)
        self.assertEqual(old.release_metrics.first_activation_time_s,
                         new.release_metrics.first_activation_time_s)
        self.assertEqual(new.release_metrics.reactivation_count, 0)
        self.assertFalse(new.release_metrics.active_at_encounter_end)

    def test_additive_release_metrics_and_chatter_proxy(self):
        def tick(time, active, activation=False, release=False):
            return {"time_s": time, "final_reflex_active": active,
                    "reflex_activation_event": activation, "reflex_release_event": release}
        ticks = [tick(.02, True, activation=True), tick(.04, False, release=True),
                 tick(.06, True, activation=True), tick(.08, False, release=True)]
        m = calculate_release_metrics(ticks, [(.08, .1, True), (.1, .05, True)], .1)
        self.assertEqual((m.activation_count, m.release_count, m.reactivation_count), (2, 2, 1))
        self.assertEqual(m.number_of_gate_chatter_events, 1)
        self.assertAlmostEqual(m.total_reflex_active_time_s, .04)
        self.assertEqual(m.minimum_clearance_after_final_release_m, .05)
        self.assertAlmostEqual(m.time_from_final_release_to_closest_approach_s, .02)
        self.assertEqual(m.minimum_clearance_during_recovery_m, .05)

    def test_unreleased_episode_is_censored_not_declared_safe(self):
        ticks = [{"time_s": .02, "final_reflex_active": True,
                  "reflex_activation_event": True, "reflex_release_event": False}]
        m = calculate_release_metrics(ticks, [(.02, .1, False)], .02)
        self.assertTrue(m.active_at_encounter_end)
        self.assertIsNone(m.final_release_time_s)
        self.assertIsNone(m.minimum_clearance_after_final_release_m)


if __name__ == "__main__":
    unittest.main()
