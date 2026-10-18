from trace_fingerprints import assert_reference_trace
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import unittest

from dynamic_metrics import evaluate_dynamic_encounter
from perception import PerceptionConfig, PerceptionModel, ideal_measurement
from reflex_release import ReflexReleaseConfig, ReflexStateMachine
from run_release_ablation import run_policy_set
from simulation_core import SimulationCore, forward_kinematics
from threat_gate import ThreatGateConfig, evaluate_threat_gate
from trajectories import TrueBallState, make_trajectory


ROOT = Path(__file__).resolve().parents[1]


class QuickRobustnessTests(unittest.TestCase):
    def state(self):
        return TrueBallState(.3, .6, .2, .1, -.2, -.4)

    def test_velocity_noise_is_seeded_and_preserves_other_corruption_samples(self):
        original = PerceptionConfig(.01, .1, .02, .05)
        noisy = replace(original, velocity_sigma_m_s=.1)
        def sample(config, seed):
            model = PerceptionModel(config, seed)
            return [model.update(self.state(), (i+1)*.02) for i in range(30)]
        actual = sample(noisy, 42)
        self.assertEqual(actual, sample(noisy, 42))
        self.assertNotEqual(actual, sample(noisy, 7))
        for clean, changed in zip(sample(original, 42), actual):
            if clean is None:
                self.assertIsNone(changed)
            else:
                self.assertEqual(clean, replace(changed, velocity_x=clean.velocity_x,
                    velocity_y=clean.velocity_y, velocity_z=clean.velocity_z))

    def test_velocity_scale_and_off_defaults(self):
        state = self.state()
        self.assertEqual(PerceptionModel().update(state, .02), ideal_measurement(state, .02))
        scaled = PerceptionModel(PerceptionConfig(velocity_scale=.9)).update(state, .02)
        for axis in 'xyz':
            self.assertEqual(getattr(scaled,'velocity_'+axis), .9*getattr(state,'velocity_'+axis))

    def test_vector_sign_flip_is_seeded_and_applied_only_on_acquisition(self):
        def sample(seed):
            model = PerceptionModel(PerceptionConfig(velocity_sign_flip_probability=.5), seed)
            return [model.update(self.state(), i*.02) for i in range(20)]
        self.assertEqual(sample(42), sample(42))
        self.assertNotEqual(sample(42), sample(7))
        self.assertTrue(any(p.velocity_z > 0 for p in sample(42)))
        model = PerceptionModel(PerceptionConfig(velocity_sign_flip_probability=1, burst_start_s=.04,
                                               burst_duration_s=.1))
        acquired = model.update(self.state(), .02)
        self.assertEqual(acquired.velocity_z, .4)
        held = model.update(self.state(), .04)
        self.assertEqual(held, replace(acquired, held=True))

    def test_corrupted_velocity_does_not_modify_true_geometry(self):
        def observe(config):
            sim = SimulationCore(trajectory=make_trajectory('collision_course_center',1.2),
                perception=PerceptionModel(config,42), controller=lambda state,dt:{},
                threat_gate=ThreatGateConfig('geometric_hysteresis'))
            return evaluate_dynamic_encounter(sim)
        self.assertEqual(observe(PerceptionConfig()), observe(PerceptionConfig(
            velocity_sigma_m_s=10, velocity_sign_flip_probability=1)))
        self.assertTrue(observe(PerceptionConfig(velocity_sign_flip_probability=1)).contact)

    def test_release_uses_corrupted_velocity_without_true_motion_fallback(self):
        _, _, hx, hy = forward_kinematics(35,55)
        # Far in front of the arm: the true ray intersects it, but a flipped
        # perceived vector predicts departure and permits the unchanged release.
        true = TrueBallState(hx,hy,.5,0,0,-1)
        ex,ey,_,_ = forward_kinematics(35,55)
        links = (((0,0,0),(ex,ey,0)),((ex,ey,0),(hx,hy,0)))
        model = PerceptionModel(PerceptionConfig(velocity_sign_flip_probability=1),42)
        machine = ReflexStateMachine(ThreatGateConfig('geometric_hysteresis'),ReflexReleaseConfig())
        correct = ideal_measurement(true,.02)
        machine.update(evaluate_threat_gate(correct,links,config=machine.gate_config),correct,links,.02)
        corrupt = model.update(true,.04)
        first = machine.update(evaluate_threat_gate(corrupt,links,config=machine.gate_config),corrupt,links,.04)
        self.assertTrue(first.release_condition_motion)
        self.assertTrue(first.final_reflex_active)
        corrupt = model.update(true,.06)
        released = machine.update(evaluate_threat_gate(corrupt,links,config=machine.gate_config),corrupt,links,.06)
        self.assertTrue(released.reflex_release_event)
        self.assertLess(true.velocity_z,0)

    def test_accelerating_approach_has_exact_state_and_increasing_speed(self):
        path = make_trajectory('accelerating_approach',1.2)
        at = path.state_at(.5)
        self.assertLess(at.velocity_z,path.initial.velocity_z)
        self.assertAlmostEqual(at.z,2-1.2*.5+.5*path.acceleration_z*.5**2)
        self.assertAlmostEqual(path.state_at(path.duration_s).z,-.2)

    def test_decelerating_approach_remains_approaching_through_the_test_tail(self):
        for speed in (.35,1.2):
            path = make_trajectory('decelerating_approach',speed)
            at = path.state_at(path.duration_s+.5)
            self.assertGreater(at.velocity_z,path.initial.velocity_z)
            self.assertLess(at.velocity_z,0)
            self.assertAlmostEqual(path.state_at(path.duration_s).z,-.2)

    def test_tangential_near_miss_is_geometrically_non_threatening(self):
        sim = SimulationCore(trajectory=make_trajectory('tangential_near_miss',1.2),
                             controller=lambda state,dt:{})
        result = evaluate_dynamic_encounter(sim)
        self.assertFalse(result.baseline_contact)
        self.assertFalse(result.contact)
        self.assertAlmostEqual(result.baseline_min_clearance_m,(.13**2+.04**2)**.5-.09)

    def test_burst_uses_existing_buffer_hold_then_pose_hold_and_resumes(self):
        sim = SimulationCore(trajectory=make_trajectory('collision_course_center',6),
            perception=PerceptionModel(PerceptionConfig(latency_s=.02,burst_start_s=.10,burst_duration_s=.20),42),
            threat_gate=ThreatGateConfig('geometric_hysteresis'))
        rows=[]
        for _ in range(20):
            sim.step(stop_at_plane=False); rows.append(sim.control_diagnostics[-1])
        self.assertTrue(any(t.get('perceived_held') for t in rows))
        held=[t for t in rows if t['reflex_state']=='PERCEPTION_HOLD']
        self.assertTrue(held)
        self.assertTrue(all(t['final_reflex_active'] and not t['recovery_active'] for t in held))
        self.assertTrue(all(t['applied_shoulder_velocity_deg_s']==t['applied_elbow_velocity_deg_s']==0 for t in held))
        self.assertTrue(any(t['perception_available'] and not t.get('perceived_held') for t in rows if t['time_s']>.32))

    def test_selected_stateful_baseline_fingerprints_remain_exact(self):
        before=json.loads((ROOT/'benchmark_results/robustness_check/stateful_before.json').read_text())
        for (benchmark,noise),results in run_policy_set(42,'geometric_hysteresis',ReflexReleaseConfig(),.5).items():
            for result,expected in zip(results,before[benchmark+'_'+noise]):
                self.assertEqual(asdict(result.metrics),expected['metrics'])
                self.assertEqual(asdict(result.diagnostics),expected['diagnostics'])
                self.assertEqual(asdict(result.release_metrics),expected['release_metrics'])
                assert_reference_trace(self, result.ticks, expected["ticks_sha256"])

    def test_invalid_new_corruption_parameters_are_rejected(self):
        for kwargs in ({'velocity_sigma_m_s':-.1},{'velocity_scale':0},
                       {'velocity_sign_flip_probability':1.01},{'burst_duration_s':-.1}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                PerceptionConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
