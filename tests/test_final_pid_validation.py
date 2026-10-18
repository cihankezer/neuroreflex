import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from motor_controller import (JointGains, PIDConfig, add_controller_arguments,
                              config_from_args, load_conventional_reference)
from run_dynamic_cases import run_dynamic_cases, write_dynamic_csv
from run_pid_study import tune
from state_activation_metrics import calculate_state_activation_metrics, summarize_state_activation_metrics


ROOT = Path(__file__).resolve().parents[1]


def tick(time, state, event=False, velocity=0):
    return {"time_s": time, "reflex_state": state, "reflex_activation_event": event,
            "applied_shoulder_velocity_deg_s": velocity}


class ActivationMetricTests(unittest.TestCase):
    def test_non_threat_state_activation_counts_even_with_zero_motor_command(self):
        metrics = calculate_state_activation_metrics([tick(.02, "REFLEX_ACTIVE", True)], False)
        self.assertTrue(metrics.state_activation_false_trigger)
        self.assertEqual(metrics.state_activation_false_trigger_events, 1)

    def test_motor_deviation_without_state_activation_is_not_false_trigger(self):
        metrics = calculate_state_activation_metrics([tick(.02, "NOMINAL", velocity=100)], False)
        self.assertFalse(metrics.state_activation_false_trigger)

    def test_threat_activation_is_true_positive(self):
        metrics = calculate_state_activation_metrics([tick(.02, "REFLEX_ACTIVE", True)], True)
        self.assertEqual(metrics.state_activation_count, 1)
        self.assertFalse(metrics.state_activation_false_trigger)
        self.assertEqual(metrics.state_activation_false_trigger_events, 0)

    def test_hold_continuation_and_reactivation_count(self):
        rows = [tick(.02, "REFLEX_ACTIVE", True), tick(.04, "PERCEPTION_HOLD"),
                tick(.06, "REFLEX_ACTIVE"), tick(.08, "NOMINAL"), tick(.10, "REFLEX_ACTIVE", True)]
        metrics = calculate_state_activation_metrics(rows, False)
        self.assertEqual(metrics.state_activation_count, 2)
        self.assertTrue(metrics.state_activation_false_trigger)
        self.assertEqual(metrics.state_activation_false_trigger_events, 2)
        self.assertEqual(metrics.state_activation_first_time_s, .02)

    def test_final_partial_interval_counts_but_later_interval_does_not(self):
        rows = [tick(.04, "REFLEX_ACTIVE", True), tick(.06, "REFLEX_ACTIVE", True)]
        metrics = calculate_state_activation_metrics(rows, False, duration_s=.03)
        self.assertEqual(metrics.state_activation_count, 1)

    def test_rate_denominator_is_non_threat_trials_not_events(self):
        def result(threat, events):
            return SimpleNamespace(metrics=SimpleNamespace(baseline_contact=threat),
                state_activation_metrics=calculate_state_activation_metrics(events, threat))
        rows = [result(False, [tick(.02, "REFLEX_ACTIVE", True), tick(.04, "REFLEX_ACTIVE", True)]),
                result(False, []), result(True, [tick(.02, "REFLEX_ACTIVE", True)])]
        summary = {r["Metric"]: r for r in summarize_state_activation_metrics(rows)}
        rate = summary["state_activation_false_trigger_rate"]
        self.assertEqual((rate["Value"], rate["Numerator"], rate["Denominator"]), (50, 1, 2))
        self.assertEqual(summary["state_activation_false_trigger_events"]["Value"], 2)
        self.assertEqual(summarize_state_activation_metrics([])[2]["Value"], "NA")

    def test_dynamic_export_adds_named_metrics_preserving_legacy_column(self):
        results = run_dynamic_cases(num_trials=6, seed=23, speed_min=6, speed_max=6)
        with tempfile.NamedTemporaryFile(dir=ROOT / "tests", suffix=".csv", delete=False) as stream:
            path = Path(stream.name)
        try:
            write_dynamic_csv(path, results, {})
            with path.open(encoding="utf-8-sig") as stream:
                records = list(csv.DictReader(stream))
            trials = [r for r in records if r["Row_Type"] == "trial"]
            for row, result in zip(trials, results):
                self.assertEqual(row["false_trigger"], str(int(result.metrics.false_trigger)))
                self.assertEqual(row["state_activation_false_trigger"],
                                 str(int(result.state_activation_metrics.state_activation_false_trigger)))
            self.assertTrue(any(r["Metric"] == "state_activation_false_trigger_rate" for r in records))
        finally:
            path.unlink(missing_ok=True)


class FrozenPresetTests(unittest.TestCase):
    def parse(self, values):
        parser = argparse.ArgumentParser()
        add_controller_arguments(parser)
        return config_from_args(parser.parse_args(values))

    def test_frozen_preset_equals_original_selected_gains_and_dt(self):
        reference = self.parse(["--controller", "pid", "--pid-preset", "conventional-reference"])
        selected = json.loads((ROOT / "benchmark_results/pid/selected.json").read_text())["config"]
        self.assertEqual(asdict(reference), selected)
        self.assertEqual(reference.shoulder, JointGains(5.25, 0, .1))
        self.assertEqual(reference.elbow, reference.shoulder)
        self.assertEqual(reference.recovery_kp, 2.5)
        self.assertEqual(reference.derivative_tau_s, .04)

    def test_preset_rejects_overrides_and_wrong_controller(self):
        for values in (["--controller", "pid", "--pid-preset", "conventional-reference", "--pid-kp", "6"],
                       ["--controller", "pid", "--pid-preset", "conventional-reference", "--pid-ki", "0"],
                       ["--pid-preset", "conventional-reference"]):
            with self.assertRaises(ValueError):
                self.parse(values)

    def test_cli_without_preset_keeps_previous_effective_defaults(self):
        self.assertEqual(self.parse([]), PIDConfig())
        self.assertEqual(self.parse(["--controller", "pid"]), PIDConfig())

    def test_historical_tuning_is_closed(self):
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            tune(ROOT / "benchmark_results/pid")


if __name__ == "__main__":
    unittest.main()
