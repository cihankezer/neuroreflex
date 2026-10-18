import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from run_static_cases import (
    DEFAULT_WORKBOOK,
    StaticCase,
    case_to_ball,
    load_static_cases,
    main,
    run_cases,
)
from simulation_core import (
    BallState,
    CONTROL_DT,
    ELBOW_MAX,
    NEUTRAL_ELBOW,
    NEUTRAL_SHOULDER,
    SHOULDER_MAX,
    SimulationCore,
    calculate_perception,
    collision_test,
    forward_kinematics,
)


ROOT = Path(__file__).resolve().parents[1]


def stationary_controller(state, dt):
    return {}


class SimulationTests(unittest.TestCase):
    def test_normalized_mapping_and_initial_ttc(self):
        case = StaticCase("example", -0.78, 0.05, 0.88, 0.38)
        ball = case_to_ball(case)
        _, _, hand_x, hand_y = forward_kinematics(35.0, 55.0)
        self.assertAlmostEqual(ball.impact_x, hand_x - 0.20 * 0.78)
        self.assertAlmostEqual(ball.impact_y, hand_y + 0.20 * 0.05)
        self.assertAlmostEqual(ball.z, 0.38 * 1.2)
        self.assertAlmostEqual(calculate_perception(ball)[1], 0.38)
        faster_ball = case_to_ball(case, ball_speed=2.4, direction_scale=0.1)
        self.assertAlmostEqual(faster_ball.z, ball.z * 2.0)
        self.assertAlmostEqual(calculate_perception(faster_ball)[1], 0.38)

    def test_actuator_limits_bound_even_a_saturated_controller(self):
        for urgency in (0.0, 0.25, 1.0):
            for direction in (-1, 1):
                with self.subTest(urgency=urgency, direction=direction):
                    seen = []

                    def saturated(state, dt):
                        seen.append((state, dt))
                        return {
                            "shoulder_velocity_deg_s": direction * 10000,
                            "elbow_velocity_deg_s": direction * 10000,
                        }

                    simulation = SimulationCore(urgency=urgency, controller=saturated)
                    simulation.step()
                    self.assertEqual(seen[0][0]["urgency"], urgency)
                    self.assertEqual(seen[0][1], CONTROL_DT)
                    self.assertAlmostEqual(
                        simulation.last_command["shoulder_velocity_deg_s"],
                        direction * urgency * SHOULDER_MAX,
                    )
                    self.assertAlmostEqual(
                        simulation.last_command["elbow_velocity_deg_s"],
                        direction * urgency * ELBOW_MAX,
                    )
                    self.assertAlmostEqual(
                        simulation.robot.shoulder_deg,
                        NEUTRAL_SHOULDER + direction * urgency * SHOULDER_MAX * CONTROL_DT,
                    )
                    self.assertAlmostEqual(
                        simulation.robot.elbow_deg,
                        NEUTRAL_ELBOW + direction * urgency * ELBOW_MAX * CONTROL_DT,
                    )

    def test_collision_checks_both_links(self):
        simulation = SimulationCore(controller=stationary_controller)
        ex, ey, hx, hy = forward_kinematics(35.0, 55.0)
        for x, y in ((ex / 2, ey / 2), ((ex + hx) / 2, (ey + hy) / 2)):
            with self.subTest(point=(x, y)):
                simulation.ball.impact_x, simulation.ball.impact_y = x, y
                self.assertTrue(collision_test(simulation.robot, simulation.ball))
        simulation.ball.impact_x, simulation.ball.impact_y = 1.2, 1.2
        self.assertFalse(collision_test(simulation.robot, simulation.ball))
        simulation.ball.impact_x = 0.0
        simulation.ball.impact_y = -0.089
        self.assertTrue(collision_test(simulation.robot, simulation.ball))
        simulation.ball.impact_y = -0.091
        self.assertFalse(collision_test(simulation.robot, simulation.ball))

    def test_trial_stops_and_terminal_steps_do_not_move_again(self):
        ball = case_to_ball(StaticCase("center", 0, 0, 0, 0.10))
        simulation = SimulationCore(ball=ball, controller=stationary_controller)
        self.assertEqual(simulation.run_trial(), "COLLISION")
        terminal_z = ball.z
        self.assertEqual(simulation.step(), "COLLISION")
        self.assertEqual(ball.z, terminal_z)
        self.assertEqual(simulation.robot.shoulder_deg, NEUTRAL_SHOULDER)

    def test_batch_resets_pose_for_each_case_and_is_order_independent(self):
        cases = load_static_cases(DEFAULT_WORKBOOK)
        initial_poses = []
        real_init = SimulationCore.__init__

        def record_init(simulation, *args, **kwargs):
            real_init(simulation, *args, **kwargs)
            initial_poses.append(
                (simulation.robot.shoulder_deg, simulation.robot.elbow_deg)
            )

        with patch.object(SimulationCore, "__init__", record_init):
            results = run_cases(cases)
        self.assertEqual(initial_poses, [(35.0, 55.0)] * len(cases))
        reversed_results = run_cases(reversed(cases))
        self.assertEqual(results, list(reversed(reversed_results)))

    def test_nonfinite_or_invalid_inputs_are_rejected(self):
        for urgency in (-0.1, 1.1, float("nan")):
            with self.subTest(urgency=urgency), self.assertRaises(ValueError):
                SimulationCore(urgency=urgency)
        for speed in (0, -1, float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                SimulationCore(ball=BallState(speed=speed))
        with self.assertRaises(ValueError):
            StaticCase("bad", 1.1, 0, 0.5, 1)


class WorkbookAndCliTests(unittest.TestCase):
    def test_project_workbook_loads_all_sixteen_cases(self):
        cases = load_static_cases(DEFAULT_WORKBOOK)
        self.assertEqual([c.case_id for c in cases], [f"S{i:02}" for i in range(1, 17)])
        self.assertEqual(cases[2], StaticCase("S03", 0, 0, 0.92, 0.32))
        self.assertEqual(cases[-1], StaticCase("S16", 0.25, -0.15, 0.6, 0.75))

    def test_reader_supports_reordered_headers_gaps_and_inline_strings(self):
        # XLSX encodings differ between Excel and other exporters.
        path = io.BytesIO()
        with ZipFile(path, "w") as workbook:
                workbook.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Static Cases" r:id="r1"/></sheets></workbook>')
                workbook.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Target="worksheets/cases.xml"/></Relationships>')
                workbook.writestr("xl/sharedStrings.xml", '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Case_ID</t></si></sst>')
                workbook.writestr("xl/worksheets/cases.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>urgency</t></is></c><c r="C1" t="s"><v>0</v></c><c r="D1" t="inlineStr"><is><t>threat_y</t></is></c><c r="F1" t="inlineStr"><is><t>ttc_s</t></is></c><c r="G1" t="inlineStr"><is><t>threat_x</t></is></c></row><row r="2"><c r="A2"><v>0.5</v></c><c r="C2" t="inlineStr"><is><t>TEST</t></is></c><c r="D2"><v>-0.2</v></c><c r="F2"><v>0.4</v></c><c r="G2"><v>0.3</v></c></row></sheetData></worksheet>')
        path.seek(0)
        self.assertEqual(load_static_cases(path), [StaticCase("TEST", 0.3, -0.2, 0.5, 0.4)])

    def test_summary_and_optional_csv_agree_with_trial_results(self):
        expected = run_cases(load_static_cases(DEFAULT_WORKBOOK))
        successes = sum(r.result == "SAFE" for r in expected)
        with tempfile.NamedTemporaryFile(dir=ROOT / "tests", suffix=".csv", delete=False) as temporary:
            csv_path = Path(temporary.name)
        try:
            output = io.StringIO()
            with patch("sys.stdout", output):
                self.assertEqual(main(["--csv", str(csv_path)]), 0)
            self.assertEqual(output.getvalue().splitlines()[:4], [
                "Total cases: 16",
                f"Successes: {successes}",
                f"Collisions: {16 - successes}",
                f"Success rate: {successes / 16:.2%}",
            ])
            with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            trials = [r for r in rows if r["Row_Type"] == "trial"]
            self.assertEqual([(r["Case_ID"], r["Result"]) for r in trials],
                             [(r.case_id, r.result) for r in expected])
            summary = {r["Metric"]: r for r in rows if r["Row_Type"] == "summary"}
            self.assertEqual(int(summary["successes"]["Value"]), successes)
            self.assertEqual(int(summary["collision_rate"]["Denominator"]), 16)
            self.assertAlmostEqual(float(summary["collision_rate"]["Value"]),
                                   100.0 * (16 - successes) / 16)
            for row, result in zip(trials, expected):
                self.assertAlmostEqual(float(row["min_clearance_mm"]),
                                       result.metrics.min_clearance_m * 1000.0)
            self.assertTrue(any(r["Row_Type"] == "configuration" for r in rows))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_batch_does_not_import_gui_dependencies(self):
        program = (
            "import sys; import run_static_cases; "
            "run_static_cases.run_cases(run_static_cases.load_static_cases(run_static_cases.DEFAULT_WORKBOOK)); "
            "assert not any(n.startswith(('PySide6', 'matplotlib')) for n in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program], cwd=ROOT,
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
