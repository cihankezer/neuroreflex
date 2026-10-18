"""Run the workbook's Static Cases without Qt, plotting, or real-time waits."""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

import numpy as np

from performance_metrics import (
    CONTACT_TOLERANCE,
    EncounterMetrics,
    GEOMETRY_DT,
    INTRUSION_CLEARANCE,
    REACTION_THRESHOLD,
    evaluate_encounter,
)
from simulation_core import (
    ARM_RADIUS,
    BALL_RADIUS,
    BallState,
    CONTROL_DT,
    DEFAULT_BALL_SPEED,
    ELBOW_MAX,
    L1,
    L2,
    NEUTRAL_ELBOW,
    NEUTRAL_SHOULDER,
    SHOULDER_MAX,
    SimulationCore,
    forward_kinematics,
)


from importlib.resources import files
DEFAULT_WORKBOOK = Path(str(files("neuroreflex").joinpath("configs/static_cases.xlsx")))
DIRECTION_SCALE = 0.20  # metres per normalized direction unit
NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass(frozen=True)
class StaticCase:
    case_id: str
    threat_x: float
    threat_y: float
    urgency: float
    ttc_s: float

    def __post_init__(self):
        values = (self.threat_x, self.threat_y, self.urgency, self.ttc_s)
        if not self.case_id.strip() or not all(math.isfinite(v) for v in values):
            raise ValueError("Case ID must be nonempty and inputs must be finite.")
        if not (-1.0 <= self.threat_x <= 1.0 and -1.0 <= self.threat_y <= 1.0):
            raise ValueError("Threat directions must be between -1 and 1.")
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError("Urgency must be between 0 and 1.")
        if self.ttc_s <= 0.0:
            raise ValueError("TTC must be positive.")


@dataclass(frozen=True)
class TrialResult:
    case_id: str
    result: str
    metrics: EncounterMetrics
    case: StaticCase


def _read_static_rows(path):
    """Read values from OOXML using only the standard library.

    Supports shared and inline strings, column gaps, and cached formula values.
    It never launches Excel or modifies the workbook.
    """
    with ZipFile(path) as workbook:
        strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(t.text or "" for t in node.findall(".//s:t", NS))
                for node in root.findall("s:si", NS)
            ]
        root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet = next(
            (s for s in root.findall("s:sheets/s:sheet", NS)
             if s.get("name") == "Static Cases"),
            None,
        )
        if sheet is None:
            raise ValueError("Workbook has no 'Static Cases' sheet.")
        relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationship_id = sheet.get("{" + NS["r"] + "}id")
        target = next(
            r.get("Target") for r in relationships
            if r.get("Id") == relationship_id
        )
        sheet_path = (
            target.lstrip("/") if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        root = ET.fromstring(workbook.read(sheet_path))
        rows = []
        for row in root.findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in row.findall("s:c", NS):
                reference = cell.get("r", "")
                column = "".join(c for c in reference if c.isalpha())
                value = cell.findtext("s:v", default="", namespaces=NS)
                if cell.get("t") == "s":
                    value = strings[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//s:t", NS)
                    )
                cells[column] = value.strip()
            if any(cells.values()):
                rows.append((row.get("r", "?"), cells))
        return rows


def load_static_cases(path):
    rows = _read_static_rows(path)
    if not rows:
        raise ValueError("'Static Cases' sheet is empty.")
    columns = {header: column for column, header in rows[0][1].items()}
    required = ("Case_ID", "threat_x", "threat_y", "urgency", "ttc_s")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    cases = []
    seen = set()
    for row_number, cells in rows[1:]:
        try:
            case_id = cells.get(columns["Case_ID"], "")
            values = [float(cells.get(columns[name], "")) for name in required[1:]]
            case = StaticCase(case_id, *values)
            if case_id in seen:
                raise ValueError("Duplicate case ID: " + case_id)
        except ValueError as error:
            raise ValueError(f"Static Cases row {row_number}: {error}") from error
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("'Static Cases' sheet contains no cases.")
    return cases


def case_to_ball(case, ball_speed=DEFAULT_BALL_SPEED, direction_scale=DIRECTION_SCALE):
    for label, value in (("Ball speed", ball_speed), ("Direction scale", direction_scale)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(label + " must be a finite positive number.")
    _, _, hand_x, hand_y = forward_kinematics(NEUTRAL_SHOULDER, NEUTRAL_ELBOW)
    return BallState(
        impact_x=hand_x + direction_scale * case.threat_x,
        impact_y=hand_y + direction_scale * case.threat_y,
        z=case.ttc_s * ball_speed,
        speed=ball_speed,
    )


def run_cases(cases, ball_speed=DEFAULT_BALL_SPEED, direction_scale=DIRECTION_SCALE,
              ttc_threshold=1.0, geometry_dt=GEOMETRY_DT,
              intrusion_clearance=INTRUSION_CLEARANCE,
              reaction_threshold=REACTION_THRESHOLD, controller="p", pid_config=None, spiking_config=None):
    results = []
    for case in cases:
        # New core per case: reset the arm and hold Excel urgency constant.
        simulation = SimulationCore(
            ball=case_to_ball(case, ball_speed, direction_scale),
            urgency=case.urgency,
            ttc_threshold=ttc_threshold,
            controller=controller, pid_config=pid_config, spiking_config=spiking_config,
        )
        try:
            metrics = evaluate_encounter(simulation, geometry_dt, intrusion_clearance,
                                         reaction_threshold)
        finally:
            if getattr(simulation.controller, "is_spiking_controller", False):
                simulation.controller.close()
        results.append(TrialResult(case.case_id, simulation.result, metrics, case))
    return results


def summarize_results(results):
    """Aggregate observed rates; undefined rates/latencies are NA, never zero."""
    if not results:
        raise ValueError("Cannot summarize an empty set of trials.")
    total = len(results)
    collisions = sum(r.metrics.contact for r in results)
    threats = [r for r in results if r.metrics.baseline_contact]
    non_threats = [r for r in results if not r.metrics.baseline_contact]
    false_triggers = sum(r.metrics.false_trigger for r in non_threats)
    no_response = sum(r.metrics.first_reflex_time_s is None for r in threats)
    late_response = sum(
        r.metrics.first_reflex_time_s is not None
        and not r.metrics.reflex_before_baseline_contact for r in threats
    )
    latencies = [r.metrics.reaction_latency_s * 1000.0 for r in threats
                 if r.metrics.reaction_status == "responded"]
    clearances = [r.metrics.min_clearance_m * 1000.0 for r in results]
    rows = []

    def add(name, value, unit, numerator="", denominator=""):
        rows.append({"Metric": name, "Value": value, "Unit": unit,
                     "Numerator": numerator, "Denominator": denominator})

    def rate(name, numerator, denominator):
        add(name, 100.0 * numerator / denominator if denominator else "NA",
            "percent", numerator, denominator)

    add("total_cases", total, "runs")
    add("successes", total - collisions, "runs")
    add("collisions", collisions, "runs")
    rate("success_rate", total - collisions, total)
    rate("collision_rate", collisions, total)
    add("baseline_threatening_cases", len(threats), "runs")
    add("baseline_non_threatening_cases", len(non_threats), "runs")
    add("false_triggers", false_triggers, "runs")
    rate("false_trigger_rate", false_triggers, len(non_threats))
    add("threats_without_response", no_response, "runs")
    rate("threat_no_response_rate", no_response, len(threats))
    add("threats_with_late_response", late_response, "runs")
    add("collisions_after_reflex", sum(
        r.metrics.reflex_before_actual_contact is True for r in results
    ), "runs")
    add("min_clearance_worst", min(clearances), "mm")
    add("min_clearance_median", float(np.median(clearances)), "mm")
    add("min_clearance_p05", float(np.percentile(clearances, 5)), "mm")
    add("reaction_latency_samples", len(latencies), "runs")
    add("pre_intrusion_responses", sum(
        r.metrics.reaction_status == "pre_intrusion" for r in results
    ), "runs")
    for label, percentile in (("median", 50), ("p95", 95), ("max", 100)):
        add("reaction_latency_" + label,
            float(np.percentile(latencies, percentile)) if latencies else "NA", "ms")
    add("non_threatening_joint_motion_mean",
        float(np.mean([r.metrics.total_joint_motion_deg for r in non_threats]))
        if non_threats else "NA", "deg")
    return rows


def write_metrics_csv(path, results, configuration):
    """One rectangular CSV containing trial, summary, and configuration rows."""
    fields = [
        "Row_Type", "Case_ID", "Result", "threat_x", "threat_y", "urgency",
        "initial_ttc_s", "contact", "min_clearance_mm", "min_clearance_time_s",
        "nearest_link", "first_contact_time_s", "intrusion_time_s",
        "first_reflex_time_s", "reaction_latency_ms", "reaction_status",
        "baseline_contact", "baseline_min_clearance_mm",
        "baseline_first_contact_time_s", "false_trigger",
        "reflex_before_actual_contact", "reflex_before_baseline_contact",
        "total_joint_motion_deg", "max_joint_deviation_deg", "duration_s",
        "Metric", "Value", "Unit", "Numerator", "Denominator",
    ]
    summary = summarize_results(results)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            m, case = result.metrics, result.case
            row = {
                "Row_Type": "trial", "Case_ID": result.case_id, "Result": result.result,
                "threat_x": case.threat_x, "threat_y": case.threat_y,
                "urgency": case.urgency, "initial_ttc_s": case.ttc_s,
                "contact": int(m.contact), "min_clearance_mm": m.min_clearance_m * 1000.0,
                "min_clearance_time_s": m.min_clearance_time_s, "nearest_link": m.nearest_link,
                "first_contact_time_s": m.first_contact_time_s,
                "intrusion_time_s": m.intrusion_time_s,
                "first_reflex_time_s": m.first_reflex_time_s,
                "reaction_latency_ms": m.reaction_latency_s * 1000.0
                if m.reaction_latency_s is not None else None,
                "reaction_status": m.reaction_status,
                "baseline_contact": int(m.baseline_contact),
                "baseline_min_clearance_mm": m.baseline_min_clearance_m * 1000.0,
                "baseline_first_contact_time_s": m.baseline_first_contact_time_s,
                "false_trigger": int(m.false_trigger),
                "reflex_before_actual_contact": int(m.reflex_before_actual_contact)
                if m.reflex_before_actual_contact is not None else None,
                "reflex_before_baseline_contact": int(m.reflex_before_baseline_contact)
                if m.reflex_before_baseline_contact is not None else None,
                "total_joint_motion_deg": m.total_joint_motion_deg,
                "max_joint_deviation_deg": m.max_joint_deviation_deg,
                "duration_s": m.duration_s,
            }
            writer.writerow(row)
        writer.writerows({"Row_Type": "summary", **row} for row in summary)
        for name, value, unit in configuration:
            writer.writerow({"Row_Type": "configuration", "Metric": name,
                             "Value": value, "Unit": unit})


def main(argv=None):
    from motor_controller import add_controller_arguments, config_from_args
    from spiking_config import spiking_config_from_args
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("workbook", nargs="?", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--ball-speed", type=float, default=DEFAULT_BALL_SPEED,
                        help="Constant approach speed in m/s (default: 1.2).")
    parser.add_argument("--direction-scale", type=float, default=DIRECTION_SCALE,
                        help="Metres per normalized direction unit (default: 0.20).")
    parser.add_argument("--ttc-threshold", type=float, default=1.0,
                        help="Activate avoidance below this TTC in seconds (default: 1.0).")
    parser.add_argument("--geometry-dt", type=float, default=GEOMETRY_DT,
                        help="Geometry observation step in seconds (default: 0.002).")
    parser.add_argument("--intrusion-clearance", type=float, default=INTRUSION_CLEARANCE,
                        help="Nominal-arm surface clearance defining intrusion in metres (default: 1.5).")
    parser.add_argument("--reaction-threshold", type=float, default=REACTION_THRESHOLD,
                        help="Command deviation defining a reflex in deg/s (default: 1.0).")
    parser.add_argument("--csv", type=Path, default=Path("controller_metrics.csv"),
                        help="Metrics CSV path (default: controller_metrics.csv).")
    args = parser.parse_args(argv)
    try:
        cases = load_static_cases(args.workbook)
        results = run_cases(cases, args.ball_speed, args.direction_scale,
                            args.ttc_threshold, args.geometry_dt,
                            args.intrusion_clearance, args.reaction_threshold,
                            args.controller, config_from_args(args),
                            spiking_config_from_args(args) if args.controller == "spiking-pid" else None)
        configuration = [
            ("workbook_path", str(args.workbook.resolve()), "path"),
            ("ball_speed", args.ball_speed, "m/s"),
            ("direction_scale", args.direction_scale, "m"),
            ("ttc_threshold", args.ttc_threshold, "s"),
            ("control_dt", CONTROL_DT, "s"),
            ("geometry_dt", args.geometry_dt, "s"),
            ("intrusion_clearance", args.intrusion_clearance, "m"),
            ("reaction_threshold", args.reaction_threshold, "deg/s"),
            ("contact_tolerance", CONTACT_TOLERANCE, "m"),
            ("ball_radius", BALL_RADIUS, "m"),
            ("arm_radius", ARM_RADIUS, "m"),
            ("link_1_length", L1, "m"),
            ("link_2_length", L2, "m"),
            ("shoulder_max", SHOULDER_MAX, "deg/s"),
            ("elbow_max", ELBOW_MAX, "deg/s"),
            ("neutral_shoulder", NEUTRAL_SHOULDER, "deg"),
            ("neutral_elbow", NEUTRAL_ELBOW, "deg"),
        ]
        if args.controller == "pid":
            from dataclasses import asdict
            configuration.append(("controller", "pid", ""))
            configuration.extend(("pid_" + k, str(v), "")
                                 for k, v in asdict(config_from_args(args)).items())
        if args.controller == "spiking-pid":
            from dataclasses import asdict
            configuration.append(("controller", "spiking-pid", ""))
            configuration.extend(("spiking_"+k,str(v),"") for k,v in asdict(spiking_config_from_args(args)).items())
        write_metrics_csv(args.csv, results, configuration)
    except (OSError, ValueError, BadZipFile, ET.ParseError, KeyError,
            StopIteration, IndexError) as error:
        parser.exit(2, f"Error: {error}\n")

    total = len(results)
    successes = sum(r.result == "SAFE" for r in results)
    print(f"Total cases: {total}")
    print(f"Successes: {successes}")
    print(f"Collisions: {total - successes}")
    print(f"Success rate: {successes / total:.2%}")
    summary = {row["Metric"]: row for row in summarize_results(results)}
    print(f"Collision rate: {(total - successes) / total:.2%}")
    print(f"Worst clearance: {summary['min_clearance_worst']['Value']:.3f} mm")
    false_rate = summary["false_trigger_rate"]["Value"]
    print("False trigger rate: " + ("NA (no non-threatening cases)" if false_rate == "NA"
                                      else f"{false_rate:.2f}%"))
    latency = summary["reaction_latency_median"]["Value"]
    print("Median reaction latency: " + ("NA (no measured threat responses)" if latency == "NA"
                                           else f"{latency:.3f} ms"))
    print(f"CSV: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
