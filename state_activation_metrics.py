"""Controller-independent trial false triggers from truth labels and reflex events.

Ground truth is the evaluator's existing baseline_contact: contact anywhere in
the physical encounter against the stationary neutral arm, using true geometry.
It is a trial label, not a tick-wise threat detector or perception heuristic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateActivationMetrics:
    state_activation_count: int
    state_activation_first_time_s: float | None
    state_activation_false_trigger: bool
    state_activation_false_trigger_events: int


def calculate_state_activation_metrics(ticks, baseline_contact, duration_s=None, control_dt=.02):
    """NOMINAL -> {REFLEX_ACTIVE, PERCEPTION_HOLD} is an activation episode.

    Use the state machine's reflex_activation_event, never a motor command.
    ACTIVE -> HOLD -> ACTIVE continues the same episode. A release followed by
    re-entry adds an event, but the trial false-trigger flag still counts once.
    Include a partial final control interval using the existing plant convention.
    """
    events = [row for row in ticks if row["reflex_activation_event"]
              and row["reflex_state"] in ("REFLEX_ACTIVE", "PERCEPTION_HOLD")
              and (duration_s is None or row["time_s"] - control_dt < duration_s - 1e-12)]
    false = not baseline_contact and bool(events)
    return StateActivationMetrics(len(events), events[0]["time_s"] if events else None,
                                  false, len(events) if not baseline_contact else 0)


def summarize_state_activation_metrics(results):
    """Rate denominator = ground-truth non-threat trials; no denominator -> NA."""
    eligible = [r for r in results if not r.metrics.baseline_contact]
    false_trials = sum(r.state_activation_metrics.state_activation_false_trigger for r in eligible)
    events = sum(r.state_activation_metrics.state_activation_false_trigger_events for r in eligible)
    return [
        {"Metric": "state_activation_false_trigger_trials", "Value": false_trials, "Unit": "trials"},
        {"Metric": "state_activation_non_threat_trials", "Value": len(eligible), "Unit": "trials"},
        {"Metric": "state_activation_false_trigger_rate", "Value": 100 * false_trials / len(eligible)
         if eligible else "NA", "Unit": "%", "Numerator": false_trials, "Denominator": len(eligible)},
        {"Metric": "state_activation_false_trigger_events", "Value": events, "Unit": "events"},
    ]
