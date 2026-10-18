"""Passive state-transition and recovery metrics from true geometry samples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReflexEncounterMetrics:
    activation_count: int
    release_count: int
    reactivation_count: int
    total_reflex_active_time_s: float
    first_activation_time_s: float | None
    final_release_time_s: float | None
    number_of_gate_chatter_events: int
    minimum_clearance_after_final_release_m: float | None
    time_from_final_release_to_closest_approach_s: float | None
    minimum_clearance_during_recovery_m: float | None
    active_at_encounter_end: bool
    transition_count: int
    last_release_time_s: float | None


def active_episodes(ticks, duration_s, control_dt=0.02):
    """Event times use tick ends; durations sum clipped active tick intervals."""
    episodes = []
    current = None
    for row in ticks:
        if row["reflex_activation_event"]:
            current = {"activation_time_s": row["time_s"], "release_time_s": None,
                       "active_duration_s": 0.0}
            episodes.append(current)
        if row["final_reflex_active"] and current is not None:
            weight = max(0, min(control_dt, duration_s - (row["time_s"] - control_dt)))
            current["active_duration_s"] += weight
        if row["reflex_release_event"] and current is not None:
            current["release_time_s"] = row["time_s"]
            current = None
    return episodes


def calculate_release_metrics(ticks, samples, duration_s, control_dt=0.02,
                              chatter_window_s=0.1):
    """Rapid release-to-reactivation <=100 ms is a chatter proxy, not an oracle.

    Long reactivation intervals are retained as reactivations without labelling
    them chatter. A real renewed threat can also be rapid; inspect the trace.
    samples = (time, TRUE surface clearance, recovery_active) from the observer.
    Only a completed final active episode defines final-release clearance.
    """
    activations, releases = [], []
    chatter = 0
    last_release = None
    for row in ticks:
        if row["reflex_activation_event"]:
            activations.append(row["time_s"])
            if last_release is not None and row["time_s"] - last_release <= chatter_window_s + 1e-12:
                chatter += 1
        if row["reflex_release_event"]:
            last_release = row["time_s"]
            releases.append(last_release)
    active_end = bool(ticks and ticks[-1]["final_reflex_active"])
    final_release = last_release if not active_end else None
    after = [(time, clearance) for time, clearance, _ in samples
             if final_release is not None and time + 1e-12 >= final_release]
    closest = min(after, key=lambda s: s[1]) if after else None
    recovery_clearances = [c for _, c, recovering in samples if recovering]
    active_time = sum(e["active_duration_s"] for e in active_episodes(ticks, duration_s, control_dt))
    return ReflexEncounterMetrics(len(activations), len(releases), max(0, len(activations) - 1),
        active_time, activations[0] if activations else None, final_release, chatter,
        closest[1] if closest else None,
        max(0, closest[0] - final_release) if closest else None,
        min(recovery_clearances) if recovery_clearances else None, active_end,
        len(activations) + len(releases), last_release)
