"""Explicit interval/activity metrics; unknown time is never filled with OFF."""
import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {name: None for name in ("mean", "median", "p90", "p95", "p99", "max")}
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            **{f"p{p}": float(np.percentile(values, p)) for p in (90, 95, 99)}, "max": float(values.max())}


def _merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def response_metrics(timestamps, values, first_us, last_us, *, expiry_us=10_000):
    """Causal sample coverage [timestamp, next/expiry/EOF), integer us.

    Invalid/untimed samples cannot create an observation. ON/OFF transitions
    require contiguous known intervals; unknown gaps can split positive episodes
    but are not labelled OFF. The last partial-bin nominal timestamp beyond EOF
    is logged by the native model but contributes no recorded-time coverage.
    """
    if len(timestamps) != len(values) or last_us <= first_us or expiry_us <= 0:
        raise ValueError("Invalid sample/time dimensions")
    samples = [(int(t), value) for t, value in zip(timestamps, values) if t is not None]
    if any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise ValueError("Timed observations must strictly increase")
    if any(v is not None and v not in (False, True, 0, 1) for _, v in samples):
        raise ValueError("Response observations must be binary or explicitly unknown")
    intervals = []
    for index, (time, value) in enumerate(samples):
        if value is None:
            continue
        end = min(last_us, time+expiry_us,
                  samples[index+1][0] if index+1 < len(samples) else last_us)
        start = max(first_us, time)
        if end > start:
            intervals.append((start, end, bool(value)))
    positive = _merge((a, b) for a, b, on in intervals if on)
    negative = _merge((a, b) for a, b, on in intervals if not on)
    known = _merge((a, b) for a, b, _ in intervals)
    missing, cursor = [], first_us
    for start, end in known:
        if cursor < start:
            missing.append((cursor, start))
        cursor = end
    if cursor < last_us:
        missing.append((cursor, last_us))
    positive_us = sum(b-a for a, b in positive)
    known_us = sum(b-a for a, b in known)
    duration = last_us-first_us
    episode_s = [(b-a)/1e6 for a, b in positive]
    gaps, known_off_gaps = [], []
    for (_, end), (start, _) in zip(positive, positive[1:]):
        gap = start-end
        gaps.append(gap/1e6)
        covered_off = sum(max(0, min(start, b)-max(end, a)) for a, b in negative)
        if covered_off == gap:
            known_off_gaps.append(gap/1e6)
    transitions = sum(a[1] == b[0] and a[2] != b[2] for a, b in zip(intervals, intervals[1:]))
    positive_samples = [t for t, v in samples if v is not None and bool(v) and first_us <= t <= last_us]
    # Separate observed sample runs from timed intervals. Small native timestamp
    # gaps above 10ms split coverage without constituting an observed OFF sample.
    sample_runs, sample_changes, previous = 0, 0, None
    for time, value in zip(timestamps, values):
        current = bool(value) if time is not None and value is not None else None
        if current is True and previous is not True:
            sample_runs += 1
        if previous is not None and current is not None and previous != current:
            sample_changes += 1
        previous = current
    return {"expiry_us": expiry_us, "sample_count": len(values),
            "valid_output_window_fraction": sum(t is not None and v is not None for t, v in zip(timestamps, values))/len(values) if values else None,
            "recorded_time_output_coverage_fraction": known_us/duration,
            "positive_fraction_full_recording": positive_us/duration,
            "positive_fraction_valid_output_time": positive_us/known_us if known_us else None,
            "total_positive_s": positive_us/1e6, "positive_intervals": len(positive),
            "on_off_transitions": int(transitions), "positive_episode_count": len(positive),
            "positive_run_count_in_observation_sequence": sample_runs,
            "on_off_changes_between_consecutive_timed_samples": sample_changes,
            "longest_positive_s": max(episode_s) if episode_s else 0.,
            "shortest_positive_s": min(episode_s) if episode_s else None,
            "median_positive_s": float(np.median(episode_s)) if episode_s else None,
            "median_known_off_gap_s": float(np.median(known_off_gaps)) if known_off_gaps else None,
            "median_inter_episode_gap_s_including_unknown": float(np.median(gaps)) if gaps else None,
            "missing_output_s": sum(b-a for a, b in missing)/1e6,
            "longest_missing_output_s": max((b-a for a, b in missing), default=0)/1e6,
            "first_positive_relative_s": (positive_samples[0]-first_us)/1e6 if positive_samples else None,
            "last_positive_relative_s": (positive_samples[-1]-first_us)/1e6 if positive_samples else None,
            "positive_intervals_us": positive, "missing_intervals_us": missing}


def localization_metrics(timestamps, centers, first_us, last_us):
    def zone(center):
        return "LEFT" if center[0] < 1/3 else "RIGHT" if center[0] > 2/3 else "CENTER"
    valid = [c is not None for c in centers]
    coverage = response_metrics(timestamps, valid, first_us, last_us)
    jitter, transitions = [], 0
    for index in range(1, len(centers)):
        if (centers[index] is not None and centers[index-1] is not None and
                timestamps[index] is not None and timestamps[index-1] is not None and
                0 < timestamps[index]-timestamps[index-1] <= 10_000):
            jitter.append(float(np.linalg.norm(np.subtract(centers[index], centers[index-1]))))
            transitions += zone(centers[index]) != zone(centers[index-1])
    available = [c for c in centers if c is not None]
    return {"localized_window_fraction": sum(valid)/len(valid) if valid else None,
            "localized_time_fraction": coverage["positive_fraction_full_recording"],
            "center_x_normalized": distribution([c[0] for c in available]),
            "center_y_normalized": distribution([c[1] for c in available]),
            "jitter_normalized": distribution(jitter), "zone_transitions": transitions,
            "zone_window_counts": {name: sum(zone(c) == name for c in available) for name in ("LEFT", "CENTER", "RIGHT")}}


def safe_ratio(numerator, denominator):
    return {"value": numerator/denominator if denominator else None,
            "status": "defined" if denominator else "undefined_zero_denominator"}


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y):
        raise ValueError("Correlation samples must be aligned")
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 0 and y.std() > 0 else None


def spike_characterization(rows, neuron_totals, recording_duration_s, dt_s=.01):
    layers = {}
    for name, totals in neuron_totals.items():
        counts = np.array([r["spike_counts"][name] for r in rows], dtype=float)
        n = len(totals)
        layers[name] = {"output_unit_count": n, "total_spikes": int(counts.sum()),
                        "spikes_per_recorded_second": float(counts.sum()/recording_duration_s),
                        "spikes_per_window": distribution(counts),
                        "population_rate_hz_per_native_step": distribution(counts/(n*dt_s)),
                        "per_neuron_rate_hz_per_recorded_second": distribution(np.array(totals)/recording_duration_s),
                        "active_neuron_fraction_ever": sum(v > 0 for v in totals)/n,
                        "mean_active_fraction_per_window": float(counts.mean()/n),
                        "temporal_sparsity_zero_spike_fraction": float(np.mean(counts == 0)),
                        "population_sparsity_never_active_fraction": sum(v == 0 for v in totals)/n}
    all_counts = np.array([sum(r["spike_counts"].values()) for r in rows], dtype=float)
    lif_counts = np.array([sum(r["spike_counts"][n] for n in ("s_on", "s_off", "g", "lgmd")) for r in rows], dtype=float)
    filtered_events = sum(r["input_on_events"]+r["input_off_events"] for r in rows)
    positive_windows = sum(r["collision"] for r in rows)
    return {"comparison_status": "NEW MODEL CHARACTERIZATION — NOT DIRECTLY COMPARABLE",
            "layers": layers, "lif_neuron_count": 28, "p_binary_threshold_channel_count": 41472,
            "gme_integer_counter_count": 1296, "explicit_plastic_weight_count": 18,
            "total_hardware_synapse_count": None, "native_step_count": len(rows),
            "native_process_update_count": 9*len(rows), "lif_state_update_count": 28*len(rows),
            "total_coded_spikes_including_p": int(all_counts.sum()), "total_lif_spikes": int(lif_counts.sum()),
            "lif_population_rate_hz_per_native_step": distribution(lif_counts/(28*dt_s)),
            "spikes_per_native_window": distribution(all_counts),
            "spikes_per_recorded_second_including_p": float(all_counts.sum()/recording_duration_s),
            "spikes_per_filtered_input_event_including_p": float(all_counts.sum()/filtered_events) if filtered_events else None,
            "output_spikes": positive_windows, "output_spikes_per_positive_output_bin": 1. if positive_windows else None,
            "all_layer_spikes_per_positive_output_bin": float(all_counts.sum()/positive_windows) if positive_windows else None}
