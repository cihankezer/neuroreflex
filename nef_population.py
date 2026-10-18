"""Offline regularized NEF decoders fitted to Lava's discrete LIF recurrence.

Only construction/calibration uses linear algebra. Online signals are spikes.
Matched +/- encoder pairs enforce zero reconstruction without a host deadband.
"""

from functools import lru_cache
import numpy as np


def discrete_rates(currents, decay, dt):
    drive = decay * currents
    valid = drive > decay
    ratio = np.zeros_like(drive)
    ratio[valid] = np.log1p(-decay / drive[valid]) / np.log1p(-decay)
    return np.where(valid, 1 / ((np.floor(ratio) + 1) * dt), 0.)


@lru_cache(maxsize=16)
def fit_populations(config):
    half = config.neurons_per_ensemble // 2
    grid = np.linspace(-1, 1, 801)
    decay = 1 - np.exp(-config.neural_dt_s / config.membrane_tau_s)
    result = {}
    for kind, offset in (("error", 0), ("output", 10000)):
        parameters = {k: [] for k in ("gain", "bias", "decoder", "initial_v", "intercepts", "max_rates")}
        for joint in range(2):
            rng = np.random.default_rng(config.seed + offset + joint * 101)
            intercepts = rng.uniform(*config.intercept_range, half)
            max_rates = rng.uniform(*config.max_rate_range_hz, half)
            max_current = 1 / (1 - np.exp(-1 / (max_rates * config.membrane_tau_s)))
            gain = (max_current - 1) / (1 - intercepts)
            bias = 1 - gain * intercepts
            positive = discrete_rates(grid[:, None] * gain + bias, decay, config.neural_dt_s)
            negative = discrete_rates(-grid[:, None] * gain + bias, decay, config.neural_dt_s)
            activity = positive - negative
            gram = activity.T @ activity
            ridge = config.decoder_regularization * np.trace(gram) / half
            decoder = np.linalg.solve(gram + ridge * np.eye(half), activity.T @ grid)
            phase = rng.uniform(0, 1, half)
            parameters["gain"].append(np.r_[gain, -gain])
            parameters["bias"].append(np.r_[bias, bias])
            parameters["decoder"].append(decoder)
            parameters["initial_v"].append(np.r_[phase, phase])
            parameters["intercepts"].append(intercepts)
            parameters["max_rates"].append(max_rates)
        result[kind] = {k: np.asarray(v) for k, v in parameters.items()}
    return result
