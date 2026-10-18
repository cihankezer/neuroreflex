"""Explicit NEF realization parameters; the conventional gains are never tuned."""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


SPIKING_PRESET = Path(__file__).with_name("spiking_reference.json")
if not SPIKING_PRESET.is_file():
    from importlib.resources import files
    SPIKING_PRESET = files("neuroreflex").joinpath("configs/spiking_reference.json")


@dataclass(frozen=True)
class SpikingConfig:
    neurons_per_ensemble: int = 250
    seed: int = 2026
    neural_dt_s: float = .001
    membrane_tau_s: float = .005
    error_radius_deg: tuple = (140., 100.)
    output_radius_deg_s: tuple = (1250., 900.)
    intercept_range: tuple = (-.9, .9)
    max_rate_range_hz: tuple = (120., 360.)
    decoder_regularization: float = .001
    p_synapse_tau_s: float = .005
    output_synapse_tau_s: float = .01
    tau_short_s: float = .005
    tau_long_s: float = .049326069247528636
    derivative_scale_per_s: float = 25.

    def __post_init__(self):
        if not isinstance(self.neurons_per_ensemble, int) or self.neurons_per_ensemble < 8 or self.neurons_per_ensemble % 2:
            raise ValueError("Use an even population size >=8 for paired bipolar encoders.")
        for key in ("error_radius_deg", "output_radius_deg_s", "intercept_range", "max_rate_range_hz"):
            values = tuple(getattr(self, key))
            object.__setattr__(self, key, values)
            if len(values) != 2 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"{key} must contain two finite values.")
        if min(*self.error_radius_deg, *self.output_radius_deg_s) <= 0:
            raise ValueError("Representation radii must be positive.")
        if not -1 < self.intercept_range[0] < self.intercept_range[1] < 1:
            raise ValueError("Intercepts must lie inside (-1,1).")
        for key in ("neural_dt_s", "membrane_tau_s", "p_synapse_tau_s", "output_synapse_tau_s",
                    "tau_short_s", "tau_long_s", "derivative_scale_per_s", "decoder_regularization"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite.")
        if not 0 < self.max_rate_range_hz[0] < self.max_rate_range_hz[1] < 1 / self.neural_dt_s:
            raise ValueError("Firing range must be positive and below one spike/neural step.")
        if self.tau_short_s >= self.tau_long_s:
            raise ValueError("Derivative requires tau_short < tau_long.")
        if abs(.02 / self.neural_dt_s - round(.02 / self.neural_dt_s)) > 1e-10:
            raise ValueError("Neural dt must exactly subdivide the frozen 20 ms interval.")


def load_spiking_config(path=SPIKING_PRESET):
    return SpikingConfig(**json.loads(Path(path).read_text(encoding="utf-8"))["config"])


def add_spiking_arguments(parser):
    parser.add_argument("--spiking-config", type=Path, default=SPIKING_PRESET)
    parser.add_argument("--spiking-neurons", type=int)
    parser.add_argument("--spiking-seed", type=int)
    parser.add_argument("--neural-dt", type=float)


def spiking_config_from_args(args):
    if getattr(args, "controller", None) == "spiking-pid" and any(
            value is not None for key,value in vars(args).items() if key.startswith("pid_")):
        raise ValueError("Spiking PD uses the frozen conventional gains; use its separate neural configuration.")
    config = asdict(load_spiking_config(args.spiking_config))
    for argument, field in (("spiking_neurons", "neurons_per_ensemble"), ("spiking_seed", "seed"),
                            ("neural_dt", "neural_dt_s")):
        if getattr(args, argument) is not None:
            config[field] = getattr(args, argument)
    return SpikingConfig(**config)
