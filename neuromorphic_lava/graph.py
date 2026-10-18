"""Lava CPU NEF population graph, with no conventional PD in the forward path.

This fused CPU ProcessModel executes LIF populations, fixed NEF transforms and
leaky synaptic states. It is NOT a compiled Loihi 2 ProcessModel. Its hardware
topology export exposes the primitive mapping; quantization/NC models are a
deployment requirement, not an implied capability of Loihi2SimCfg.
"""

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from nef_population import fit_populations


# Physical units unless explicitly normalized; row index = shoulder/elbow.
DIAG = ("decoded_error_deg", "p_deg_s", "short_filter_error_deg", "long_filter_error_deg",
        "d_deg_s", "requested_velocity_deg_s", "error_spikes", "output_spikes",
        "mean_firing_rate_hz", "peak_firing_rate_hz", "output_representation_out_of_range")


class NEFPD(AbstractProcess):
    def __init__(self, config, reference, replay_inputs=None, replay_modes=None, batch_output_path=None):
        steps = round(.02 / config.neural_dt_s)
        rows = len(replay_inputs) if replay_inputs is not None else 1
        super().__init__(config=config, steps=steps, replay=bool(replay_inputs is not None),
                         radii=config.error_radius_deg, output_radii=config.output_radius_deg_s,
                         batch_output_path=batch_output_path)
        n, half = config.neurons_per_ensemble, config.neurons_per_ensemble // 2
        populations = fit_populations(config)
        self.input_norm = Var(shape=(2,), init=0.)
        self.mode = Var(shape=(1,), init=1.)
        self.reset_flag = Var(shape=(1,), init=1.)
        self.last_mode = Var(shape=(1,), init=-1.)
        self.first_interval = Var(shape=(1,), init=1.)
        self.interval_step = Var(shape=(1,), init=0.)
        self.cursor = Var(shape=(1,), init=0.)
        self.p_trace = Var(shape=(2, half), init=0.)
        self.short_trace = Var(shape=(2, half), init=0.)
        self.long_trace = Var(shape=(2, half), init=0.)
        self.output_trace = Var(shape=(2, half), init=0.)
        self.error_counts = Var(shape=(2, n), init=0.)
        self.output_counts = Var(shape=(2, n), init=0.)
        self.error_spikes = Var(shape=(2, n), init=0.)
        self.output_spikes = Var(shape=(2, n), init=0.)
        self.error_v = Var(shape=(2, n), init=populations["error"]["initial_v"])
        self.output_v = Var(shape=(2, n), init=populations["output"]["initial_v"])
        for kind in ("error", "output"):
            for field in ("gain", "bias", "initial_v", "decoder"):
                data = populations[kind][field]
                setattr(self, f"{kind}_{field}", Var(shape=data.shape, init=data))
        # These are NEF decoding transformation weights, not host Kp*input.
        physical_decoder = populations["error"]["decoder"] * np.asarray(config.error_radius_deg)[:, None]
        self.p_weights_active = Var(shape=(2, half), init=physical_decoder * reference.shoulder.kp)
        self.p_weights_nominal = Var(shape=(2, half), init=physical_decoder * reference.recovery_kp)
        self.d_weights = Var(shape=(2, half), init=physical_decoder * reference.shoulder.kd * config.derivative_scale_per_s)
        self.integral_state = Var(shape=(2,), init=0.)
        self.i_weights = Var(shape=(2,), init=0.)  # Ki=0: no recurrence or output influence in this phase.
        self.diagnostics = Var(shape=(2, len(DIAG)), init=0.)
        self.readout_sum = Var(shape=(2, 6), init=0.)
        self.replay_inputs = Var(shape=(rows, 2), init=replay_inputs if replay_inputs is not None else 0.)
        self.replay_modes = Var(shape=(rows,), init=replay_modes if replay_modes is not None else 1.)
        self.history = Var(shape=(rows, 2, len(DIAG)), init=0.)
        self.raw_spike_projection = Var(shape=(rows * steps, 2), init=0.)


@implements(proc=NEFPD, protocol=LoihiProtocol)
@requires(CPU)
@tag("floating_pt")
class PyNEFPD(PyLoihiProcessModel):
    input_norm: np.ndarray = LavaPyType(np.ndarray, float)
    mode: np.ndarray = LavaPyType(np.ndarray, float)
    reset_flag: np.ndarray = LavaPyType(np.ndarray, float)
    last_mode: np.ndarray = LavaPyType(np.ndarray, float)
    first_interval: np.ndarray = LavaPyType(np.ndarray, float)
    interval_step: np.ndarray = LavaPyType(np.ndarray, float)
    cursor: np.ndarray = LavaPyType(np.ndarray, float)
    p_trace: np.ndarray = LavaPyType(np.ndarray, float)
    short_trace: np.ndarray = LavaPyType(np.ndarray, float)
    long_trace: np.ndarray = LavaPyType(np.ndarray, float)
    output_trace: np.ndarray = LavaPyType(np.ndarray, float)
    error_counts: np.ndarray = LavaPyType(np.ndarray, float)
    output_counts: np.ndarray = LavaPyType(np.ndarray, float)
    error_spikes: np.ndarray = LavaPyType(np.ndarray, float)
    output_spikes: np.ndarray = LavaPyType(np.ndarray, float)
    error_v: np.ndarray = LavaPyType(np.ndarray, float)
    output_v: np.ndarray = LavaPyType(np.ndarray, float)
    error_gain: np.ndarray = LavaPyType(np.ndarray, float)
    error_bias: np.ndarray = LavaPyType(np.ndarray, float)
    error_initial_v: np.ndarray = LavaPyType(np.ndarray, float)
    error_decoder: np.ndarray = LavaPyType(np.ndarray, float)
    output_gain: np.ndarray = LavaPyType(np.ndarray, float)
    output_bias: np.ndarray = LavaPyType(np.ndarray, float)
    output_initial_v: np.ndarray = LavaPyType(np.ndarray, float)
    output_decoder: np.ndarray = LavaPyType(np.ndarray, float)
    p_weights_active: np.ndarray = LavaPyType(np.ndarray, float)
    p_weights_nominal: np.ndarray = LavaPyType(np.ndarray, float)
    d_weights: np.ndarray = LavaPyType(np.ndarray, float)
    integral_state: np.ndarray = LavaPyType(np.ndarray, float)
    i_weights: np.ndarray = LavaPyType(np.ndarray, float)
    diagnostics: np.ndarray = LavaPyType(np.ndarray, float)
    readout_sum: np.ndarray = LavaPyType(np.ndarray, float)
    replay_inputs: np.ndarray = LavaPyType(np.ndarray, float)
    replay_modes: np.ndarray = LavaPyType(np.ndarray, float)
    history: np.ndarray = LavaPyType(np.ndarray, float)
    raw_spike_projection: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        config = proc_params["config"]
        self.h = config.neural_dt_s
        self.steps = proc_params["steps"]
        self.replay = proc_params["replay"]
        self.batch_output_path = proc_params.get("batch_output_path")
        self.radii = np.asarray(proc_params["radii"])
        self.output_radii = np.asarray(proc_params["output_radii"])
        self.half = config.neurons_per_ensemble // 2
        self.dv = 1 - np.exp(-self.h / config.membrane_tau_s)
        self.ap = np.exp(-self.h / config.p_synapse_tau_s)
        self.ash = np.exp(-self.h / config.tau_short_s)
        self.al = np.exp(-self.h / config.tau_long_s)
        self.ao = np.exp(-self.h / config.output_synapse_tau_s)

    def reset_neurons(self):
        self.error_v[:] = self.error_initial_v
        self.output_v[:] = self.output_initial_v
        for state in (self.p_trace, self.short_trace, self.long_trace, self.output_trace,
                      self.integral_state):
            state[:] = 0.
        self.first_interval[:] = 1.
        self.reset_flag[:] = 0.

    def run_spk(self):
        index = int(self.cursor[0])
        step = int(self.interval_step[0])
        if self.replay:
            self.input_norm[:] = self.replay_inputs[index]
            self.mode[:] = self.replay_modes[index]
        mode = int(self.mode[0])
        if step == 0:
            if self.reset_flag[0] or self.last_mode[0] != mode:
                self.reset_neurons()
            self.last_mode[:] = mode
            self.error_counts[:] = 0.
            self.output_counts[:] = 0.
            self.readout_sum[:] = 0.
        if mode == 2:  # Existing perception hold: no neural update or motor output.
            self.diagnostics[:] = 0.
            self.error_spikes[:] = self.output_spikes[:] = 0.
            projection = np.zeros(2)
        else:
            # Lava Float LIF recurrence: v <- (1-dv)*v + activation+bias; threshold >1; reset 0.
            self.error_v[:] = (1-self.dv)*self.error_v + self.dv*(
                self.error_gain*self.input_norm[:, None] + self.error_bias)
            self.error_spikes[:] = self.error_v > 1.
            self.error_v[self.error_spikes.astype(bool)] = 0.
            self.error_counts[:] += self.error_spikes
            differential_spikes = (self.error_spikes[:, :self.half] - self.error_spikes[:, self.half:]) / self.h
            projection = np.sum(self.error_decoder*differential_spikes, axis=1)
            self.p_trace[:] = self.ap*self.p_trace + (1-self.ap)*differential_spikes
            if self.first_interval[0]:
                # Initialize both D synapses from actual population spikes, not the host error.
                self.short_trace[:] = self.p_trace
                self.long_trace[:] = self.p_trace
            else:
                self.short_trace[:] = self.ash*self.short_trace + (1-self.ash)*differential_spikes
                self.long_trace[:] = self.al*self.long_trace + (1-self.al)*differential_spikes
            weights = self.p_weights_nominal if mode == 0 else self.p_weights_active
            p = np.sum(weights*self.p_trace, axis=1)
            d = np.sum(self.d_weights*(self.short_trace-self.long_trace), axis=1)
            # NEF outer-product transform factored into decoder/encoder/gain. No actuator clamp.
            normalized_drive = (p+d) / self.output_radii
            self.output_v[:] = (1-self.dv)*self.output_v + self.dv*(
                self.output_gain*normalized_drive[:, None] + self.output_bias)
            self.output_spikes[:] = self.output_v > 1.
            self.output_v[self.output_spikes.astype(bool)] = 0.
            self.output_counts[:] += self.output_spikes
            out_diff = (self.output_spikes[:, :self.half] - self.output_spikes[:, self.half:]) / self.h
            self.output_trace[:] = self.ao*self.output_trace + (1-self.ao)*out_diff
            requested = self.output_radii*np.sum(self.output_decoder*self.output_trace, axis=1)
            decoded_error = self.radii*np.sum(self.error_decoder*self.p_trace, axis=1)
            short = self.radii*np.sum(self.error_decoder*self.short_trace, axis=1)
            long = self.radii*np.sum(self.error_decoder*self.long_trace, axis=1)
            # Fixed linear observation window; all six signals remain spike derived.
            # Decode at the interval boundary using this interval's mean, not a queued tick.
            self.readout_sum[:] += np.stack((decoded_error, p, short, long, d, requested), axis=1)
            self.diagnostics[:, :6] = self.readout_sum / (step+1)
            self.diagnostics[:, 10] = np.abs(normalized_drive) > 1.
        self.diagnostics[:, 6] = self.error_counts.sum(axis=1)
        self.diagnostics[:, 7] = self.output_counts.sum(axis=1)
        self.diagnostics[:, 8] = (self.error_counts.sum(axis=1) + self.output_counts.sum(axis=1)) / (
            2*self.half*2*(step+1)*self.h)
        self.diagnostics[:, 9] = np.maximum(self.error_counts.max(axis=1), self.output_counts.max(axis=1)) / ((step+1)*self.h)
        if self.replay:
            self.raw_spike_projection[index*self.steps+step] = projection
        step += 1
        if step == self.steps:
            if self.replay:
                self.history[index] = self.diagnostics
                self.cursor[:] += 1.
                if self.batch_output_path and index == len(self.replay_modes)-1:
                    # CPU-only passive recorder. Avoid millions of scalar Lava Var.get RPCs.
                    # This output is never fed back into the control computation.
                    np.savez(self.batch_output_path, history=self.history,
                             raw_projection=self.raw_spike_projection)
            self.first_interval[:] = 0.
            step = 0
        self.interval_step[:] = step
