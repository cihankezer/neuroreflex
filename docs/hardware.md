# Lava runtime and hardware status

| Execution path | Status in this release |
| --- | --- |
| NumPy/classical CPU simulation | Clean installation and portable tests verified |
| Lava CPU floating-point motor graph | 20 unittest checks passed in existing Python 3.10.22 / Lava 0.10.0 environment |
| Visual event SNN / GME engine | Teammate contribution; excluded |
| Loihi 2 deployment | Not implemented or verified by this release |
| Real robot, sensor-to-motor latency, energy measurement | Not executed |

The code uses `Loihi2SimCfg(select_tag="floating_pt")` with a CPU `PyLoihiProcessModel`. The configuration's name does not mean it executes on Loihi 2 hardware. The fused graph runs LIF population recurrences, NEF transforms, and synaptic states in a Python CPU model.

The historical pinned dependencies are in `requirements-spiking.txt`. Use a separate Python 3.10 environment; do not install them into the portable Python 3.13 environment:

```bash
python3.10 -m venv .venv-lava
# Activate it, then:
python -m pip install -r requirements-spiking.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -p test_spiking.py -v
```

The tests passed using the already provisioned local Lava environment. A fresh installation of all pinned Lava dependencies was not attempted. The Windows CI job validates only the portable core. Hardware execution would require compatible neuromorphic process models, quantization/mapping work, an appropriate runtime, and access to a device; this repository supplies no validated hardware deployment procedure.

See the [official Lava execution documentation](https://lava-nc.org/lava/notebooks/in_depth/tutorial04_execution.html).
