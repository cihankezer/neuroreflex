# Development inventory and implementation choices

The private development workspace was recursively inventoried: 1,163 tracked files, including 1,035 benchmark artifacts and 123 Python files across the current snapshot. Local branches and available remote-tracking branches were inspected read-only; no branch was checked out or merged.

| Development family | Finding | Release decision |
| --- | --- | --- |
| Controller | Classical control, synthetic perception, threat gate, release policy, metrics and robustness runners | Include the author's mature core and selected regression fixtures |
| Controller/Lava motor | NEF/LIF motor graph with a CPU floating-point process model | Include author's graph; verify CPU tests; no hardware claim |
| Spiking perception | Native event SNN/GME experiments, callback adapters, recording studies and visual demos | Exclude teammates' engines and recordings; retain only author's standalone adapters as integration interfaces |
| Looming / pipeline / simple GME | Divergent event-processing implementations with different APIs and gating behavior | Do not merge into personal release |
| Backup branch | Pre-merge historical state | Inventory privately, no duplicate public implementation |

Important conflicts: the team's `lava/` directory collides with Lava SDK discovery; older/newer visual callbacks return different tuples; legacy visual policy and native GME-gated policy have distinct semantics; geometric memoryless release can toggle as the arm moves clear; static and dynamic benchmarks use different observation assumptions. None is resolved by silently combining branches or pooling unlike result tables.

Portable numerical behavior is regression-verified. The frozen conventional PD and CPU spiking motor graph are distinct backends. Event-camera integration and hardware portability remain experimental/out of scope here. Historical reports and results remain intact in the original repository and local private audit snapshot; only the minimal fixtures and newly reproduced evidence are publicly staged.
