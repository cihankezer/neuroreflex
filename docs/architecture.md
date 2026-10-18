# Architecture

The portable pipeline has three information boundaries. `trajectories.py` defines physical motion; `perception.py` produces estimates with independent corruption; `simulation_core.py` sends only those estimates and robot feedback to the controller. True obstacle state is reserved for evaluation.

`threat_gate.py` computes exact segment-to-segment distance between the perceived obstacle sweep and each current arm centre line, then subtracts radii. TTC and clearance jointly gate entry. `reflex_release.py` implements asymmetric release, fresh-evidence dwell, and missing-perception pose hold. Legacy and memoryless policies remain selectable for comparison.

The escape/IK function produces a common joint-space objective. The motor backend returns requested velocities, and the core applies urgency-dependent velocity clamps and joint bounds. `dynamic_metrics.py` observes encounter geometry independently; `diagnostics.py`, `release_metrics.py`, and `state_activation_metrics.py` retain motor and state diagnostics.

Public modules keep the established flat import layout. `neuroreflex` exposes a small stable interface; `pyproject.toml` explicitly enumerates installable modules and packages preset/scenario resources. Only data-resource lookup was adjusted for wheel installation; numerical controller logic is preserved.

The original shared GUI and teammates' visual SNN/GME code are absent. The public `neuromorphic_lava` package is the author's motor graph, not the team's visual looming graph.
