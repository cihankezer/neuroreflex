# Perception and integration boundary

`PerceptionModel` consumes synthetic physical state and returns position, velocity, predicted impact, TTC, looming, bearings, urgency, timestamp, and held status. Dynamic observations use a future Z=0 crossing when available. Parallel motion can use positive radial closing speed. Receding/post-plane trajectories have infinite TTC and zero looming. Static compatibility arithmetic has its own documented behavior in the historical fixtures.

Gaussian XY error, relative TTC noise, buffered transport delay, Bernoulli acquisition dropout, hold timeout, outliers, velocity scaling/sign corruption, and acquisition bursts are configurable and seeded per trial. They are synthetic stress models. Geometry metrics retain true uncorrupted state. No perfect-state fallback is inserted into control.

The author's `perception_adapter.py` and `visual_mapping.py` define separate event-observation and uncalibrated left/centre/right mapping interfaces. Visual bearing/localization is not metric depth, TTC, or calibrated robot coordinates. `visual_reflex_policy.py` and its legacy policy are integration artifacts; they are not exercised by the portable benchmark table.

The team's gated GME/LGMD/LIF looming engines and raw event recordings are excluded from this personal repository. An event-camera-to-motor end-to-end demonstration cannot be reproduced from this release alone. No live sensor performance is claimed.
