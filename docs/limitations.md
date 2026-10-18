# Limitations

This is research/portfolio simulation software. The plant is a planar, velocity-integrated arm interacting with prescribed 3D obstacle paths. It omits inertia, torque, actuator lag, gravity/contact response, measured sensor calibration, human behavior, and real robot safety supervision.

Geometric threat prediction freezes current arm geometry and assumes perceived constant obstacle velocity over its horizon. The controller's future arm motion is not predicted. Hysteretic hold can preserve intent during missing perception but cannot establish collision freedom. Timing follows the historical tick-end update/interpolation convention; it is not a causal model of a physical sensor/actuator system.

Noise and dropout distributions are configurable synthetic perturbations, not fitted event-camera statistics. Curated and seeded trials do not establish real-world reliability. No confidence interval or superiority claim is inferred from zero observed collisions. Reaction summaries exclude negative pre-intrusion timings and missing responses rather than treating them as zero.

The Lava implementation runs on CPU. Neither spiking representation nor a Loihi-named simulation configuration demonstrates deployment, latency advantage, or energy savings. Integral action is implemented classically but disabled in the frozen PD/spiking reference.

Only the author's contribution is released. Shared GUI, teammates' perception engines, event recordings, reports/slides, and unrelated third-party assets are excluded. Historical full-project preservation checks are documented separately from passing portable tests. No license grant is provided.
