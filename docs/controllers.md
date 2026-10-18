# Controllers

The common avoidance path moves an end-effector target away from predicted impact, clamps it to the reachable workspace, and computes analytic two-link IK. Nominal control returns toward the neutral shoulder/elbow pose of 35/55 degrees.

- **P:** historical TTC-scheduled proportional escape control; default CLI behavior.
- **PD/PID:** `motor_controller.PIDController` integrates joint error conditionally to prevent windup, filters error differentiation, bounds the integral contribution, and resets state on mode changes. `Ki=0` disables active integration. The frozen PD reference has Kp=5.25, Ki=0, Kd=0.1, derivative filter tau=40 ms, and nominal recovery gain=2.5.
- **Spiking PD:** paired bipolar LIF ensembles encode error, fixed NEF transforms produce proportional drive, short/long synaptic traces produce a derivative estimate, and output ensembles decode requested joint velocity. The reference has 250 neurons per ensemble and a 1 ms neural step within each 20 ms control interval. The integral state has zero output influence.

The CLI name `spiking-pid` is historical; it does not imply a validated integral controller. Mode resets, hold behavior, firing activity, deterministic replay, derivative response, joint independence, and frozen gains are tested on Lava CPU. The conventional backend is the explicit baseline, not a hidden actuator fallback inside the spiking forward path.

Final actuator constraints are shared. No model of torque, inertia, actuator lag, force sensing, or a safety-rated control loop is included. `export_spiking_topology.py` remains in the private historical archive rather than the public executable package: hardware export is outside this release's validated scope.
