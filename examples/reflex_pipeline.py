"""One reproducible synthetic reflex encounter, without a sensor or Lava runtime."""
from run_dynamic_cases import run_dynamic_cases
from threat_gate import ThreatGateConfig

result = run_dynamic_cases(num_trials=1, seed=42, speed_min=1.2, speed_max=1.2,
    trajectory_types=("collision_course_center",),
    threat_gate=ThreatGateConfig("geometric_hysteresis"), post_encounter_time_s=0.5)[0]
print(f"contact={result.metrics.contact}, clearance={result.metrics.min_clearance_m:.6f} m")
