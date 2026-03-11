# Open Humanoid — Master Specification Schema v0.1

This schema is referenced by every article in the series. Each subsystem spec follows this format.

## Subsystem Spec Template
```yaml
subsystem: [name]
version: 0.1
status: research | specified | validated | simulated
dependencies: [list of other subsystems]
constraints:
  mass_budget_kg: [value]
  power_budget_w: [value]
  volume_mm: [L x W x H]
  cost_usd: [target BOM cost]
performance_targets:
  [metric]: [value with unit]
open_challenges:
  - [challenge 1]
  - [challenge 2]
references:
  - [DOI or arXiv]
```

## Global Robot Constraints
- Total mass: ≤ 80kg
- Height: 160–180cm
- Battery life: > 60 minutes under normal operation
- Operating temperature: 0–40°C
- IP rating: IP54 minimum
- Emergency stop: hardware + software, <100ms response
- Communication: onboard WiFi6 + Bluetooth 5.2, optional 5G

## Locomotion Subsystem Spec (v0.1)

```yaml
subsystem: locomotion
version: 0.1
status: specified
dependencies:
  - actuation
  - sensing
  - compute
  - structure
  - power

constraints:
  mass_budget_kg: 16.0
  power_budget_w: 800        # peak; 350W average normal walking
  cost_usd: 8000

performance_targets:
  gait_speed_normal_ms: 1.2
  gait_speed_fast_ms: 2.5
  step_frequency_hz: 1.8
  balance_recovery_ms: 300   # from 15-degree tilt
  zmp_margin_mm: 20
  controller_loop_hz: 1000
  step_length_m: 0.65
  lateral_deviation_m: 0.12
  fall_detection_ms: 50

degrees_of_freedom:
  per_leg: 6
  hip: 3    # flexion/extension, abduction/adduction, rotation
  knee: 1   # flexion/extension
  ankle: 2  # dorsiflexion/plantarflexion, inversion/eversion
  total_leg_dof: 12

controller:
  architecture: hybrid_mpc_rl
  mpc_horizon_steps: 50
  mpc_dt_ms: 20
  rl_residual: true
  state_estimator: extended_kalman_filter

article_ref: "Article 3 — Bipedal Locomotion"
doi: "10.5281/zenodo.18956673"
```
