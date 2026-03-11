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

## Actuation Subsystem Spec (v0.1)

```yaml
subsystem: actuation
version: 0.1
status: specified
dependencies:
  - structure
  - power
  - compute

constraints:
  mass_budget_kg: 25.0      # total motor mass (39 joints)
  power_budget_w: 1800      # peak full-body active; 1245W normal walking
  cost_usd: 35000           # estimated BOM for 39 QDD actuators

topology: quasi_direct_drive  # QDD with proprioceptive torque sensing
communication: EtherCAT       # deterministic 1kHz update, ring topology
encoder_resolution_bits: 17   # minimum absolute encoder per joint

motor_classes:
  S:
    peak_torque_nm: 250
    continuous_torque_nm: 180
    peak_power_w: 1200
    mass_kg: 1.8
    example: CubeMars AK80-9
    joints: [knee_pitch]
  A:
    peak_torque_nm: 150
    continuous_torque_nm: 100
    peak_power_w: 800
    mass_kg: 1.2
    example: T-Motor AK80-8
    joints: [hip_pitch, hip_roll, ankle_pitch]
  B:
    peak_torque_nm: 80
    continuous_torque_nm: 55
    peak_power_w: 400
    mass_kg: 0.7
    example: T-Motor AK60-6
    joints: [hip_yaw, ankle_roll, elbow_pitch, shoulder_pitch]
  C:
    peak_torque_nm: 40
    continuous_torque_nm: 28
    peak_power_w: 200
    mass_kg: 0.4
    example: CubeMars AK10-9
    joints: [shoulder_roll, shoulder_yaw, elbow_yaw, wrist_pitch, wrist_roll, hands, spine, neck]

degrees_of_freedom:
  legs: 12        # 6 DOF x 2 (hip3 + knee1 + ankle2)
  arms: 14        # 7 DOF x 2 (shoulder3 + elbow2 + wrist2)
  hands: 8        # 4 DOF x 2 (simplified)
  spine: 2        # lumbar pitch + roll
  neck_head: 3    # pitch + roll + yaw
  total: 39

torque_budget_peak_nm:
  knee_pitch: 220
  hip_pitch: 130
  ankle_pitch: 112
  hip_roll: 95
  ankle_roll: 55
  hip_yaw: 45
  elbow_pitch: 55
  shoulder_pitch: 45
  shoulder_roll: 35
  wrist: 15

performance_targets:
  torque_control_bandwidth_hz: 200
  ethercat_cycle_ms: 1.0
  feedback_latency_ms: 0.5
  thermal_limit_deg_c: 80
  peak_operation_window_s: 45   # before thermal throttle at 100% torque

open_challenges:
  - Thermal management at sustained high-torque maneuvers (>45s)
  - Cycloidal QDD backlash characterization under dynamic loads
  - EtherCAT ring recovery time after single-node fault
  - Hand DOF simplification vs manipulation task coverage

references:
  - arXiv:2410.16591   # Cycloidal QDD actuators
  - arXiv:2401.09233   # Boston Dynamics electric Atlas
  - arXiv:2503.01171   # SEA bipedal locomotion benchmark
  - arXiv:2408.01056   # NING Humanoid concurrent design
  - doi:10.3390/act14050243  # MDPI humanoid actuator design 2025
  - doi:10.5281/zenodo.18959349  # This article

article_ref: "Article 4 — Actuation"
doi: "10.5281/zenodo.18959349"
```

## Structural Subsystem Spec (v0.1)

```yaml
subsystem: structure
version: 0.1
status: specified
dependencies:
  - actuation
  - power
  - compute
  - sensing

constraints:
  mass_budget_kg: 18.0          # structural members only; housings ~8kg separate
  ip_rating: IP54               # labyrinth + NBR/FKM lip seals at all 39 joints
  operating_temperature_c: 0-40
  safety_factor_min: 2.5        # on all primary structural members

material_zones:
  limb_links:
    material: CFRP
    spec: T700 carbon fiber, epoxy matrix, 3mm wall tube
    ods_mm: [30, 35, 40, 50]    # forearm, upper-arm, shin, thigh
    yield_strength_mpa: 600     # fiber direction
  joint_housings:
    material: 7075-T6 Aluminum
    yield_strength_mpa: 503
    density_g_cm3: 2.81
    finish: anodized
  hip_knee_ankle_brackets:
    material: Ti-6Al-4V
    yield_strength_mpa: 880
    density_g_cm3: 4.43
    fatigue_life: ">1e7 cycles at design load"
  outer_skin:
    material: TPU
    durometer_shore_a:
      torso_arms: 65
      hands_palms: 50
      ankle_foot: 88
    thickness_mm: 8
    standard: ISO/TS 15066 compliant (contact force attenuation 40-60%)

load_cases:
  static_standing:
    load_per_leg_n: 400
    governing_stress_mpa: 1.36   # thigh CFRP link
  walking_heel_strike:
    peak_grf_n: 2350             # 3x body weight
    ankle_bracket_stress_mpa: 23.5
    safety_factor: 37.4
  worst_case_fall:
    fall_height_m: 0.9
    impact_velocity_m_s: 4.2
    peak_force_n: 67200          # delta_t = 5ms rigid
    ankle_bracket_stress_mpa: 672
    safety_factor: 1.31          # mitigated by TPU sole + fall trajectories

bearings:
  hip_knee: crossed_roller       # combined load capacity
  wrist_shoulder: angular_contact_pair  # back-to-back, preload 5-10% Cdyn
  seals: NBR default, FKM/Viton ankle+foot

bom_mass_summary_kg:
  cfrp_links: 1.84
  ti_brackets: 2.22
  al_frames_housings: 8.15
  tpu_shell: 0.80
  bearings_39x: 4.68
  fasteners_inserts: 0.80
  misc_wiring_thermal: 0.62
  total: 18.2

open_challenges:
  - CFRP tube-to-metal insert bonding fatigue under cyclic loading
  - TPU skin abrasion life quantification (floor contact cycles)
  - Fall-arrest trajectory integration with structural safety factor targets
  - IP54 seal friction contribution to joint torque budget (to be measured)
  - Additive vs machined Ti for complex bracket geometries (cost vs mass)

references:
  - arXiv:2504.17249   # Berkeley Humanoid Lite CFRP structure
  - arXiv:2602.03177   # Ground reaction force estimation (2026)
  - arXiv:2602.17822   # ISO 10218/15066 evolution (2026)
  - arXiv:2409.06369   # Adaptive robot skin ISO 15066
  - arXiv:2502.01256   # Soft-is-safe CoboSkin
  - ISO/TS 15066:2016  # Collaborative robot contact force limits
  - doi:10.5281/zenodo.18960243  # This article

article_ref: "Article 5 — Structure"
doi: "10.5281/zenodo.18960243"
```
