# How Long Can a Humanoid Robot Hold a Box?

Joint torque, motor heat and battery limits on five open humanoid models (Unitree G1, Unitree H1,
Booster T1, Apptronik Apollo, Fourier N1).

## Question
1. How much of each joint's torque limit does a humanoid use when it stands, squats or stands on one leg
   while holding a 0-20 kg box, and which joint is the bottleneck?
2. How long can the bottleneck motor hold that load before its winding reaches the thermal limit, and what
   payload can be held indefinitely?
3. For the Unitree G1, does the battery or motor heat limit holding time?

## Method
- **Statics.** MJCF models from MuJoCo Menagerie (commit `98a70c569c4c1b6da467a20012f766ee1f8dc521`, copied to `data/menagerie/` with their
  licences; meshes are not needed because every body has explicit inertia). Each posture is solved so that the
  feet are flat and the whole-body centre of mass (robot plus box) lies over the support centre (or, in the
  `com_over_ankle` variant, over the ankle joints). The arms are placed by inverse kinematics on the grip point.
  Joint torques follow from static equilibrium, `tau = g(q) - Jc^T w - Jp^T F`, with the foot wrenches `w`
  as the minimum-norm solution that balances the floating base and the box weight `F` split between the hands.
  Utilisation = |tau| / torque limit from the MJCF (actuator `forcerange`, or `ctrlrange x gear` for H1 motors).
- **Thermal.** A two-node (winding, housing) thermal model with the parameters measured by Urs et al. on a
  quasi-direct-drive actuator (arXiv:2202.12395, fans on). Heating is normalised so that the continuous torque
  gives a steady-state winding rise equal to the allowed rise; continuous torque = 1/3 of the MJCF limit
  (Unitree H1-2 rated/peak arm payload, 7/21 kg), swept 0.25-0.40; thermal capacitance swept x0.5-x4.
  The model is checked against the published 20 s and 2 s peak currents of the same actuator.
- **Battery (G1).** Power = 135 W measured rest power + sum of b_j tau_j^2, with the arm copper-loss coefficients
  b_j measured on a G1 by Deniz et al. (arXiv:2606.15915); other joints are scaled to equal copper loss at their
  torque limit (our assumption). Energy = 9 Ah x 13 cells x 3.6 V (nominal cell voltage is our assumption).

## Sources
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- Urs et al., Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion, arXiv:2202.12395
- Deniz et al., Identification of a Physics-Based Electrical Power Consumption Model for the Unitree G1 Humanoid Arm, arXiv:2606.15915
- Unitree G1 and H1 product pages: https://www.unitree.com/g1 , https://www.unitree.com/h1
- CubeMars AK80-64 product page (rated 48 N m, peak 120 N m)

## Run
```
pip install mujoco numpy matplotlib
python3 analysis.py
```
CPU only, about two minutes. Writes `results.json` and `charts/chart1.png` ... `chart4.png`.

## Outputs
- `results.json`: per robot and posture: bottleneck joint, torque, utilisation, time to thermal limit
  (for each continuous ratio and capacitance scale), feasibility flags (joint ranges, inverse-kinematics error),
  payload caps; thermal-model validation; G1 battery budget.
- chart1: bottleneck utilisation vs box mass. chart2: time to thermal limit vs box mass.
  chart3: thermal model and validation. chart4: G1 battery-limited vs thermal-limited holding time.

## Limitations
Static postures only; MJCF torque limits treated as peak limits; one thermal parameter set scaled for all
actuators; cold start; no thermal coupling between joints; no friction or gearbox losses in the battery budget.
