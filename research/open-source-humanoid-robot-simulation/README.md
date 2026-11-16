# Open Source Humanoid Robots Compared in Simulation

Seven open humanoid models from MuJoCo Menagerie, compared side by side on one CPU: Unitree G1, Unitree H1,
Booster T1, Fourier N1, Apptronik Apollo, PAL TALOS and the Berkeley Humanoid.

## Questions
1. What do the open model files actually release, and how do the robots compare on DoF, mass and torque-to-weight
   when everything is read from the MJCF files themselves?
2. How fast does each model simulate on one CPU core, with identical settings and with the settings shipped in Menagerie?
3. Under one identical joint PD posture hold, does each model stand for 10 s, and what is the smallest horizontal push
   that topples it? What predicts that number?

## Method
- **Models.** MJCF files, licences and READMEs copied from MuJoCo Menagerie (commit in `data/menagerie/COMMIT`).
  Mesh assets are not included: every body has explicit inertia, so masses are unaffected. Mesh geoms are removed.
  The Fourier N1 uses mesh collision for its feet; those geoms are replaced by boxes fitted to the bounding box of the
  foot meshes (constant `N1_FOOT_BOX`). Mesh file counts and sizes come from the GitHub git-trees API at the same
  commit (`data/menagerie_file_sizes.json`).
- **Spec sheet.** DoF = generalised velocities minus the 6 of the floating base; torque limits from the actuator
  `forcerange` (or `ctrlrange x gear` for plain motors); the TALOS uses `talos_motor.xml`. Licence from each `LICENSE`.
  Height is the highest body frame in the standing pose (heads are meshes, so this is a lower bound).
- **Uniform simulation settings.** 2 ms step, implicitfast integrator, Newton solver (100 iterations), pyramidal cone,
  robot-floor contacts only (self-collision off), geom-level friction and solref as shipped.
- **Identical PD controller.** Every actuated joint gets a position servo with `kp = tau_max / 0.05 rad`,
  `kv = 0.03 s x kp`, torque clipped at `tau_max`. The posture is the Menagerie keyframe with feet flat and the ankles
  offset so that the whole-body CoM is over the centre of the support area. Sensitivity: `theta_sat` = 0.025, 0.05,
  0.1, 0.2 rad.
- **Standing.** 10 s; fall = whole-body CoM below 70% of its initial height.
- **Push test.** 1 s settle, then a 0.1 s horizontal force at the root body (pelvis or trunk), forward, backward and
  lateral; 3.9 s observation. Bisection (12 iterations, range 0-3 m/s) on the velocity change `dv = F x 0.1 s / m`.
  Monotonicity is spot-checked at 0.8x to 4x the threshold.
- **Capture-point bound.** `dv_bound = (distance from CoM to support edge) x sqrt(g / CoM height)`, the velocity a linear
  inverted pendulum can absorb with the centre of pressure at the edge of the feet.
- **Speed.** 5 x 5000 steps from the standing state, median steps/s; real-time factor = steps/s x timestep. Single thread.

## Run
```
pip install mujoco numpy matplotlib
python3 analysis.py              # about 75 s on one core
python3 analysis.py --charts-only  # redraw charts from results.json
```

## Outputs
- `results.json`: run metadata (CPU, MuJoCo version, settings); per robot: spec sheet, standing calibration, speed
  (uniform and native), 10 s stand test, push thresholds and capture-point bounds per direction, gain sensitivity;
  correlations and rankings; every chart value under `chart_values`.
- chart1: mass, DoF and torque-to-weight. chart2: real-time factor on one CPU core. chart3: toppling push by direction.
  chart4: gain sensitivity and measured push vs capture-point bound.

## Sources
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- MuJoCo: Todorov, Erez and Tassa, "MuJoCo: A physics engine for model-based control", IROS 2012, doi:10.1109/IROS.2012.6386109
- Capture point: Pratt et al., "Capture Point: A Step toward Humanoid Push Recovery", Humanoids 2006, doi:10.1109/ICHR.2006.321385

## Limitations
The PD posture hold is not any vendor's controller: it has no balance feedback, no stepping and no whole-body
coordination, so the push numbers measure passive-ish robustness under identical PD, not what the real robots can
recover from. MJCF torque limits are what the simulator enforces, not certified actuator ratings. Collision geometry
is the primitive geometry in the Menagerie files (plus the fitted N1 foot boxes). Speed figures vary by about 10%
between runs on a shared virtual machine.
