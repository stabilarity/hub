#!/usr/bin/env python3
"""Open source humanoid robots compared: seven open MuJoCo Menagerie models in simulation.

1. Spec sheet from the MJCF files: DoF, actuators, mass, height, torque limits, licence, file sizes.
2. Simulation speed on this CPU: steps/s and real-time factor, uniform settings and native settings.
3. Standing: an identical joint PD posture hold for 10 s.
4. Push test: the smallest 0.1 s horizontal push at the root body that topples each model (bisection).

Run:   python3 analysis.py        (CPU only, a few minutes)
Needs: mujoco>=3.2, numpy, matplotlib.
Writes results.json and charts/chart1.png ... chart4.png next to this file.

Robot files are copied from MuJoCo Menagerie (commit in data/menagerie/COMMIT) without their mesh
assets. Every body has explicit inertia, so masses do not depend on the meshes; mesh geoms are
removed. The Fourier N1 uses mesh geoms for its foot collision, so they are replaced by boxes fitted
to the bounding box of the two foot meshes (see N1_FOOT_BOX).
"""
import json
import math
import os
import platform
import re
import statistics
import time

import numpy as np
import mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "menagerie")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)
G = 9.81

# ---------------------------------------------------------------------------
# Robots: scene file, model file(s), knee and ankle-pitch joints
# ---------------------------------------------------------------------------
ROBOTS = {
    "Unitree G1": dict(dir="unitree_g1", scene="scene.xml", model="g1.xml",
                       knee="left_knee_joint", ankle=("left_ankle_pitch_joint", "right_ankle_pitch_joint")),
    "Unitree H1": dict(dir="unitree_h1", scene="scene.xml", model="h1.xml",
                       knee="left_knee", ankle=("left_ankle", "right_ankle")),
    "Booster T1": dict(dir="booster_t1", scene="scene.xml", model="t1.xml",
                       knee="Left_Knee_Pitch", ankle=("Left_Ankle_Pitch", "Right_Ankle_Pitch")),
    "Fourier N1": dict(dir="fourier_n1", scene="scene.xml", model="n1.xml",
                       knee="left_knee_pitch_joint", ankle=("left_ankle_pitch_joint", "right_ankle_pitch_joint")),
    "Apptronik Apollo": dict(dir="apptronik_apollo", scene="scene.xml", model="apptronik_apollo.xml",
                             knee="l_knee_fe", ankle=("l_ankle_pd", "r_ankle_pd")),
    "PAL TALOS": dict(dir="pal_talos", scene="scene_motor.xml", model="talos_motor.xml",
                      knee="leg_left_4_joint", ankle=("leg_left_5_joint", "leg_right_5_joint")),
    "Berkeley Humanoid": dict(dir="berkeley_humanoid", scene="scene.xml", model="berkeley_humanoid.xml",
                              knee="LL_KFE", ankle=("LL_FFE", "LR_FFE")),
}
# Bounding box of fourier_n1/assets/{left,right}_foot_pitch_link.stl in the foot frame
# (x -0.0735..0.1405, y -0.0385..0.0385, z -0.0445..0.0115; right foot within 0.3 mm).
N1_FOOT_BOX = dict(size=[0.107, 0.0385, 0.028], pos=[0.0335, 0.0, -0.0165])

# Uniform simulation settings used for speed, standing and push tests
TIMESTEP = 0.002
THETA_SAT = 0.05        # rad: PD gain rule kp = tau_max / THETA_SAT (saturates at 0.05 rad error)
KD_RATIO = 0.03         # s:   kv = KD_RATIO * kp
THETA_SWEEP = [0.025, 0.05, 0.10, 0.20]
STAND_T = 10.0
SETTLE_T = 1.0
PUSH_T = 0.1
OBSERVE_T = 3.9
FALL_RATIO = 0.7        # fall = whole-body CoM height below 70% of its initial height
DV_MAX = 3.0            # m/s, upper end of the bisection range (impulse per kg)
BISECT_ITERS = 12
SPEED_STEPS = 5000
SPEED_REPEATS = 5
DIRECTIONS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "lateral": (0.0, 1.0)}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def actuator_limits(m):
    """Per-joint torque limit [N m] from the MJCF actuators."""
    lim = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = m.actuator_trnid[a, 0]
        gear = abs(m.actuator_gear[a, 0])
        c = []
        if m.actuator_forcelimited[a]:
            c.append(max(abs(m.actuator_forcerange[a])) * gear)
        elif m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE and m.actuator_ctrllimited[a]:
            c.append(max(abs(m.actuator_ctrlrange[a])) * gear * abs(m.actuator_gainprm[a, 0]))
        if m.jnt_actfrclimited[j]:
            c.append(max(abs(m.jnt_actfrcrange[j])))
        if c:
            lim[m.joint(j).name] = float(min(c))
    return lim


def strip_meshes(spec, robot_dir):
    """Remove mesh geoms and file assets; replace N1 foot mesh collision with boxes."""
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            if robot_dir == "fourier_n1" and (g.contype or g.conaffinity):
                ng = g.parent.add_geom()
                ng.type = mujoco.mjtGeom.mjGEOM_BOX
                ng.size = N1_FOOT_BOX["size"]
                ng.pos = N1_FOOT_BOX["pos"]
                for a in ("contype", "conaffinity", "condim", "friction", "solref", "solimp", "group",
                          "margin", "priority"):
                    setattr(ng, a, getattr(g, a))
                ng.mass = 0.0
                ng.density = 0.0
            spec.delete(g)
    for x in list(spec.meshes):
        spec.delete(x)
    for t in list(spec.textures):
        if t.file:
            spec.delete(t)


def explicit_inertia(spec):
    return all(b.explicitinertial or b.mass == 0 for b in spec.bodies if b.name != "world")


def build(cfg, theta_sat=THETA_SAT, uniform=True):
    """uniform=True: identical PD position actuators (kp = tau_max/theta_sat), robot-floor contacts only,
    timestep 2 ms, implicitfast, Newton solver defaults. uniform=False: the Menagerie scene as shipped."""
    spec = mujoco.MjSpec.from_file(os.path.join(DATA, cfg["dir"], cfg["scene"]))
    strip_meshes(spec, cfg["dir"])
    info = {"explicit_inertia": explicit_inertia(spec)}
    m0 = spec.compile()
    lim = actuator_limits(m0)
    if not uniform:
        return m0, lim, info
    for g in spec.geoms:
        if g.name == "floor":
            g.contype, g.conaffinity = 1, 1
        elif g.contype or g.conaffinity:
            g.contype, g.conaffinity = 0, 1   # collides with the floor only
    for p in list(spec.pairs):
        if "floor" not in (p.geomname1, p.geomname2):
            spec.delete(p)
    for p in list(spec.excludes):
        spec.delete(p)
    for a in list(spec.actuators):
        spec.delete(a)
    for jn, t in lim.items():
        kp = t / theta_sat
        a = spec.add_actuator()
        a.name = jn + "_pd"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.target = jn
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0], a.biasprm[1], a.biasprm[2] = 0.0, -kp, -kp * KD_RATIO
        a.forcelimited = 1
        a.forcerange = [-t, t]
        a.ctrllimited = 0
    o = spec.option
    o.timestep = TIMESTEP
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.iterations = 100
    o.ls_iterations = 50
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m = spec.compile()
    return m, lim, info


# ---------------------------------------------------------------------------
# Standing pose: Menagerie keyframe, ankles calibrated so the CoM is over the support centre
# ---------------------------------------------------------------------------
def floor_id(m):
    return [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE][0]


def floor_geoms(m):
    fl = floor_id(m)
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        col = (m.geom_contype[g] & m.geom_conaffinity[fl]) or (m.geom_conaffinity[g] & m.geom_contype[fl])
        pair = np.any(((m.pair_geom1 == g) & (m.pair_geom2 == fl)) | ((m.pair_geom2 == g) & (m.pair_geom1 == fl)))
        if col or pair:
            out.append(g)
    return fl, out


def ground_fit(m, d, pen=0.0005):
    fl, gs = floor_geoms(m)
    mujoco.mj_forward(m, d)
    dmin = min(mujoco.mj_geomDistance(m, d, g, fl, 10.0, None) for g in gs)
    d.qpos[2] -= dmin + pen
    mujoco.mj_forward(m, d)


def support_centre(m, d):
    """Centre of the floor-contact points (contacts within 5 mm) in x and y."""
    saved, savedp = m.geom_margin.copy(), m.pair_margin.copy()
    m.geom_margin[:] = 0.01
    m.pair_margin[:] = 0.01
    mujoco.mj_forward(m, d)
    pts = np.array([d.contact[i].pos for i in range(d.ncon) if d.contact[i].dist < 0.005])
    m.geom_margin[:] = saved
    m.pair_margin[:] = savedp
    mujoco.mj_forward(m, d)
    return 0.5 * (pts[:, 0].min() + pts[:, 0].max()), 0.5 * (pts[:, 1].min() + pts[:, 1].max()), pts


def set_pose(m, d, key_qpos, ank_ids, delta, R0, p0, foot_body):
    d.qpos[:] = key_qpos
    d.qpos[3:7] = [1, 0, 0, 0]
    for j in ank_ids:
        d.qpos[m.jnt_qposadr[j]] += delta
    mujoco.mj_forward(m, d)
    Rf = d.xmat[foot_body].reshape(3, 3)
    Rb = R0 @ Rf.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rb.flatten())
    d.qpos[3:7] = q
    mujoco.mj_forward(m, d)
    d.qpos[0:3] += p0 - d.xpos[foot_body]
    mujoco.mj_forward(m, d)


def foot_target(m, d, side_sign=1.0):
    """Distal foot body of one side and its flat-sole orientation.

    The zero joint configuration defines the foot orientation; if the foot touches the floor with a box,
    the box's thinnest axis is additionally aligned with the vertical (sole parallel to the floor)."""
    fl, gs = floor_geoms(m)
    d.qpos[:] = m.qpos0
    d.qpos[2] = 2.0
    mujoco.mj_forward(m, d)
    side = [g for g in gs if side_sign * d.geom_xpos[g][1] > 0]
    g_low = min(side, key=lambda g: d.geom_xpos[g][2])
    body = int(m.geom_bodyid[g_low])
    Rb0 = d.xmat[body].reshape(3, 3).copy()
    boxes = [g for g in side if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if boxes:
        g = boxes[0]
        Rg = d.geom_xmat[g].reshape(3, 3)
        v = Rg[:, int(np.argmin(m.geom_size[g]))].copy()
        v *= np.sign(v[2]) if v[2] != 0 else 1.0
        z = np.array([0.0, 0.0, 1.0])
        ax = np.cross(v, z)
        s_, c_ = np.linalg.norm(ax), float(np.dot(v, z))
        if s_ > 1e-9:
            ax /= s_
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            C = np.eye(3) + s_ * K + (1 - c_) * K @ K
            Rb0 = C @ Rb0
    return body, Rb0


def standing_state(m, cfg):
    d = mujoco.MjData(m)
    foot_body, R0 = foot_target(m, d)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    key_qpos = d.qpos.copy()
    ank_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg["ankle"]]
    p0 = np.zeros(3)
    set_pose(m, d, key_qpos, ank_ids, 0.0, R0, p0, foot_body)
    ground_fit(m, d)
    sx, sy, pts = support_centre(m, d)
    p0 = d.xpos[foot_body].copy()

    def f(delta):
        set_pose(m, d, key_qpos, ank_ids, delta, R0, p0, foot_body)
        return d.subtree_com[1][0] - sx

    grid = np.round(np.arange(-0.30, 0.3001, 0.01), 4)
    vals = [f(x) for x in grid]
    best = float(grid[int(np.argmin(np.abs(vals)))])
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            flo = vals[i]
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if fm * flo <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            best = 0.5 * (lo + hi)
            break
    off0 = f(0.0)
    set_pose(m, d, key_qpos, ank_ids, best, R0, p0, foot_body)
    ground_fit(m, d)
    sx2, sy2, pts2 = support_centre(m, d)
    d.qvel[:] = 0
    qref = np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)])
    d.ctrl[:] = qref
    mujoco.mj_forward(m, d)
    in_range = all((not m.jnt_limited[j]) or
                   (m.jnt_range[j, 0] - 1e-6 <= d.qpos[m.jnt_qposadr[j]] <= m.jnt_range[j, 1] + 1e-6)
                   for j in ank_ids)
    calib = dict(ankle_offset_rad=round(float(best), 4), com_minus_support_x_before_m=round(float(off0), 4),
                 com_minus_support_x_after_m=round(float(d.subtree_com[1][0] - sx2), 4),
                 com_minus_support_y_after_m=round(float(d.subtree_com[1][1] - sy2), 4),
                 support_contact_points=int(len(pts2)),
                 margin_forward_m=round(float(pts2[:, 0].max() - d.subtree_com[1][0]), 4),
                 margin_backward_m=round(float(d.subtree_com[1][0] - pts2[:, 0].min()), 4),
                 margin_lateral_m=round(float(pts2[:, 1].max() - d.subtree_com[1][1]), 4),
                 support_x_extent_m=round(float(pts2[:, 0].max() - pts2[:, 0].min()), 4),
                 support_y_extent_m=round(float(pts2[:, 1].max() - pts2[:, 1].min()), 4),
                 ankle_within_range=bool(in_range), foot_body=m.body(foot_body).name)
    return d, qref, calib


def snapshot(m, d):
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), act=d.act.copy())


def restore(m, d, s):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = s["qpos"]
    d.qvel[:] = s["qvel"]
    d.ctrl[:] = s["ctrl"]
    mujoco.mj_forward(m, d)


def tilt_deg(d, z0):
    z = d.xmat[1].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, z0))))))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def stand_test(m, d, s0, lim_arr):
    restore(m, d, s0)
    com0 = d.subtree_com[1][2]
    x0 = d.subtree_com[1][:2].copy()
    z0 = d.xmat[1].reshape(3, 3)[:, 2].copy()
    n = int(round(STAND_T / m.opt.timestep))
    min_ratio, max_tilt, util = 1.0, 0.0, []
    fell_at = None
    for i in range(n):
        mujoco.mj_step(m, d)
        r = d.subtree_com[1][2] / com0
        min_ratio = min(min_ratio, r)
        max_tilt = max(max_tilt, tilt_deg(d, z0))
        if fell_at is None and (r < FALL_RATIO or not np.isfinite(r)):
            fell_at = (i + 1) * m.opt.timestep
        if i >= n // 2:
            util.append(np.max(np.abs(d.actuator_force) / lim_arr))
    drift = float(np.linalg.norm(d.subtree_com[1][:2] - x0))
    return dict(stands_10s=fell_at is None, fell_at_s=fell_at, min_com_height_ratio=round(float(min_ratio), 4),
                max_root_tilt_deg=round(max_tilt, 2), com_drift_m=round(drift, 4),
                mean_peak_joint_utilisation_last5s=round(float(np.mean(util)), 4),
                max_joint_utilisation_last5s=round(float(np.max(util)), 4))


def push_falls(m, d, s0, direction, dv, mass):
    restore(m, d, s0)
    com0 = d.subtree_com[1][2]
    dt = m.opt.timestep
    n_settle, n_push, n_obs = int(round(SETTLE_T / dt)), int(round(PUSH_T / dt)), int(round(OBSERVE_T / dt))
    F = dv * mass / PUSH_T
    for i in range(n_settle + n_push + n_obs):
        if n_settle <= i < n_settle + n_push:
            d.xfrc_applied[1, 0] = F * direction[0]
            d.xfrc_applied[1, 1] = F * direction[1]
        else:
            d.xfrc_applied[1, :] = 0.0
        mujoco.mj_step(m, d)
        z = d.subtree_com[1][2]
        if not np.isfinite(z) or z < FALL_RATIO * com0:
            return True
    return False


def push_threshold(m, d, s0, direction, mass):
    """Smallest velocity change (impulse per kg, m/s) of a 0.1 s push that topples the model."""
    if push_falls(m, d, s0, direction, 0.0, mass):
        return 0.0, "falls without push"
    if not push_falls(m, d, s0, direction, DV_MAX, mass):
        return DV_MAX, "survives the largest push tested"
    lo, hi = 0.0, DV_MAX
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if push_falls(m, d, s0, direction, mid, mass):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), "bisection"


def speed_test(m, d, s0, steps=SPEED_STEPS, repeats=SPEED_REPEATS):
    rates, ncon, nefc = [], [], []
    for _ in range(repeats):
        restore(m, d, s0)
        t0 = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(m, d)
        rates.append(steps / (time.perf_counter() - t0))
    restore(m, d, s0)
    for _ in range(500):
        mujoco.mj_step(m, d)
        ncon.append(d.ncon)
        nefc.append(d.nefc)
    sps = statistics.median(rates)
    return dict(steps_per_s=round(sps, 1), real_time_factor=round(sps * m.opt.timestep, 2),
                timestep_s=float(m.opt.timestep), mean_contacts=round(float(np.mean(ncon)), 2),
                mean_constraint_rows=round(float(np.mean(nefc)), 2), runs=[round(r, 1) for r in rates])


# ---------------------------------------------------------------------------
# Spec sheet
# ---------------------------------------------------------------------------
def licence_info(robot_dir):
    txt = open(os.path.join(DATA, robot_dir, "LICENSE")).read()
    if "Apache License" in txt:
        lic = "Apache-2.0"
    elif "Redistribution and use in source and binary forms" in txt and "Neither the name" in txt:
        lic = "BSD-3-Clause"
    else:
        lic = "other"
    holder = ""
    for line in txt.splitlines():
        if re.match(r"\s*Copyright (\(c\)|[0-9])", line):
            holder = line.strip()
            break
    readme = open(os.path.join(DATA, robot_dir, "README.md")).read()
    up = re.search(r"derived from the (?:publicly available )?\[([^\]]+)\]\(([^)]+)\)", readme.replace("\n", " "))
    up2 = re.findall(r"\((https://github\.com/[^)]+)\)", readme)
    if up:
        src = up.group(2)
    elif "provided\ndirectly by" in readme or "provided directly by" in readme.replace("\n", " "):
        src = "URDF and assets provided directly by the manufacturer (per Menagerie README)"
    else:
        src = "not stated in the Menagerie README"
    return lic, holder, src


def spec_sheet(name, cfg, sizes):
    m, lim, info = build(cfg, uniform=False)
    nv_j = m.nv - 6
    act_joints = set(lim.keys())
    hinge = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                                 int(mujoco.mjtJoint.mjJNT_SLIDE))]
    mass = float(np.sum(m.body_mass))
    taus = np.array(list(lim.values()))
    jmax = max(lim, key=lim.get)
    knee = lim[cfg["knee"]]
    lic, holder, upstream = licence_info(cfg["dir"])
    files = sizes[cfg["dir"]]
    assets = {p: s for p, s in files.items() if p.startswith("assets/")}
    meshes = {p: s for p, s in assets.items() if p.lower().endswith((".stl", ".obj", ".msh"))}
    xml_bytes = sum(s for p, s in files.items() if p.endswith(".xml"))
    return dict(
        dof=int(nv_j), actuated_dof=len(act_joints), passive_dof=int(len(hinge) - len(act_joints)),
        actuators_in_mjcf=int(m.nu), mass_kg=round(mass, 2),
        sum_torque_limit_Nm=round(float(taus.sum()), 1), max_torque_limit_Nm=round(float(taus.max()), 1),
        max_torque_joint=jmax, knee_torque_limit_Nm=round(knee, 1),
        torque_per_kg_Nm_per_kg=round(float(taus.sum()) / mass, 2),
        knee_torque_to_weight_m=round(knee / (mass * G), 4),
        ankle_pitch_torque_limit_Nm=round(lim[cfg["ankle"][0]], 1),
        ankle_torque_to_weight_m=round(lim[cfg["ankle"][0]] / (mass * G), 4),
        licence=lic, copyright=holder, upstream_description=upstream,
        mjcf_xml_bytes=int(xml_bytes), mesh_files=len(meshes), mesh_MB=round(sum(meshes.values()) / 1e6, 2),
        all_assets_MB=round(sum(assets.values()) / 1e6, 2),
        explicit_inertia=info["explicit_inertia"],
        native_timestep_s=float(m.opt.timestep), native_iterations=int(m.opt.iterations),
        native_integrator=int(m.opt.integrator),
    ), m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    t_start = time.time()
    sizes = json.load(open(os.path.join(HERE, "data", "menagerie_file_sizes.json")))["files"]
    res = dict(
        meta=dict(mujoco=mujoco.__version__, python=platform.python_version(), cpu=cpu_name(),
                  logical_cpus=os.cpu_count(), threads_used=1,
                  menagerie_commit=open(os.path.join(DATA, "COMMIT")).read().strip(),
                  settings=dict(timestep_s=TIMESTEP, integrator="implicitfast", solver="Newton", iterations=100,
                                ls_iterations=50, cone="pyramidal", contacts="robot-floor only, geom-level "
                                "friction/solref as shipped", pd_rule="kp = tau_max / theta_sat, kv = %.2f s * kp, "
                                "torque clipped at tau_max" % KD_RATIO, theta_sat_rad=THETA_SAT,
                                stand_s=STAND_T, settle_s=SETTLE_T, push_s=PUSH_T, observe_s=OBSERVE_T,
                                fall_rule="whole-body CoM height < %.1f x initial" % FALL_RATIO,
                                dv_max_m_per_s=DV_MAX, bisection_iterations=BISECT_ITERS,
                                speed_steps=SPEED_STEPS, speed_repeats=SPEED_REPEATS)),
        robots={})
    for name, cfg in ROBOTS.items():
        t0 = time.time()
        r = {}
        r["spec"], m_nat = spec_sheet(name, cfg, sizes)
        # native speed (scene as shipped, mesh-free; key ctrl if present)
        d_nat = mujoco.MjData(m_nat)
        mujoco.mj_resetDataKeyframe(m_nat, d_nat, 0)
        mujoco.mj_forward(m_nat, d_nat)
        r["speed_native"] = speed_test(m_nat, d_nat, snapshot(m_nat, d_nat))
        # uniform model
        m, lim, _ = build(cfg)
        d, qref, calib = standing_state(m, cfg)
        r["calibration"] = calib
        r["height"] = dict(com_height_m=round(float(d.subtree_com[1][2]), 4),
                           highest_body_frame_m=round(float(np.max(d.xpos[1:, 2])), 4),
                           highest_body=m.body(int(np.argmax(d.xpos[1:, 2])) + 1).name,
                           root_body=m.body(1).name)
        r["height"]["ankle_stiffness_to_gravity_ratio"] = round(
            2 * lim[cfg["ankle"][0]] / THETA_SAT / (r["spec"]["mass_kg"] * G * r["height"]["com_height_m"]), 3)
        s0 = snapshot(m, d)
        lim_arr = np.array([lim[m.joint(m.actuator_trnid[a, 0]).name] for a in range(m.nu)])
        r["speed_uniform"] = speed_test(m, d, s0)
        r["stand"] = stand_test(m, d, s0, lim_arr)
        mass = r["spec"]["mass_kg"]
        pushes = {}
        for dn, vec in DIRECTIONS.items():
            dv, how = push_threshold(m, d, s0, vec, mass)
            pushes[dn] = dict(dv_m_per_s=round(dv, 4), impulse_Ns=round(dv * mass, 1),
                              peak_force_N=round(dv * mass / PUSH_T, 1),
                              force_over_weight=round(dv / PUSH_T / G, 3), method=how)
        # capture-point bound: an inverted pendulum with the centre of pressure at the support edge
        # can absorb a CoM velocity of (edge distance) * sqrt(g / CoM height)
        omega = math.sqrt(G / r["height"]["com_height_m"])
        for dn in pushes:
            marg = r["calibration"]["margin_%s_m" % dn]
            pushes[dn]["capture_point_bound_m_per_s"] = round(marg * omega, 4)
            pushes[dn]["measured_over_bound"] = round(pushes[dn]["dv_m_per_s"] / (marg * omega), 3)
        # monotonicity spot check around the threshold (forward)
        thr = pushes["forward"]["dv_m_per_s"]
        pushes["forward"]["monotonic_check"] = {f"{k:g}x": push_falls(m, d, s0, DIRECTIONS["forward"], thr * k, mass)
                                                for k in (0.8, 0.95, 1.05, 1.5, 2.0, 4.0) if thr * k <= 4 * DV_MAX}
        r["push"] = pushes
        r["push_min_dv_m_per_s"] = round(min(p["dv_m_per_s"] for p in pushes.values()), 4)
        r["push_weakest_direction"] = min(pushes, key=lambda k: pushes[k]["dv_m_per_s"])
        sens = {}
        for th in THETA_SWEEP:
            if th == THETA_SAT:
                sens[str(th)] = dict(stands_10s=r["stand"]["stands_10s"], forward_dv_m_per_s=pushes["forward"]["dv_m_per_s"])
                continue
            m2, lim2, _ = build(cfg, theta_sat=th)
            d2, _, _ = standing_state(m2, cfg)
            s2 = snapshot(m2, d2)
            la2 = np.array([lim2[m2.joint(m2.actuator_trnid[a, 0]).name] for a in range(m2.nu)])
            st = stand_test(m2, d2, s2, la2)
            dv, how = push_threshold(m2, d2, s2, DIRECTIONS["forward"], mass)
            sens[str(th)] = dict(stands_10s=st["stands_10s"], forward_dv_m_per_s=round(dv, 4), method=how)
        r["gain_sensitivity_forward"] = sens
        r["wall_time_s"] = round(time.time() - t0, 1)
        res["robots"][name] = r
        print(f"{name}: stand={r['stand']['stands_10s']} push={ {k: v['dv_m_per_s'] for k, v in pushes.items()} } "
              f"sps={r['speed_uniform']['steps_per_s']} ({r['wall_time_s']} s)", flush=True)

    # derived rankings
    rob = res["robots"]
    res["rankings"] = dict(
        speed_uniform=sorted(rob, key=lambda n: -rob[n]["speed_uniform"]["steps_per_s"]),
        push_min_dv=sorted(rob, key=lambda n: -rob[n]["push_min_dv_m_per_s"]),
        torque_per_kg=sorted(rob, key=lambda n: -rob[n]["spec"]["torque_per_kg_Nm_per_kg"]),
    )
    sp = [rob[n]["speed_uniform"]["steps_per_s"] for n in rob]
    res["summary"] = dict(
        n_stand=sum(rob[n]["stand"]["stands_10s"] for n in rob),
        speed_ratio_fastest_to_slowest=round(max(sp) / min(sp), 2),
        licences={n: rob[n]["spec"]["licence"] for n in rob},
        corr_push_vs_torque_per_kg=round(float(np.corrcoef(
            [rob[n]["spec"]["torque_per_kg_Nm_per_kg"] for n in rob],
            [rob[n]["push_min_dv_m_per_s"] for n in rob])[0, 1]), 3),
        corr_push_vs_knee_ratio=round(float(np.corrcoef(
            [rob[n]["spec"]["knee_torque_to_weight_m"] for n in rob],
            [rob[n]["push_min_dv_m_per_s"] for n in rob])[0, 1]), 3),
        corr_push_vs_ankle_ratio=round(float(np.corrcoef(
            [rob[n]["spec"]["ankle_torque_to_weight_m"] for n in rob],
            [rob[n]["push_min_dv_m_per_s"] for n in rob])[0, 1]), 3),
        corr_push_vs_capture_point_bound=round(float(np.corrcoef(
            [rob[n]["push"][k]["capture_point_bound_m_per_s"] for n in rob for k in DIRECTIONS],
            [rob[n]["push"][k]["dv_m_per_s"] for n in rob for k in DIRECTIONS])[0, 1]), 3),
        measured_over_bound_range=[min(rob[n]["push"][k]["measured_over_bound"] for n in rob for k in DIRECTIONS),
                                   max(rob[n]["push"][k]["measured_over_bound"] for n in rob for k in DIRECTIONS)],
        corr_push_vs_mass=round(float(np.corrcoef(
            [rob[n]["spec"]["mass_kg"] for n in rob], [rob[n]["push_min_dv_m_per_s"] for n in rob])[0, 1]), 3),
        corr_speed_vs_constraint_rows=round(float(np.corrcoef(
            [rob[n]["speed_uniform"]["mean_constraint_rows"] for n in rob], sp)[0, 1]), 3),
        corr_speed_vs_dof=round(float(np.corrcoef(
            [rob[n]["spec"]["dof"] for n in rob], sp)[0, 1]), 3),
        total_wall_time_s=round(time.time() - t_start, 1),
    )
    res["chart_values"] = make_charts(res)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote results.json and charts in", res["summary"]["total_wall_time_s"], "s")


# ---------------------------------------------------------------------------
# Charts (site palette: greys, status colours only for status)
# ---------------------------------------------------------------------------
SRCLINE = "Source: our MuJoCo %s simulations of MuJoCo Menagerie models; stabilarity/hub research code"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def make_charts(res):
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    rob = res["robots"]
    names = list(rob)
    cv = {}

    # chart 1: mass vs summed torque limits (bubble = DoF), and torque per kg bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=110)
    ax = axes[0]
    xs = [rob[n]["spec"]["mass_kg"] for n in names]
    ys = [rob[n]["spec"]["sum_torque_limit_Nm"] for n in names]
    dof = [rob[n]["spec"]["dof"] for n in names]
    ax.scatter(xs, ys, s=[30 * k for k in dof], color="#555555", edgecolors="#000000", alpha=0.8)
    for n, x, y, k in zip(names, xs, ys, dof):
        ax.annotate(f"{n}\n{k} DoF", (x, y), textcoords="offset points", xytext=(12, -4), fontsize=11)
    ax.set_xlabel("Model mass (kg)")
    ax.set_ylabel("Sum of actuator torque limits (N·m)")
    ax.set_title("Mass vs total torque (marker area = degrees of freedom)")
    ax.set_xlim(0, max(xs) * 1.3)
    ax.set_ylim(0, max(ys) * 1.15)
    style(ax)
    ax = axes[1]
    order = sorted(names, key=lambda n: rob[n]["spec"]["torque_per_kg_Nm_per_kg"])
    vals = [rob[n]["spec"]["torque_per_kg_Nm_per_kg"] for n in order]
    kn = [rob[n]["spec"]["knee_torque_to_weight_m"] * 100 for n in order]
    yi = np.arange(len(order))
    ax.barh(yi + 0.2, vals, 0.4, color="#000000", label="sum of torque limits per kg (N·m/kg)")
    ax.barh(yi - 0.2, kn, 0.4, color="#bbbbbb", label="knee limit / body weight (cm of lever arm)")
    for i, (v, k) in enumerate(zip(vals, kn)):
        ax.text(v + 0.3, i + 0.2, f"{v:.2f}", va="center", fontsize=10)
        ax.text(k + 0.3, i - 0.2, f"{k:.1f}", va="center", fontsize=10, color="#555555")
    ax.set_yticks(yi)
    ax.set_yticklabels(order)
    ax.set_xlabel("Value (units in legend)")
    ax.set_title("Torque-to-weight")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.suptitle("Chart 1. Humanoid robot dynamics from the MJCF files: mass, DoF and torque limits",
                 fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Source: MJCF files of MuJoCo Menagerie (commit %s); our extraction"
             % res["meta"]["menagerie_commit"][:12], fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(CHARTS, "chart1.png"), facecolor="white")
    plt.close(fig)
    cv["chart1"] = {n: dict(mass_kg=rob[n]["spec"]["mass_kg"], sum_torque_Nm=rob[n]["spec"]["sum_torque_limit_Nm"],
                            dof=rob[n]["spec"]["dof"], torque_per_kg=rob[n]["spec"]["torque_per_kg_Nm_per_kg"],
                            knee_lever_cm=round(rob[n]["spec"]["knee_torque_to_weight_m"] * 100, 2)) for n in names}

    # chart 2: simulation speed (uniform and native)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    order = sorted(names, key=lambda n: rob[n]["speed_uniform"]["steps_per_s"])
    yi = np.arange(len(order))
    u = [rob[n]["speed_uniform"]["real_time_factor"] for n in order]
    nat = [rob[n]["speed_native"]["real_time_factor"] for n in order]
    ax.barh(yi + 0.2, u, 0.4, color="#000000", label="uniform settings: 2 ms step, robot-floor contacts, PD hold")
    ax.barh(yi - 0.2, nat, 0.4, color="#bbbbbb", label="native settings as shipped in Menagerie (own timestep)")
    for i, n in enumerate(order):
        ax.text(u[i] + 0.5, i + 0.2, f"{u[i]:.1f}x  ({rob[n]['speed_uniform']['steps_per_s']:,.0f} steps/s)",
                va="center", fontsize=10)
        ax.text(nat[i] + 0.5, i - 0.2, f"{nat[i]:.1f}x  ({rob[n]['speed_native']['timestep_s'] * 1000:g} ms step)",
                va="center", fontsize=10, color="#555555")
    ax.axvline(1.0, color="#555555", ls=":", lw=1.5)
    ax.set_yticks(yi)
    ax.set_yticklabels([f"{n} ({rob[n]['spec']['dof']} DoF)" for n in order])
    ax.set_xlabel("Real-time factor on one CPU core (simulated seconds per wall-clock second)")
    ax.set_xlim(0, max(u + nat) * 1.35)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    ax.set_title("Chart 2. Humanoid robot simulation speed on a CPU (%s, MuJoCo %s, single thread)"
                 % (res["meta"]["cpu"], res["meta"]["mujoco"]), loc="left", fontsize=14)
    fig.text(0.02, 0.01, SRCLINE % "speed", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "chart2.png"), facecolor="white")
    plt.close(fig)
    cv["chart2"] = {n: dict(rtf_uniform=rob[n]["speed_uniform"]["real_time_factor"],
                            sps_uniform=rob[n]["speed_uniform"]["steps_per_s"],
                            rtf_native=rob[n]["speed_native"]["real_time_factor"],
                            sps_native=rob[n]["speed_native"]["steps_per_s"]) for n in names}

    # chart 3: push thresholds by direction
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    order = sorted(names, key=lambda n: rob[n]["push_min_dv_m_per_s"])
    yi = np.arange(len(order))
    cols = {"forward": "#000000", "backward": "#555555", "lateral": "#bbbbbb"}
    for k, (dn, off) in enumerate((("forward", 0.27), ("backward", 0.0), ("lateral", -0.27))):
        v = [rob[n]["push"][dn]["dv_m_per_s"] for n in order]
        ax.barh(yi + off, v, 0.25, color=cols[dn], label=f"{dn} push")
        for i, x in enumerate(v):
            lab = f"{x:.3f}" if x < DV_MAX else f">= {DV_MAX:g}"
            ax.text(x + 0.02, i + off, lab, va="center", fontsize=9)
    ax.set_yticks(yi)
    ax.set_yticklabels([f"{n} ({rob[n]['spec']['mass_kg']} kg)" for n in order])
    ax.set_xlim(0, max(rob[n]["push"][k]["dv_m_per_s"] for n in names for k in DIRECTIONS) * 1.3)
    ax.set_xlabel("Smallest toppling push, as velocity change of the whole body (m/s = N·s per kg), 0.1 s pulse at the root body")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    ax.set_title("Chart 3. Push test under an identical joint PD posture hold (not the vendors' controllers)",
                 loc="left", fontsize=14)
    fig.text(0.02, 0.01, SRCLINE % "push" + "; kp = tau_max / %g rad" % THETA_SAT, fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "chart3.png"), facecolor="white")
    plt.close(fig)
    cv["chart3"] = {n: {dn: rob[n]["push"][dn]["dv_m_per_s"] for dn in DIRECTIONS} for n in names}

    # chart 4: gain sensitivity (forward push) and the spec-vs-push relation
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=110)
    ax = axes[0]
    ths = [str(t) for t in THETA_SWEEP]
    mk = ["o", "s", "^", "D", "v", "P", "X"]
    for i, n in enumerate(names):
        v = [rob[n]["gain_sensitivity_forward"][t]["forward_dv_m_per_s"] for t in ths]
        ax.plot([float(t) for t in ths], v, marker=mk[i], color=["#000000", "#555555", "#bbbbbb"][i % 3],
                ls=["-", "--", ":"][i // 3], lw=2, ms=8, label=n)
    ax.set_xlabel("PD saturation angle theta_sat (rad); kp = tau_max / theta_sat, so left = stiffer")
    ax.set_ylabel("Forward toppling push (m/s velocity change)")
    ax.set_title("Sensitivity to the PD gain rule")
    ax.set_xticks([float(t) for t in ths])
    ax.legend(frameon=False, fontsize=10)
    style(ax)
    ax = axes[1]
    dmk = {"forward": "o", "backward": "s", "lateral": "^"}
    allv = []
    for dn, mkr in dmk.items():
        xs = [rob[n]["push"][dn]["capture_point_bound_m_per_s"] for n in names]
        ys = [rob[n]["push"][dn]["dv_m_per_s"] for n in names]
        allv += xs + ys
        ax.scatter(xs, ys, s=110, marker=mkr, color={"forward": "#000000", "backward": "#555555",
                                                    "lateral": "#bbbbbb"}[dn], edgecolors="#000000", label=f"{dn} push")
    offs = {"PAL TALOS": (-70, 8), "Apptronik Apollo": (8, -14), "Unitree G1": (8, -14)}
    for n in names:
        ax.annotate(n, (rob[n]["push"]["forward"]["capture_point_bound_m_per_s"], rob[n]["push"]["forward"]["dv_m_per_s"]),
                    textcoords="offset points", xytext=offs.get(n, (6, -12)), fontsize=9, color="#555555")
    lim_ = max(allv) * 1.1
    ax.plot([0, lim_], [0, lim_], color="#bbbbbb", ls="--", lw=1.5)
    ax.text(lim_ * 0.62, lim_ * 0.66, "measured = bound", color="#555555", fontsize=10, rotation=40)
    ax.set_xlim(0, lim_)
    ax.set_ylim(0, lim_)
    ax.set_xlabel("Capture-point bound: support-edge distance x sqrt(g / CoM height) (m/s)")
    ax.set_ylabel("Measured toppling push (m/s)")
    ax.set_title("Feet and CoM height predict the push (Pearson r = %.2f, 21 points)"
                 % res["summary"]["corr_push_vs_capture_point_bound"])
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.suptitle("Chart 4. What the push result depends on: the PD gains and the foot geometry",
                 fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.01, SRCLINE % "push", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(CHARTS, "chart4.png"), facecolor="white")
    plt.close(fig)
    cv["chart4"] = {n: dict(forward_by_theta={t: rob[n]["gain_sensitivity_forward"][t]["forward_dv_m_per_s"] for t in ths},
                            capture_point_bound={k: rob[n]["push"][k]["capture_point_bound_m_per_s"] for k in DIRECTIONS},
                            measured={k: rob[n]["push"][k]["dv_m_per_s"] for k in DIRECTIONS}) for n in names}
    return cv


if __name__ == "__main__":
    import sys
    if "--charts-only" in sys.argv:   # redraw charts from an existing results.json
        _res = json.load(open(os.path.join(HERE, "results.json")))
        make_charts(_res)
    else:
        main()
