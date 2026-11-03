#!/usr/bin/env python3
"""How long can a humanoid robot hold a box?

Static joint torques (MuJoCo inverse statics), a two-node actuator thermal model
and a battery budget for five open humanoid models from MuJoCo Menagerie.

Run:  python3 analysis.py      (CPU only, about one minute)
Needs: mujoco>=3.2, numpy, matplotlib.
Outputs: results.json and charts/chart1.png ... chart4.png next to this file.
"""
import json
import math
import os

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
DEG = math.pi / 180.0

# ---------------------------------------------------------------------------
# Sourced constants (see README.md for links)
# ---------------------------------------------------------------------------
SRC = {
    # Two-node thermal model of a quasi-direct-drive actuator, fans on
    # (Urs et al., arXiv:2202.12395, Table 3): winding-housing and housing-ambient
    # resistances [K/W], winding and housing capacitances [J/K].
    "thermal_fan": {"R_wh": 0.828, "R_ha": 1.44, "C_w": 15.9, "C_h": 146.0},
    "thermal_nofan": {"R_wh": 1.09, "R_ha": 6.68, "C_w": 18.4, "C_h": 138.0},
    # Same paper: continuous torque 4.4 N m at 7.5:1, Kt = 0.105 N m/A;
    # published sustainable peaks: 9.2 A for 20 s and 22 A for 2 s.
    "urs_cont_torque_Nm": 4.4, "urs_ratio": 7.5, "urs_kt": 0.105,
    "urs_peak20_A": 9.2, "urs_peak2_A": 22.0,
    # Continuous/peak anchors
    "h1_2_arm_payload_rated_kg": 7.0, "h1_2_arm_payload_peak_kg": 21.0,  # unitree.com/h1
    "ak80_64_rated_Nm": 48.0, "ak80_64_peak_Nm": 120.0,  # cubemars.com AK80-64
    "mini_cheetah_cont_Nm": 6.9, "mini_cheetah_peak_Nm": 17.0,  # arXiv:2202.12395 Table 2
    # Unitree G1 measured copper-loss coefficients b [W/(N m)^2], left arm
    # (Deniz et al., arXiv:2606.15915, Table I) and whole-robot rest power (BMS).
    "g1_b": {"shoulder_pitch": 0.4184, "shoulder_roll": 0.3563, "shoulder_yaw": 0.2799,
             "elbow": 0.3942, "wrist_roll": 0.0, "wrist_pitch": 0.3768, "wrist_yaw": 0.0},
    "g1_rest_power_W": 135.0,
    # Unitree G1 product page: 9000 mAh, 13-string lithium pack, battery life about 2 h.
    "g1_capacity_Ah": 9.0, "g1_cells_series": 13, "g1_battery_life_h": 2.0,
    # Vendor statements used only for comparison (not as model inputs)
    "g1_arm_max_load_kg": 2.0, "g1_edu_arm_max_load_kg": 3.0, "g1_vendor_mass_kg": 35.0,
    "g1_vendor_knee_peak_Nm": 90.0, "g1_edu_vendor_knee_peak_Nm": 120.0,
    "h1_vendor_mass_kg": 47.0, "h1_vendor_knee_Nm": 360.0, "h1_vendor_ankle_Nm": 59.0, "h1_battery_Wh": 864.0,
    "apollo_vendor_payload_kg": 25.0, "apollo_vendor_mass_kg": 72.5, "apollo_battery_h": 4.0,
}
CELL_NOMINAL_V = 3.6  # our assumption for a Li-ion cell (stated in the article)
R_CONT = SRC["h1_2_arm_payload_rated_kg"] / SRC["h1_2_arm_payload_peak_kg"]  # 0.333
R_SWEEP = [0.25, R_CONT, 0.40]
K_SWEEP = [0.5, 1.0, 2.0, 4.0]  # thermal-capacitance scale factors (sensitivity)
PAYLOADS = [0.0, 5.0, 10.0, 20.0]

# ---------------------------------------------------------------------------
# Robot configurations: joint names for the semantic pose, hand and foot bodies
# ---------------------------------------------------------------------------
ROBOTS = {
    "Unitree G1": dict(
        file="unitree_g1/g1.xml",
        leg={s: dict(hip_pitch=f"{n}_hip_pitch_joint", hip_roll=f"{n}_hip_roll_joint",
                     knee=f"{n}_knee_joint", ankle_pitch=f"{n}_ankle_pitch_joint",
                     ankle_roll=f"{n}_ankle_roll_joint", foot=f"{n}_ankle_roll_link") for s, n in (("L", "left"), ("R", "right"))},
        arm={s: dict(sh_pitch=f"{n}_shoulder_pitch_joint", sh_roll=f"{n}_shoulder_roll_joint",
                     elbow=f"{n}_elbow_joint", extra=f"{n}_shoulder_yaw_joint",
                     hand=(f"{n}_wrist_yaw_link", "ipos"))
             for s, n in (("L", "left"), ("R", "right"))}),
    "Unitree H1": dict(
        file="unitree_h1/h1.xml",
        leg={s: dict(hip_pitch=f"{n}_hip_pitch", hip_roll=f"{n}_hip_roll", knee=f"{n}_knee",
                     ankle_pitch=f"{n}_ankle", ankle_roll=None, foot=f"{n}_ankle_link")
             for s, n in (("L", "left"), ("R", "right"))},
        arm={s: dict(sh_pitch=f"{n}_shoulder_pitch", sh_roll=f"{n}_shoulder_roll",
                     elbow=f"{n}_elbow", extra=f"{n}_shoulder_yaw", hand=(f"{n}_elbow_link", "sphere"))
             for s, n in (("L", "left"), ("R", "right"))}),
    "Booster T1": dict(
        file="booster_t1/t1.xml",
        leg={s: dict(hip_pitch=f"{n}_Hip_Pitch", hip_roll=f"{n}_Hip_Roll", knee=f"{n}_Knee_Pitch",
                     ankle_pitch=f"{n}_Ankle_Pitch", ankle_roll=f"{n}_Ankle_Roll", foot=f"{n.lower()}_foot_link")
             for s, n in (("L", "Left"), ("R", "Right"))},
        arm={s: dict(sh_pitch=f"{n}_Shoulder_Pitch", sh_roll=f"{n}_Shoulder_Roll",
                     elbow=f"{n}_Elbow_Pitch", extra=f"{n}_Elbow_Yaw", hand=(f"{n.lower()}_hand_link", "ipos"))
             for s, n in (("L", "Left"), ("R", "Right"))}),
    "Apptronik Apollo": dict(
        file="apptronik_apollo/apptronik_apollo.xml",
        leg={s: dict(hip_pitch=f"{n}_hip_fe", hip_roll=f"{n}_hip_aa", knee=f"{n}_knee_fe",
                     ankle_pitch=f"{n}_ankle_pd", ankle_roll=f"{n}_ankle_ie", foot=f"{n}_foot_link")
             for s, n in (("L", "l"), ("R", "r"))},
        arm={s: dict(sh_pitch=f"{n}_shoulder_fe", sh_roll=f"{n}_shoulder_aa",
                     elbow=f"{n}_elbow_fe", extra=f"{n}_shoulder_ie", hand=(f"{n}_wrist_pitch_link", "ipos"))
             for s, n in (("L", "l"), ("R", "r"))}),
    "Fourier N1": dict(
        file="fourier_n1/n1.xml",
        leg={s: dict(hip_pitch=f"{n}_hip_pitch_joint", hip_roll=f"{n}_hip_roll_joint",
                     knee=f"{n}_knee_pitch_joint", ankle_pitch=f"{n}_ankle_pitch_joint",
                     ankle_roll=f"{n}_ankle_roll_joint", foot=f"{n}_foot_pitch_link") for s, n in (("L", "left"), ("R", "right"))},
        arm={s: dict(sh_pitch=f"{n}_shoulder_pitch_joint", sh_roll=f"{n}_shoulder_roll_joint",
                     elbow=f"{n}_elbow_pitch_joint", extra=f"{n}_shoulder_yaw_joint",
                     hand=(f"{n}_end_effector_link", "origin"))
             for s, n in (("L", "left"), ("R", "right"))}),
}

# Leg postures: (hip flexion, knee flexion) in degrees. Ankle pitch and trunk
# pitch are solved so that each foot is flat and the whole-body centre of mass
# (robot + box) lies over the middle of the support area.
LEG_POSES = {"stand": (10.0, 20.0), "half_squat": (45.0, 90.0), "deep_squat": (70.0, 110.0)}
SWING_LEG = (30.0, 60.0)
# Arm postures: hand target relative to the shoulder pitch axis, in units of the
# arm reach L: (forward, down).
ARM_POSES = {"carry": (0.60, 0.45), "reach": (0.95, 0.05)}
SCENARIOS = [  # (name, leg pose, arm pose, single support?)
    ("stand_carry", "stand", "carry", False),
    ("stand_reach", "stand", "reach", False),
    ("half_squat_carry", "half_squat", "carry", False),
    ("deep_squat_carry", "deep_squat", "carry", False),
    ("one_leg_carry", "stand", "carry", True),
    ("stand_carry_com_over_ankle", "stand", "carry", False),
]
CAP_SCENARIOS = ["stand_carry", "stand_reach", "half_squat_carry", "stand_carry_com_over_ankle"]


def load_model(path):
    """Load an MJCF, drop visual/collision meshes (inertias are explicit)."""
    spec = mujoco.MjSpec.from_file(path)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for m in list(spec.meshes):
        spec.delete(m)
    for t in list(spec.textures):
        spec.delete(t)
    m = spec.compile()
    m.body_gravcomp[:] = 0.0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return m


def actuator_limits(m):
    """Per-dof torque limit [N m] from the MJCF and the field it came from."""
    lim = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = m.actuator_trnid[a, 0]
        gear = abs(m.actuator_gear[a, 0])
        cands = []
        if m.actuator_forcelimited[a]:
            cands.append((max(abs(m.actuator_forcerange[a])) * gear, "actuator forcerange"))
        elif (m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE and m.actuator_ctrllimited[a]):
            cands.append((max(abs(m.actuator_ctrlrange[a])) * gear * abs(m.actuator_gainprm[a, 0]),
                          "motor ctrlrange x gear"))
        if m.jnt_actfrclimited[j]:
            cands.append((max(abs(m.jnt_actfrcrange[j])), "joint actuatorfrcrange"))
        if cands:
            lim[m.jnt_dofadr[j]] = min(cands)
    return lim


class Robot:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.m = load_model(os.path.join(DATA, cfg["file"]))
        self.d = mujoco.MjData(self.m)
        m = self.m
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.lim = actuator_limits(m)
        self.support_mode = "sole"
        self.mass = float(m.body_subtreemass[0])
        self.jid = lambda n: m.joint(n).id
        self.qadr = lambda n: m.jnt_qposadr[m.joint(n).id]
        self.dadr = lambda n: m.jnt_dofadr[m.joint(n).id]
        # reference orientation of each foot with all joints at zero
        self.d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, self.d)
        self.foot_R0, self.sole = {}, {}
        for s in "LR":
            fb = m.body(cfg["leg"][s]["foot"]).id
            self.foot_R0[s] = self.d.xmat[fb].reshape(3, 3).copy()
            self.sole[s] = self._sole_center(fb)
        # flexion signs
        self.sign = {}
        for s in "LR":
            leg = cfg["leg"][s]
            kr = m.jnt_range[self.jid(leg["knee"])]
            self.sign[leg["knee"]] = 1.0 if kr[0] + kr[1] > 0 else -1.0
            self.sign[leg["hip_pitch"]] = self._hip_flex_sign(leg)
        # arm reach
        self.reach = {s: self._arm_reach(s) for s in "LR"}

    # -- geometry helpers ---------------------------------------------------
    def _sole_center(self, fb):
        """Support centre in the foot frame: mean of non-mesh geoms or sites on the foot."""
        m, d = self.m, self.d
        pts = [m.geom_pos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == fb]
        pts += [m.site_pos[s] for s in range(m.nsite) if m.site_bodyid[s] == fb and "foot" in m.site(s).name]
        if not pts:  # mesh-only foot: use the foot's centre of mass as the support centre
            c = m.body_ipos[fb].copy()
            c[2] = 0.0
            return c
        c = np.mean(np.array(pts), axis=0)
        c[2] = min(p[2] for p in pts)
        return c

    def _hip_flex_sign(self, leg):
        m, d = self.m, self.d
        kb = int(m.jnt_bodyid[self.jid(leg["knee"])])
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
        x0 = d.xpos[kb, 0]
        d.qpos[self.qadr(leg["hip_pitch"])] += 0.2
        mujoco.mj_kinematics(m, d)
        x1 = d.xpos[kb, 0]
        d.qpos[:] = m.qpos0
        return 1.0 if x1 > x0 else -1.0

    def proximal(self, arm):
        """Id of the shoulder joint closest to the torso (pitch or roll)."""
        a, b = self.jid(arm["sh_pitch"]), self.jid(arm["sh_roll"])
        return a if self.m.jnt_bodyid[a] < self.m.jnt_bodyid[b] else b

    def hand_point(self, s):
        """World position of the grip point and its body id (after kinematics)."""
        m, d = self.m, self.d
        body, mode = self.cfg["arm"][s]["hand"]
        b = m.body(body).id
        if mode == "ipos":
            local = m.body_ipos[b]
        elif mode == "sphere":
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            local = m.geom_pos[gs[0]]
        else:
            local = np.zeros(3)
        R = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + R @ local, b

    def _arm_reach(self, s):
        m, d = self.m, self.d
        arm = self.cfg["arm"][s]
        d.qpos[:] = m.qpos0
        best = 0.0
        lo, hi = m.jnt_range[self.jid(arm["elbow"])]
        for e in np.linspace(lo, hi, 60):
            d.qpos[self.qadr(arm["elbow"])] = e
            mujoco.mj_kinematics(m, d)
            p, _ = self.hand_point(s)
            a = d.xanchor[self.proximal(arm)]
            best = max(best, float(np.linalg.norm(p - a)))
        d.qpos[:] = m.qpos0
        return best

    # -- posture --------------------------------------------------------------
    def set_base(self, pitch, roll):
        q = np.zeros(4)
        qr, qp = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qr, np.array([1.0, 0, 0]), roll)
        mujoco.mju_axisAngle2Quat(qp, np.array([0, 1.0, 0]), pitch)
        mujoco.mju_mulQuat(q, qp, qr)
        self.d.qpos[3:7] = q
        self.d.qpos[0:3] = [0, 0, 1.0]

    def foot_tilt(self, s):
        fb = self.m.body(self.cfg["leg"][s]["foot"]).id
        R = self.d.xmat[fb].reshape(3, 3) @ self.foot_R0[s].T
        return math.atan2(R[0, 2], R[2, 2]), math.atan2(-R[1, 2], R[2, 2])  # pitch, roll

    def sole_world(self, s):
        fb = self.m.body(self.cfg["leg"][s]["foot"]).id
        return self.d.xpos[fb] + self.d.xmat[fb].reshape(3, 3) @ self.sole[s], fb

    def arm_ik(self, s, fwd, down, seed=False):
        """Shoulder pitch, shoulder roll and elbow so that the grip point sits at
        (fwd*L forward, down*L below) the shoulder axis, in the torso frame."""
        m, d = self.m, self.d
        arm = self.cfg["arm"][s]
        js = [arm["sh_pitch"], arm["sh_roll"], arm["elbow"], arm["extra"]]
        jp = self.proximal(arm)
        L = self.reach[s]
        tb = m.body_parentid[m.jnt_bodyid[jp]]
        jacp = np.zeros((3, m.nv))
        err = None
        if seed:
            # seed the elbow so that the shoulder-to-grip distance matches the target
            want = L * math.hypot(fwd, down)
            ea = self.qadr(arm["elbow"])
            lo, hi = m.jnt_range[self.jid(arm["elbow"])]
            best = None
            for e in np.linspace(lo, hi, 80):
                d.qpos[ea] = e
                mujoco.mj_kinematics(m, d)
                dist = np.linalg.norm(self.hand_point(s)[0] - d.xanchor[jp])
                if best is None or abs(dist - want) < best[0]:
                    best = (abs(dist - want), e)
            d.qpos[ea] = best[1]
        # seed: arm straight down with elbow bent a little toward flexion
        for it in range(200):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            Rt = d.xmat[1].reshape(3, 3)  # pelvis/base frame: x forward, z up
            anchor = d.xanchor[jp]
            target = anchor + Rt @ np.array([fwd * L, 0.0, -down * L])
            # keep the hand in the shoulder's sagittal plane (torso frame y of anchor)
            p, b = self.hand_point(s)
            err = target - p
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jac(m, d, jacp, None, p, b)
            cols = [self.dadr(j) for j in js]
            J = jacp[:, cols]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            # null-space pull of the extra joint toward zero keeps the elbow down
            qx = d.qpos[self.qadr(arm["extra"])]
            N = np.eye(len(js)) - np.linalg.pinv(J) @ J
            dq = dq + N @ np.array([0, 0, 0, -0.3 * qx])
            for j, dqi in zip(js, dq):
                a = self.qadr(j)
                lo, hi = m.jnt_range[self.jid(j)]
                d.qpos[a] = float(np.clip(d.qpos[a] + 0.7 * dqi, lo, hi))
        return float(np.linalg.norm(err))

    def com(self, box_mass):
        m, d = self.m, self.d
        c = np.zeros(3)
        tot = 0.0
        for b in range(1, m.nbody):
            c = c + m.body_mass[b] * d.xipos[b]
            tot += m.body_mass[b]
        hp = [self.hand_point(s)[0] for s in "LR"]
        box = 0.5 * (hp[0] + hp[1])
        c = c + box_mass * box
        return c / (tot + box_mass)

    def pose(self, leg_pose, arm_pose, box_mass, single):
        """Solve the posture; returns diagnostics."""
        m, d = self.m, self.d
        cfg = self.cfg
        d.qpos[:] = m.qpos0
        hip, knee = LEG_POSES[leg_pose]
        for s in "LR":
            leg = cfg["leg"][s]
            h, k = (hip, knee)
            if single and s == "R":
                h, k = SWING_LEG
            d.qpos[self.qadr(leg["hip_pitch"])] = self.sign[leg["hip_pitch"]] * h * DEG
            d.qpos[self.qadr(leg["knee"])] = self.sign[leg["knee"]] * k * DEG
        self.set_base(0.0, 0.0)
        mujoco.mj_kinematics(m, d)
        # seed the arms near hanging and solve IK repeatedly inside the balance loop
        fwd, down = ARM_POSES[arm_pose]
        stance = ["L"] if single else ["L", "R"]
        # unknowns
        legL = cfg["leg"]["L"]
        has_ar = legL.get("ankle_roll") is not None
        names = ["pitch", "ankle"] + (["hiproll", "ankleroll" if has_ar else "baseroll"] if single else [])
        x = np.zeros(len(names))

        def apply(x):
            pitch = x[0]
            roll = x[3] if (single and not has_ar) else 0.0
            self.set_base(pitch, roll)
            for s in stance:
                d.qpos[self.qadr(cfg["leg"][s]["ankle_pitch"])] = x[1]
            if not single:
                d.qpos[self.qadr(cfg["leg"]["R"]["ankle_pitch"])] = x[1]
            if single:
                d.qpos[self.qadr(legL["hip_roll"])] = x[2]
                if has_ar:
                    d.qpos[self.qadr(legL["ankle_roll"])] = x[3]
                # swing ankle: keep its foot level-ish (cosmetic, light)
            mujoco.mj_kinematics(m, d)

        def resid(x):
            apply(x)
            r = []
            fp, fr = self.foot_tilt("L")
            sol = [self.sole_world(s)[0] for s in stance]
            sup = np.mean(sol, axis=0)
            if self.support_mode == "ankle":  # centre of mass over the ankle joints instead
                sup = np.mean([self.d.xpos[self.m.body(cfg["leg"][s]["foot"]).id] for s in stance], axis=0)
            c = self.com(box_mass)
            r += [fp, c[0] - sup[0]]
            if single:
                r += [fr, c[1] - sup[1]]
            return np.array(r)

        ik_err = 0.0
        for outer in range(6):
            for it in range(40):
                r = resid(x)
                if np.max(np.abs(r)) < 1e-6:
                    break
                J = np.zeros((len(r), len(x)))
                for i in range(len(x)):
                    dx = np.zeros(len(x))
                    dx[i] = 1e-5
                    J[:, i] = (resid(x + dx) - r) / 1e-5
                step = np.linalg.lstsq(J, r, rcond=None)[0]
                x = x - np.clip(step, -0.2, 0.2)
            apply(x)
            ik_err = max(self.arm_ik(s, fwd, down, seed=(outer == 0)) for s in "LR")
        r = resid(x)
        # joint-range check over every limited hinge joint
        out_of_range = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                q = d.qpos[m.jnt_qposadr[j]]
                if q < lo - 1e-4 or q > hi + 1e-4:
                    out_of_range.append(f"{m.joint(j).name}={q / DEG:.1f}deg")
        inrange = not out_of_range
        arm_q = {k: round(float(d.qpos[self.qadr(v)] / DEG), 1) for k, v in cfg["arm"]["L"].items() if k != "hand"}
        return dict(residual=float(np.max(np.abs(r))), arm_ik_err_m=ik_err, ankle_in_range=inrange, out_of_range=out_of_range, arm_q_left_deg=arm_q,
                    trunk_pitch_deg=float(x[0] / DEG), ankle_deg=float(x[1] / DEG))

    # -- statics ---------------------------------------------------------------
    def static_torques(self, box_mass, single):
        """Joint torques for static equilibrium with foot wrenches (min-norm split)."""
        m, d = self.m, self.d
        d.qvel[:] = 0
        d.qacc[:] = 0
        mujoco.mj_forward(m, d)
        bias = d.qfrc_bias.copy() - d.qfrc_passive.copy()
        # payload: half the box weight at each grip point
        Fp = np.zeros(m.nv)
        jacp = np.zeros((3, m.nv))
        for s in "LR":
            p, b = self.hand_point(s)
            mujoco.mj_jac(m, d, jacp, None, p, b)
            Fp += jacp.T @ np.array([0, 0, -0.5 * box_mass * G])
        stance = ["L"] if single else ["L", "R"]
        Jc = []
        for s in stance:
            p, fb = self.sole_world(s)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, p, fb)
            Jc += [jp, jr]
        Jc = np.vstack(Jc)
        rhs = bias[:6] - Fp[:6]
        w = np.linalg.lstsq(Jc[:, :6].T, rhs, rcond=None)[0]
        tau = bias - Jc.T @ w - Fp
        base_res = float(np.max(np.abs(tau[:6])))
        # centre of pressure offsets per stance foot [m]
        cops = []
        for i, s in enumerate(stance):
            f = w[6 * i:6 * i + 3]
            mo = w[6 * i + 3:6 * i + 6]
            cops.append(float(math.hypot(mo[0], mo[1]) / max(f[2], 1e-9)))
        return tau, w, base_res, cops


def joint_label(name):
    n = name.lower()
    side = "left" if (n.startswith("left") or n.startswith("l_")) else ("right" if (n.startswith("right") or n.startswith("r_")) else "")
    for key, lab in (("wrist_pitch", "wrist pitch"), ("ankle", "ankle"), ("knee", "knee"), ("hip_fe", "hip pitch"), ("hip_aa", "hip roll"),
                     ("hip_pitch", "hip pitch"), ("hip_roll", "hip roll"), ("shoulder_fe", "shoulder pitch"),
                     ("shoulder_pitch", "shoulder pitch"), ("shoulder_aa", "shoulder roll"),
                     ("shoulder_roll", "shoulder roll"), ("elbow", "elbow"), ("waist", "waist"), ("torso", "torso"),
                     ("wrist_pitch", "wrist pitch"), ("wrist", "wrist"), ("hip_ie", "hip yaw"), ("hip_yaw", "hip yaw")):
        if key in n:
            return (side + " " + lab).strip()
    return name


def utilisation(rob, tau):
    m = rob.m
    best = (0.0, None, None, None)
    per = {}
    for dof, (lim, src) in rob.lim.items():
        u = abs(tau[dof]) / lim
        jn = m.joint(int(m.dof_jntid[dof])).name
        per[jn] = (float(tau[dof]), float(lim), float(u))
        if u > best[0]:
            best = (float(u), jn, float(tau[dof]), float(lim))
    return best, per


def evaluate(rob, scen, box):
    name, leg, arm, single = scen
    rob.support_mode = "ankle" if name.endswith("com_over_ankle") else "sole"
    diag = rob.pose(leg, arm, box, single)
    tau, w, base_res, cops = rob.static_torques(box, single)
    best, per = utilisation(rob, tau)
    return dict(diag=diag, base_residual=base_res, cop_offset_m=cops, u=best[0], joint=best[1],
                tau=best[2], limit=best[3], per=per)


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------
def time_to_limit(x, p, k=1.0, t_max=4 * 3600.0, dt=0.02):
    """Seconds until the winding reaches its limit from a cold start, for a constant
    torque equal to x times the continuous torque. Heating is normalised so that the
    continuous torque gives a steady-state winding rise equal to the allowed rise."""
    Rwh, Rha, Cw, Ch = p["R_wh"], p["R_ha"], p["C_w"] * k, p["C_h"] * k
    if x <= 1.0:
        return math.inf
    P = x * x / (Rwh + Rha)  # allowed rise normalised to 1
    w = h = 0.0
    t = 0.0
    # exact discretisation of the linear two-node system
    A = np.array([[-1 / (Rwh * Cw), 1 / (Rwh * Cw)], [1 / (Rwh * Ch), -(1 / Rwh + 1 / Rha) / Ch]])
    from math import exp  # noqa
    # use matrix exponential via eigen decomposition
    ev, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    xs = -np.linalg.solve(A, np.array([P / Cw, 0.0]))  # steady state
    # T(t) = xs + V exp(ev t) Vi (x0 - xs), x0 = 0
    c = Vi @ (-xs)

    def wind(t):
        return float(np.real(xs[0] + (V @ (np.exp(ev * t) * c))[0]))

    lo, hi = 0.0, 1.0
    while wind(hi) < 1.0:
        hi *= 2
        if hi > t_max:
            return math.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wind(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def peak_ratio_for_duration(t_target, p):
    lo, hi = 1.0 + 1e-6, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if time_to_limit(mid, p) > t_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def r3(v, n=3):
    if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
        return None
    return round(float(v), n)


def main():
    res = {"menagerie_commit": open(os.path.join(DATA, "COMMIT")).read().strip(),
           "mujoco_version": mujoco.__version__, "sources": SRC,
           "assumptions": {"r_cont_central": round(R_CONT, 4), "r_sweep": [round(r, 4) for r in R_SWEEP],
                           "k_sweep": K_SWEEP, "cell_nominal_V": CELL_NOMINAL_V,
                           "leg_poses_deg": LEG_POSES, "swing_leg_deg": SWING_LEG,
                           "arm_poses_reach_units": ARM_POSES, "payloads_kg": PAYLOADS}}

    # ---- thermal validation -------------------------------------------------
    i_cont = SRC["urs_cont_torque_Nm"] / SRC["urs_ratio"] / SRC["urs_kt"]
    val = {"i_cont_A": r3(i_cont)}
    for key, p in (("fan", SRC["thermal_fan"]), ("nofan", SRC["thermal_nofan"])):
        x20 = peak_ratio_for_duration(20.0, p)
        x2 = peak_ratio_for_duration(2.0, p)
        val[key] = {"pred_peak20_A": r3(x20 * i_cont, 2), "pred_peak2_A": r3(x2 * i_cont, 2),
                    "pred_ratio_2s_over_20s": r3(x2 / x20), "tau_winding_s": r3(p["R_wh"] * p["C_w"], 1),
                    "tau_housing_s": r3(p["R_ha"] * p["C_h"], 1), "R_total_KperW": r3(p["R_wh"] + p["R_ha"], 3)}
    val["published_peak20_A"] = SRC["urs_peak20_A"]
    val["published_peak2_A"] = SRC["urs_peak2_A"]
    val["published_ratio_2s_over_20s"] = r3(SRC["urs_peak2_A"] / SRC["urs_peak20_A"])
    val["fan_err_peak20_pct"] = r3(100 * (val["fan"]["pred_peak20_A"] / SRC["urs_peak20_A"] - 1), 1)
    val["fan_err_peak2_pct"] = r3(100 * (val["fan"]["pred_peak2_A"] / SRC["urs_peak2_A"] - 1), 1)
    val["fan_err_ratio_pct"] = r3(100 * (val["fan"]["pred_ratio_2s_over_20s"] / val["published_ratio_2s_over_20s"] - 1), 1)
    res["thermal_validation"] = val
    res["cont_peak_anchors"] = {
        "h1_2_arm_payload_rated_over_peak": r3(R_CONT, 4),
        "ak80_64_rated_over_peak": r3(SRC["ak80_64_rated_Nm"] / SRC["ak80_64_peak_Nm"], 4),
        "mini_cheetah_cont_over_peak": r3(SRC["mini_cheetah_cont_Nm"] / SRC["mini_cheetah_peak_Nm"], 4)}
    PF = SRC["thermal_fan"]

    # time-to-limit curve vs x for chart 3
    xs = np.exp(np.linspace(np.log(1.02), np.log(4.5), 120))
    curve = {k: [time_to_limit(x, PF, k) for x in xs] for k in K_SWEEP}

    # ---- robots -------------------------------------------------------------
    robots = {}
    for rname, cfg in ROBOTS.items():
        rob = Robot(rname, cfg)
        out = {"sole_center_ahead_of_ankle_m": r3(float(rob.sole["L"][0]), 3),
               "mass_kg": r3(rob.mass, 2), "arm_reach_m": r3(rob.reach["L"]), "neq": int(rob.m.neq),
               "n_limited_dofs": len(rob.lim),
               "limit_source": sorted(set(v[1] for v in rob.lim.values())), "scenarios": {}}
        # selected joint limits
        lims = {}
        for dof, (lim, src) in rob.lim.items():
            jn = rob.m.joint(int(rob.m.dof_jntid[dof])).name
            lims[jn] = r3(lim, 1)
        out["limits_Nm"] = lims
        for scen in SCENARIOS:
            sname = scen[0]
            loads = PAYLOADS if not scen[3] else [0.0, 5.0, 10.0]
            rows = []
            for box in loads:
                ev = evaluate(rob, scen, box)
                rows.append({"payload_kg": box, "u_max": r3(ev["u"]), "bottleneck": ev["joint"],
                             "bottleneck_label": joint_label(ev["joint"]),
                             "tau_Nm": r3(ev["tau"], 2), "limit_Nm": r3(ev["limit"], 1),
                             "t_limit_s": {str(round(r, 3)): r3(time_to_limit(ev["u"] / r, PF), 1) for r in R_SWEEP},
                             "t_limit_s_k": {str(k): r3(time_to_limit(ev["u"] / R_CONT, PF, k), 1) for k in K_SWEEP},
                             "trunk_pitch_deg": r3(ev["diag"]["trunk_pitch_deg"], 1),
                             "ankle_deg": r3(ev["diag"]["ankle_deg"], 1),
                             "feasible": ev["diag"]["ankle_in_range"] and ev["diag"]["arm_ik_err_m"] < 0.05,
                             "out_of_range": ev["diag"]["out_of_range"],
                             "arm_q_left_deg": ev["diag"]["arm_q_left_deg"],
                             "balance_residual": r3(ev["diag"]["residual"], 6),
                             "arm_ik_err_m": r3(ev["diag"]["arm_ik_err_m"], 4),
                             "base_residual_N": r3(ev["base_residual"], 6),
                             "cop_offset_m": [r3(c, 4) for c in ev["cop_offset_m"]],
                             "top_joints": sorted(({"joint": k, "tau": r3(v[0], 2), "limit": r3(v[1], 1), "u": r3(v[2])}
                                                   for k, v in ev["per"].items()), key=lambda z: -z["u"])[:6]})
            out["scenarios"][sname] = rows
        # payload limits for standing carry/reach and half squat (bisection)
        caps = {}
        for scen in [sc for sc in SCENARIOS if sc[0] in CAP_SCENARIOS]:
            sc = {}
            for label, target in (("peak_u1", 1.0), ("continuous", R_CONT)):
                u0 = evaluate(rob, scen, 0.0)["u"]
                if u0 >= target:
                    sc[label] = 0.0
                    continue
                lo, hi = 0.0, 120.0
                if evaluate(rob, scen, hi)["u"] < target:
                    sc[label] = None
                    continue
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    if evaluate(rob, scen, mid)["u"] < target:
                        lo = mid
                    else:
                        hi = mid
                ev = evaluate(rob, scen, lo)
                sc[label] = r3(lo, 2)
                sc[label + "_joint"] = joint_label(ev["joint"])
            caps[scen[0]] = sc
        out["payload_caps_kg"] = caps
        robots[rname] = out
        if rname == "Unitree G1":
            g1 = rob
        print(rname, "done", flush=True)
    res["robots"] = robots

    # ---- G1 battery budget -----------------------------------------------------
    m = g1.m
    b_meas = {}
    for side in ("left", "right"):
        for k, v in SRC["g1_b"].items():
            b_meas[f"{side}_{k}_joint"] = v
    elbow_lim = g1.lim[g1.dadr("left_elbow_joint")][0]
    p_peak_elbow = SRC["g1_b"]["elbow"] * elbow_lim ** 2

    def b_of(jn):
        if jn in b_meas:
            return b_meas[jn], "measured"
        lim = g1.lim[g1.dadr(jn)][0]
        return p_peak_elbow / lim ** 2, "scaled"

    E_Wh = SRC["g1_capacity_Ah"] * SRC["g1_cells_series"] * CELL_NOMINAL_V
    walk_avg_W = E_Wh / SRC["g1_battery_life_h"]
    batt = {"energy_Wh_est": r3(E_Wh, 1), "avg_power_from_2h_W": r3(walk_avg_W, 1),
            "rest_power_W": SRC["g1_rest_power_W"], "elbow_limit_Nm": r3(elbow_lim, 1),
            "elbow_copper_loss_at_limit_W": r3(p_peak_elbow, 1),
            "elbow_copper_loss_at_cont_W": r3(SRC["g1_b"]["elbow"] * (R_CONT * elbow_lim) ** 2, 2),
            "elbow_steady_rise_at_cont_K_fanR": r3(SRC["g1_b"]["elbow"] * (R_CONT * elbow_lim) ** 2 * (PF["R_wh"] + PF["R_ha"]), 1),
            "scenarios": {}}
    for scen in SCENARIOS:
        rows = []
        loads = [0.0, 2.0, 5.0, 10.0, 20.0] if not scen[3] else [0.0, 2.0, 5.0, 10.0]
        for box in loads:
            ev = evaluate(g1, scen, box)
            cu_meas = cu_scaled = 0.0
            for jn, (tau, lim, u) in ev["per"].items():
                b, kind = b_of(jn)
                if kind == "measured":
                    cu_meas += b * tau * tau
                else:
                    cu_scaled += b * tau * tau
            P = SRC["g1_rest_power_W"] + cu_meas + cu_scaled
            t_th = time_to_limit(ev["u"] / R_CONT, PF)
            rows.append({"payload_kg": box, "copper_arm_W": r3(cu_meas, 2), "copper_other_W": r3(cu_scaled, 2),
                         "total_W": r3(P, 1), "battery_hold_h": r3(E_Wh / P, 2),
                         "thermal_hold_s": r3(t_th, 1) if math.isfinite(t_th) else None,
                         "bottleneck": joint_label(ev["joint"]), "u_max": r3(ev["u"])})
        batt["scenarios"][scen[0]] = rows
    res["g1_battery"] = batt

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)

    make_charts(res, xs, curve)
    print("wrote results.json and charts")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
PAL = ["#000000", "#555555", "#bbbbbb", "#111111", "#555555"]
LS = ["-", "-", "-", "--", "--"]
MARK = ["o", "s", "^", "D", "v"]
SRCLINE = "Source: our MuJoCo inverse statics on MuJoCo Menagerie models; stabilarity/hub research code"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def make_charts(res, xs, curve):
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    names = list(res["robots"].keys())
    # chart 1: bottleneck utilisation vs payload, standing carry and reach
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=110, sharey=True)
    for ax, sc, title in ((axes[0], "stand_carry", "Standing, elbows bent (carry)"),
                          (axes[1], "stand_reach", "Standing, arms extended (reach)")):
        for i, n in enumerate(names):
            rows = res["robots"][n]["scenarios"][sc]
            rows = [r for r in rows if r["feasible"]]
            ax.plot([r["payload_kg"] for r in rows], [100 * r["u_max"] for r in rows], marker=MARK[i],
                    color=PAL[i], ls=LS[i], lw=2, ms=8, label=n)
        ax.axhline(100, color="#c62828", ls="--", lw=1.5)
        ax.axhline(100 * R_CONT, color="#f9a825", ls="--", lw=1.5)
        ax.text(0.2, 103, "MJCF torque limit", color="#c62828", fontsize=11)
        ax.text(0.2, 100 * R_CONT + 3, "continuous estimate (1/3 of limit)", color="#555555", fontsize=11)
        ax.set_title(title)
        ax.set_xlabel("Box mass held in both hands (kg)")
        style(ax)
    axes[0].set_ylabel("Bottleneck joint torque as % of its limit")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.suptitle("Chart 1. Static torque utilisation of the most loaded joint", fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.01, SRCLINE, fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(CHARTS, "chart1.png"), facecolor="white")
    plt.close(fig)

    # chart 2: time to thermal limit vs payload (stand carry, half squat carry)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=110, sharey=True)
    cap = 4 * 3600.0
    for ax, sc, title in ((axes[0], "stand_carry", "Standing carry"),
                          (axes[1], "stand_reach", "Standing, arms extended")):
        for i, n in enumerate(names):
            rows = [r for r in res["robots"][n]["scenarios"][sc] if r["feasible"]]
            ys = []
            for r in rows:
                t = r["t_limit_s"][str(round(R_CONT, 3))]
                ys.append(cap if t is None else min(t, cap))
            ax.plot([r["payload_kg"] for r in rows], ys, marker=MARK[i], color=PAL[i], ls=LS[i], lw=2, ms=8, label=n)
        ax.set_yscale("log")
        ax.set_yticks([1, 10, 60, 600, 3600, cap])
        ax.set_yticklabels(["1 s", "10 s", "1 min", "10 min", "1 h", "no limit"])
        ax.axhline(cap, color="#bbbbbb", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Box mass held in both hands (kg)")
        style(ax)
    axes[0].set_ylabel("Time until the bottleneck winding reaches its limit (log)")
    axes[1].legend(frameon=False, loc="lower left")
    axes[0].set_ylim(0.3, cap * 2)
    fig.suptitle("Chart 2. Holding time before the thermal limit (cold start, continuous torque = 1/3 of MJCF limit)",
                 fontsize=15, x=0.02, ha="left")
    fig.text(0.02, 0.01, SRCLINE + "; two-node thermal model from arXiv:2202.12395", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(CHARTS, "chart2.png"), facecolor="white")
    plt.close(fig)

    # chart 3: thermal model curve and validation
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    for i, k in enumerate(K_SWEEP):
        ax.plot(xs, [t if math.isfinite(t) else np.nan for t in curve[k]], color=PAL[i], ls=LS[i], lw=2.2,
                label=f"thermal capacitance x{k:g}" + (" (measured actuator)" if k == 1.0 else ""))
    v = res["thermal_validation"]
    ic = v["i_cont_A"]
    ax.scatter([v["published_peak20_A"] / ic, v["published_peak2_A"] / ic], [20, 2], s=140, facecolors="white", edgecolors="#000000",
               linewidths=2, zorder=5, label="published sustainable peaks (9.2 A for 20 s, 22 A for 2 s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 5.0)
    ax.set_xticks([1, 1.5, 2, 3, 4, 5])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Held torque / continuous torque (current ratio)")
    ax.set_ylabel("Seconds until the winding reaches its limit (cold start)")
    ax.set_title("Chart 3. Two-node actuator thermal model and its check against published peak durations",
                 loc="left", fontsize=15)
    ax.legend(frameon=False)
    style(ax)
    fig.text(0.02, 0.01, "Source: parameters and peak durations from Urs et al., arXiv:2202.12395; our model",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "chart3.png"), facecolor="white")
    plt.close(fig)

    # chart 4: G1 battery hours vs thermal time, log scale
    b = res["g1_battery"]
    labels, bat, th = [], [], []
    for sc, lab in (("stand_carry", "stand, carry"), ("stand_reach", "stand, arms extended"),
                    ("half_squat_carry", "half squat, carry"), ("deep_squat_carry", "deep squat, carry")):
        for r in b["scenarios"][sc]:
            if r["payload_kg"] in (0.0, 2.0, 5.0):  # 10 kg exceeds the G1 wrist torque limit
                labels.append(f"{lab}, {r['payload_kg']:g} kg")
                bat.append(r["battery_hold_h"] * 3600)
                th.append(r["thermal_hold_s"] if r["thermal_hold_s"] is not None else 4 * 3600.0)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    yi = np.arange(len(labels))[::-1]
    ax.barh(yi + 0.2, bat, 0.4, color="#bbbbbb", label="battery-limited holding time")
    ax.barh(yi - 0.2, th, 0.4, color="#000000", label="thermal-limited holding time (bars at 4 h = no thermal limit)")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 60, 600, 3600, 4 * 3600])
    ax.set_xticklabels(["1 s", "10 s", "1 min", "10 min", "1 h", "4 h"])
    ax.axvline(7200, color="#555555", ls=":", lw=1.5)
    ax.text(7400, yi[0] + 0.45, "vendor battery life, about 2 h", fontsize=10, color="#555555")
    ax.set_yticks(yi)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Holding time (log scale)")
    ax.set_title("Chart 4. Unitree G1: the battery lasts hours, the bottleneck motor overheats first",
                 loc="left", fontsize=15)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.text(0.02, 0.01, "Source: our estimate; 135 W rest power and arm copper-loss coefficients from arXiv:2606.15915; "
             "9 Ah 13-string pack from unitree.com/g1", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(CHARTS, "chart4.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
