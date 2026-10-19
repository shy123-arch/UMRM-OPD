"""Reaction plan: a structured, automatically extracted description of a reaction motion.

Extracted from any canonical motion (88x24x3 @20 Hz, +x right, -z forward). Used as
  * supervision / oracle for the "reason" stage (no manual labelling needed),
  * an interface to the generator (plan -> root trajectory constraints),
  * a verifier signal (does the generated motion follow the plan?).
"""
import numpy as np

FPS = 20
L_HIP, R_HIP, HEAD, L_WRIST, R_WRIST = 1, 2, 15, 20, 21
GROUPS = {"legs": [4, 5, 7, 8, 10, 11], "torso": [3, 6, 9, 12], "head": [15],
          "left_arm": [16, 18, 20], "right_arm": [17, 19, 21]}
DIRS = ["forward", "forward_right", "right", "back_right", "back", "back_left", "left", "forward_left"]
PLAN_KEYS = ["duration_s", "onset_s", "settle_s", "forward_m", "right_m", "distance_m", "path_m",
             "peak_speed_mps", "yaw_deg", "pelvis_drop_m", "pelvis_rise_m",
             "left_hand_raise_m", "right_hand_raise_m", "left_hand_reach_m", "right_hand_reach_m",
             "energy_legs", "energy_torso", "energy_head", "energy_left_arm", "energy_right_arm"]


def facing(x):
    r = x[:, R_HIP, [0, 2]] - x[:, L_HIP, [0, 2]]
    return np.stack([r[:, 1], -r[:, 0]], -1)  # (x,z) forward; initial right (1,0) -> (0,-1)


def extract(x, valid, speed_thresh=0.25):
    T = int(valid.sum()); x = np.asarray(x[:T], dtype=np.float64)
    root = x[:, 0]; h = root[:, [0, 2]]
    disp = h[-1] - h[0]
    step = np.linalg.norm(np.diff(h, axis=0), axis=-1)
    body_speed = np.linalg.norm(np.diff(x, axis=0), axis=-1).mean(1) * FPS
    body_speed = np.convolve(body_speed, np.ones(3) / 3, mode="same")
    active = np.where(body_speed > speed_thresh)[0]
    onset = active[0] / FPS if len(active) else T / FPS
    settle = (active[-1] + 1) / FPS if len(active) else T / FPS
    f = facing(x); a0 = np.arctan2(f[0, 0], -f[0, 1]); a1 = np.arctan2(f[-1, 0], -f[-1, 1])
    yaw = np.degrees(np.angle(np.exp(1j * (a0 - a1))))  # + = turned left
    rel = x - root[:, None]
    fwd0 = np.array([f[0, 0], 0, f[0, 1]]) / (np.linalg.norm(f[0]) + 1e-9)
    p = {
        "duration_s": T / FPS, "onset_s": float(onset), "settle_s": float(settle),
        "forward_m": float(-disp[1]), "right_m": float(disp[0]), "distance_m": float(np.linalg.norm(disp)),
        "path_m": float(step.sum()), "peak_speed_mps": float(np.percentile(step * FPS, 95)) if len(step) else 0.0,
        "yaw_deg": float(yaw),
        "pelvis_drop_m": float(root[0, 1] - root[:, 1].min()), "pelvis_rise_m": float(root[:, 1].max() - root[0, 1]),
        "left_hand_raise_m": float((rel[:, L_WRIST, 1] - rel[0, L_WRIST, 1]).max()),
        "right_hand_raise_m": float((rel[:, R_WRIST, 1] - rel[0, R_WRIST, 1]).max()),
        "left_hand_reach_m": float(((rel[:, L_WRIST] - rel[0, L_WRIST]) @ fwd0).max()),
        "right_hand_reach_m": float(((rel[:, R_WRIST] - rel[0, R_WRIST]) @ fwd0).max()),
    }
    for g, j in GROUPS.items():
        p[f"energy_{g}"] = float(np.linalg.norm(np.diff(rel[:, j], axis=0), axis=-1).sum(0).mean())
    p.update(categorical(p))
    return p


def categorical(p):
    d, path = p["distance_m"], p["path_m"]
    loco = "stay" if d < 0.3 and path < 0.6 else "step" if d < 1.0 else "walk" if p["peak_speed_mps"] < 2.2 else "run"
    if d < 0.3:
        direction = "none"
    else:
        ang = np.degrees(np.arctan2(p["right_m"], p["forward_m"]))  # 0 fwd, 90 right
        direction = DIRS[int(np.round(ang / 45)) % 8]
    y = p["yaw_deg"]
    turn = "none" if abs(y) < 30 else ("around" if abs(y) > 135 else ("left" if y > 0 else "right"))
    en = {g: p[f"energy_{g}"] for g in GROUPS}
    main = max(en, key=en.get)
    posture = "crouch" if p["pelvis_drop_m"] > 0.25 else "jump" if p["pelvis_rise_m"] > 0.15 else "upright"
    return {"locomotion": loco, "direction": direction, "turn": turn, "main_part": main, "posture": posture}


def vector(p):
    v = [p[k] for k in PLAN_KEYS if k != "yaw_deg"]
    y = np.radians(p["yaw_deg"])
    return np.array(v + [np.sin(y), np.cos(y)], dtype=np.float64)


def smoothstep(u):
    u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)


def coarse_root_path(p, n_frames, fps):
    """Plan -> horizontal root path in canonical (x=right, z=-forward), linear-in-smoothstep between
    onset and settle. Only final displacement + timing are used (not the true path shape)."""
    t = np.arange(n_frames) / fps
    span = max(p["settle_s"] - p["onset_s"], 1.0 / fps)
    u = smoothstep((t - p["onset_s"]) / span)
    return np.stack([u * p["right_m"], -u * p["forward_m"]], -1)
