"""Augmentations for fused food-scan clouds (numpy, operate on a scene dict, metres, z up).

Scene dict keys: xyz(N,3) rgb(N,3 in 0..1) normal(N,3) label(N) lift_label(N) lift_share(N) lift_nvotes(N)
and optionally boundary(N). Everything is toggleable from AugConfig.  Scale is deliberately limited to
+-5% because absolute size carries the volume signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

FOOD_MAX = 252          # label ids 1..251 are food; 252 table, 253 clutter, 254 container


def is_food(l):
    return (l > 0) & (l < 252)


@dataclass
class AugConfig:
    enable: bool = True
    rot_z: bool = True
    tilt_deg: float = 3.0
    scale: float = 0.05                    # +-5% only
    flip: bool = True
    jitter: bool = True
    jitter_sigma_mm: tuple = (0.5, 1.0)
    jitter_clip_mm: float = 3.0
    crop_p: float = 0.5
    crop_radius_m: tuple = (0.12, 0.25)
    cap_p: float = 0.3
    cap_frac: tuple = (0.2, 0.5)
    voxel_mm: tuple = (1.5, 3.0)           # per-sample voxel size jitter (train); eval uses DataConfig.voxel_mm
    color: bool = True
    brightness: float = 0.3
    hue: float = 0.05
    wb_gain: float = 0.1
    grey_p: float = 0.1
    color_noise: float = 0.02
    drop_rgb_p: float = 0.15
    drop_normal_p: float = 0.10
    drop_lift_p: float = 0.15
    lift_corrupt: bool = True
    lift_corrupt_p_present: float = 0.5    # extra corruption prob when the file already has lift_*
    blob_p: float = 0.3
    blob_radius_mm: tuple = (10.0, 40.0)
    merge_p: float = 0.2
    drop_inst_p: float = 0.05
    morph_p: float = 0.5
    morph_vox: tuple = (1, 3)
    share_noise: float = 0.1
    frag_p: float = 0.0                    # round 4: per scene, split one food instance of the lift into 2-3 pieces (plane / k-means)
    frag_pieces: tuple = (2, 3)
    frag_min_pts: int = 60                 # instance must have at least this many points (and each piece >= 15 %)
    corrupt_real: bool = False             # extra synthetic corruption on scenes whose lift is a REAL SAM 3 vote (real_votes=True)
    eval_corrupt_stored: bool = False      # eval only: corrupt the STORED lift too (regime R2 "hard lift")


# Regime R2 ("hard lift"): stronger vote corruption, applied (a) with p=1 to every train sample and (b) deterministically
# (fixed seed per scan) at eval, on stored lift_* as well as on lift synthesised from GT for old files.
HARD_LIFT = dict(blob_p=0.5, blob_radius_mm=(15.0, 40.0), merge_p=0.6, drop_inst_p=0.15, morph_p=0.7,
                 morph_vox=(2, 4), lift_corrupt_p_present=1.0)


def eval_aug(regime: str = "r1") -> "AugConfig":
    """Augmentation config used for val/eval datasets (no geometric/colour aug, only the lift corruption)."""
    if regime == "r2":
        return AugConfig(enable=False, lift_corrupt=True, eval_corrupt_stored=True, **HARD_LIFT)
    return AugConfig(enable=False, lift_corrupt=True)


# ----------------------------------------------------------------------------- geometry
def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def _axis_angle(axis, ang):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K).astype(np.float32)


def _sel(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(m) else v) for k, v in d.items()}


def geometric(d, rng, c: AugConfig):
    # --- crop around the container centre (origin) then partial-orbit cap removal
    n_food0 = int(is_food(d["label"]).sum())
    if rng.random() < c.crop_p:
        r = rng.uniform(*c.crop_radius_m)
        keep = np.linalg.norm(d["xyz"][:, :2], axis=1) < r
        if keep.sum() > 1000:
            d = _sel(d, keep)
    if rng.random() < c.cap_p:
        v = rng.normal(size=3)
        v[2] *= 0.4
        v /= np.linalg.norm(v)
        proj = d["xyz"] @ v
        thr = np.quantile(proj, 1.0 - rng.uniform(*c.cap_frac))
        d = _sel(d, proj < thr)
    # --- rigid + scale
    R = np.eye(3, dtype=np.float32)
    if c.rot_z:
        R = _rz(rng.uniform(0, 2 * math.pi)) @ R
    if c.tilt_deg > 0:
        ax = np.r_[rng.normal(size=2), 0.0]
        R = _axis_angle(ax, math.radians(rng.uniform(0, c.tilt_deg))) @ R
    xyz, nrm = d["xyz"] @ R.T, d["normal"] @ R.T
    if c.flip:
        for ax in (0, 1):
            if rng.random() < 0.5:
                xyz[:, ax] *= -1
                nrm[:, ax] *= -1
    s = 1.0 + rng.uniform(-c.scale, c.scale)
    d = dict(d, xyz=(xyz * s).astype(np.float32), normal=nrm.astype(np.float32))
    d["scale"] = s
    if c.jitter:
        sig = rng.uniform(*c.jitter_sigma_mm) * 1e-3
        j = np.clip(rng.normal(0, sig, d["xyz"].shape), -c.jitter_clip_mm * 1e-3, c.jitter_clip_mm * 1e-3)
        d["xyz"] = (d["xyz"] + j).astype(np.float32)
    return d


# ----------------------------------------------------------------------------- colour
def color(rgb, rng, c: AugConfig):
    rgb = rgb.copy()
    rgb *= rng.uniform(1 - c.brightness, 1 + c.brightness)
    rgb *= rng.uniform(1 - c.wb_gain, 1 + c.wb_gain, size=3)
    th = rng.uniform(-c.hue, c.hue) * 2 * math.pi              # hue rotation in YIQ space
    T = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    rot = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    rgb = rgb @ (np.linalg.inv(T) @ rot @ T).T
    if rng.random() < c.grey_p:
        rgb = np.repeat((rgb @ np.array([0.299, 0.587, 0.114]))[:, None], 3, 1)
    rgb += rng.normal(0, c.color_noise, rgb.shape)
    return np.clip(rgb, 0, 1).astype(np.float32)


def channel_dropout(flags, rng, c: AugConfig):
    """flags: dict group->bool(keep). Returns updated flags; never drops every group."""
    probs = {"rgb": c.drop_rgb_p, "normal": c.drop_normal_p, "lift": c.drop_lift_p}
    keep = {k: not (rng.random() < probs[k]) for k in probs}
    if not any(keep.values()):
        keep[rng.choice(list(keep))] = True
    return keep


# ----------------------------------------------------------------------------- lift synthesis / corruption
def synth_lift(label, rng):
    """No lift_* in file: start from GT. 253/254 kept, 252 -> 0."""
    lift = np.where(label >= 252, label, label).astype(np.int32)
    lift[label == 252] = 0
    share = np.ones(len(label), np.float32)
    nv = np.zeros(len(label), np.int16)
    for i in np.unique(label[is_food(label)]):
        m = label == i
        share[m] = rng.uniform(0.7, 1.0)
        nv[m] = rng.integers(3, 20)
    for k in (253, 254):
        m = label == k
        share[m] = rng.uniform(0.8, 1.0)
        nv[m] = rng.integers(3, 20)
    share = np.clip(share + rng.normal(0, 0.08, len(share)), 0.2, 1.0).astype(np.float32)
    nv = np.where(lift > 0, np.maximum(1, rng.poisson(np.maximum(nv, 1))), 0).astype(np.int16)
    return lift, share, nv


def corrupt_lift(xyz, lift, share, nv, rng, c: AugConfig, vox_m: float):
    lift, share, nv = lift.copy(), share.copy(), nv.copy()
    orig = lift.copy()
    tree = cKDTree(xyz)
    ids = [int(i) for i in np.unique(lift[is_food(lift)])]
    if ids and rng.random() < c.blob_p:                       # flip a blob to a neighbouring instance
        for _ in range(rng.integers(1, 4)):
            fp = np.nonzero(is_food(lift))[0]
            if not len(fp):
                break
            ctr = fp[rng.integers(len(fp))]
            r = rng.uniform(*c.blob_radius_mm) * 1e-3
            ball = np.array(tree.query_ball_point(xyz[ctr], r))
            src = lift[ctr]
            near = np.array(tree.query_ball_point(xyz[ctr], 2 * r))
            others = [int(i) for i in np.unique(lift[near]) if is_food(np.array([i]))[0] and i != src]
            new = others[rng.integers(len(others))] if others else (0 if rng.random() < 0.7 else src)
            lift[ball[lift[ball] == src]] = new
    ids = [int(i) for i in np.unique(lift[is_food(lift)])]
    if len(ids) >= 2 and rng.random() < c.merge_p:             # merge two touching instances
        pairs = []
        trees = {i: cKDTree(xyz[lift == i]) for i in ids}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d, _ = trees[ids[a]].query(xyz[lift == ids[b]][::4], k=1)
                if d.min() < 6e-3:
                    pairs.append((ids[a], ids[b]))
        if pairs:
            a, b = pairs[rng.integers(len(pairs))]
            lift[lift == b] = a
    ids = [int(i) for i in np.unique(lift[is_food(lift)])]
    if ids and rng.random() < c.drop_inst_p:
        lift[lift == ids[rng.integers(len(ids))]] = 0
    ids = [int(i) for i in np.unique(lift[is_food(lift)])]
    if ids and rng.random() < c.morph_p:                       # erode / dilate by 1-3 voxels
        r = rng.integers(c.morph_vox[0], c.morph_vox[1] + 1) * vox_m
        i = ids[rng.integers(len(ids))]
        m = lift == i
        if rng.random() < 0.5:
            other = ~m
            if other.any() and m.any():
                d, _ = cKDTree(xyz[other]).query(xyz[m], k=1)
                idx = np.nonzero(m)[0][d < r]
                lift[idx] = 0
        elif m.any():
            cand = np.nonzero(lift == 0)[0]
            if len(cand):
                d, _ = cKDTree(xyz[m]).query(xyz[cand], k=1)
                lift[cand[d < r]] = i
    changed = lift != orig
    share = np.clip(share + rng.normal(0, c.share_noise, len(share)), 0.05, 1.0)
    share[changed] = rng.uniform(0.3, 0.8, changed.sum())
    nv = np.maximum(0, (nv * np.exp(rng.normal(0, 0.3, len(nv)))).round()).astype(np.int16)
    nv[lift == 0] = 0
    return lift, share.astype(np.float32), nv


def fragment_lift(xyz, lift, share, nv, rng, c: AugConfig):
    """Fragmentation aug (mirrors real SAM 3 votes, whose dominant error is one item cut into several ids): pick a food
    instance of the lift and split it into 2-3 pieces by a random plane through a point near its centroid (2 pieces) or k-means
    (3 pieces / 50 %).  The largest piece keeps the id, the others get fresh ids (<= 251, unused).  Their lift_share is lowered.
    Returns (lift, share, nv, n_new_ids)."""
    lift, share = lift.copy(), share.copy()
    ids, cnt = np.unique(lift[is_food(lift)], return_counts=True)
    ids = [int(i) for i, k in zip(ids, cnt) if k >= c.frag_min_pts]
    free = [i for i in range(1, FOOD_MAX) if i not in set(np.unique(lift).tolist())]
    if not ids or not free:
        return lift, share, nv, 0
    i = ids[rng.integers(len(ids))]
    idx = np.nonzero(lift == i)[0]
    p = xyz[idx]
    k = int(rng.integers(c.frag_pieces[0], c.frag_pieces[1] + 1))
    if k == 2 and rng.random() < 0.5:
        nrm = rng.normal(size=3)
        nrm /= np.linalg.norm(nrm) + 1e-12
        off = rng.uniform(-0.25, 0.25) * np.ptp(p @ nrm)
        part = ((p - p.mean(0)) @ nrm > off).astype(int)
    else:
        from scipy.cluster.vq import kmeans2
        try:
            _, part = kmeans2(p.astype(np.float64), k, minit="++", seed=int(rng.integers(1 << 30)))
        except Exception:                                   # noqa: BLE001
            return lift, share, nv, 0
    sizes = np.bincount(part, minlength=part.max() + 1)
    keep = int(sizes.argmax())
    n_new = 0
    for q in range(len(sizes)):
        if q == keep or sizes[q] < 0.15 * len(idx) or n_new >= len(free):
            continue
        sel = idx[part == q]
        lift[sel] = free[n_new]
        share[sel] = rng.uniform(0.4, 0.9, len(sel))
        n_new += 1
    return lift, share.astype(np.float32), nv, n_new


def apply_lift(d, rng, c: AugConfig, vox_m: float, has_lift: bool, force_corrupt: bool = False):
    """Fill lift_* in d (synthesise from label when absent) and corrupt per config."""
    if not has_lift:
        lift, share, nv = synth_lift(d["label"], rng)
        do = c.lift_corrupt or force_corrupt
    else:
        lift, share, nv = d["lift_label"], d["lift_share"], d["lift_nvotes"]
        do = (c.lift_corrupt and c.enable and rng.random() < c.lift_corrupt_p_present) or \
            (force_corrupt and c.eval_corrupt_stored)
        if d.get("real_votes") and not c.corrupt_real:
            do = False                                      # real SAM 3 votes already carry real errors
    if do:
        lift, share, nv = corrupt_lift(d["xyz"], lift, share, nv, rng, c, vox_m)
    if c.enable and c.frag_p > 0 and not d.get("real_votes") and rng.random() < c.frag_p:
        lift, share, nv, _ = fragment_lift(d["xyz"], lift, share, nv, rng, c)
    return dict(d, lift_label=lift.astype(np.int32), lift_share=share, lift_nvotes=nv)
