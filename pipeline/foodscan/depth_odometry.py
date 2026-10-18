"""Depth odometry: re-estimate camera poses with point-to-plane ICP on LiDAR depth, because
ARKit VIO can run away mid-scan while trackingState still says "normal".

    python -m foodscan.depth_odometry SCAN_DIR [--method geometric|colored|features]
        # writes SCAN_DIR/frames_refined_<method>.jsonl
    python -m foodscan.depth_odometry --check    # asserts on the local reference scans

Frame i is registered to frame i-1 (high-confidence depth < 1.5 m, voxel-downsampled), seeded
with ARKit's relative motion. If that fails or disagrees with ARKit by > 2 cm / 3 deg, identity
and constant-velocity seeds are tried too and the best ICP result wins. Frames no ICP result
passes are flagged bad and move with the previous velocity.

table="drop"/"downweight" leave the RANSAC table plane out of ICP (see refine_scan).

Limitation: on a flat table most normals are vertical, so in-plane translation is weakly constrained
(ICP can slide there); ARKit is kept whenever it fits about as well as ICP.

Poses are camera-to-world in ARKit convention (same as frames.jsonl `transform`).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .loader import _ARKIT_TO_CV, Scan, Frame

VOXEL = 0.015
SCALES = (0.03, 0.015, 0.0075)  # colored ICP coarse-to-fine voxels
MAX_DEPTH = 1.5
MIN_FITNESS, MAX_RMSE = 0.3, 0.02
MARGIN_RMSE, MARGIN_FIT = 0.002, 0.03
WINDOW = 5  # frames in the local model ICP registers against
DISAGREE_T, DISAGREE_R = 0.02, np.radians(3)
# per-frame bad flags
MIN_SUPPORT = 0.01          # weakest/strongest eigenvalue of sum(n n^T) over ICP correspondences
MAX_STEP_T, MAX_STEP_R = 0.05, np.radians(10)  # per frame (15 fps)
MIN_INLIERS = 30            # feature odometry
KEY_GAP = 4                 # keyframe distance for feature matching
METHODS = ("geometric", "colored", "features")


def _cloud(f: Frame, voxel: float = VOXEL, color: bool = False) -> o3d.geometry.PointCloud:
    """Camera-frame (OpenCV) point cloud of high-confidence depth with normals (and RGB at depth res)."""
    d = f.masked_depth(max_depth=MAX_DEPTH)
    v, u = np.nonzero(d)
    z = d[v, u].astype(np.float64)
    K = f.K_depth
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    if color:
        from PIL import Image
        img = np.asarray(Image.fromarray(f.rgb()).resize(f.depth_size, Image.BOX)) / 255.0
        pc.colors = o3d.utility.Vector3dVector(img[v, u])
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return pc


def _motion_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    D = np.linalg.inv(A) @ B
    return np.linalg.norm(D[:3, 3]), np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))


_reg = o3d.pipelines.registration


def _icp(src, tgt, init):
    r = _reg.registration_icp(src, tgt, 0.05, init, _reg.TransformationEstimationPointToPlane(),
                              _reg.ICPConvergenceCriteria(max_iteration=40))
    return r.transformation, r.fitness, r.inlier_rmse


def _colored(srcs, tgts, init, scales=SCALES, iters=(30, 20, 15)):
    """Coarse-to-fine colored ICP (Park et al. 2017); srcs/tgts are per-scale clouds."""
    T = init
    for s, src, tgt, n in zip(scales, srcs, tgts, iters):
        try:
            T = _reg.registration_colored_icp(src, tgt, 2 * s, T, _reg.TransformationEstimationForColoredICP(),
                                              _reg.ICPConvergenceCriteria(1e-6, 1e-6, n)).transformation
        except RuntimeError:  # no correspondences (sparse object-only clouds): keep the seed
            pass
    return T


def _eval(src, tgt, T):
    r = _reg.evaluate_registration(src, tgt, 0.05, T)
    return T, r.fitness, r.inlier_rmse


def _support(src, tgt, T) -> float:
    """Degeneracy: weakest/strongest eigenvalue of sum(n n^T) over correspondences (translation observability).
    Low on flat tables, where point-to-plane (and geometry-driven colored) ICP can slide in-plane."""
    c = np.asarray(_reg.evaluate_registration(src, tgt, 0.05, T).correspondence_set)
    if len(c) < 10:
        return 0.0
    n = np.asarray(src.normals)[c[:, 0]]
    w = np.linalg.eigvalsh(n.T @ n)
    return float(w[0] / w[-1])


def _features(frames):
    """Half-res SIFT keypoints (full-res pixel coords) + descriptors per frame."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=3000)
    out = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f.rgb(), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(g, None)
        out.append((np.array([k.pt for k in kp], np.float64).reshape(-1, 2) * 2, des))
    return out


def _lift(f: Frame, depth, px):
    """3D camera-frame points (OpenCV) for full-res pixels px using high-conf depth; -> (pts, valid)."""
    w, h = f.depth_size
    u = np.clip((px[:, 0] * w / f.rgb_size[0]).astype(int), 0, w - 1)
    v = np.clip((px[:, 1] * h / f.rgb_size[1]).astype(int), 0, h - 1)
    z = depth[v, u]
    K = f.K_rgb
    p = np.stack([(px[:, 0] - K[0, 2]) / K[0, 0] * z, (px[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
    return p, z > 0


def _pnp(fj, depth_j, kj, fi, ki):
    """Relative motion X_i = M X_j from SIFT matches (3D from frame j depth, pixels in frame i) -> (M, inliers)."""
    import cv2
    if kj[1] is None or ki[1] is None or len(kj[1]) < 8 or len(ki[1]) < 8:
        return None, 0
    m = [a for a, b in cv2.BFMatcher().knnMatch(kj[1], ki[1], k=2) if a.distance < 0.8 * b.distance]
    if len(m) < 8:
        return None, 0
    qi = np.array([x.queryIdx for x in m]); ti = np.array([x.trainIdx for x in m])
    P, ok = _lift(fj, depth_j, kj[0][qi])
    if ok.sum() < 8:
        return None, 0
    P, px = P[ok], ki[0][ti][ok]
    good, rv, tv, inl = cv2.solvePnPRansac(P, px, fi.K_rgb, None, iterationsCount=300, reprojectionError=3.0,
                                           confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not good or inl is None or len(inl) < 8:
        return None, 0
    inl = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(P[inl], px[inl], fi.K_rgb, None, rv, tv)
    M = np.eye(4); M[:3, :3] = cv2.Rodrigues(rv)[0]; M[:3, 3] = tv[:, 0]
    return M, len(inl)


TABLES = ("keep", "drop", "downweight")
TABLE_MARGIN = 0.01   # m, points this close to the table plane count as table
TABLE_FRAC = 0.15     # table share of points kept by "downweight"
MIN_OBJ = 300         # object points (base voxel) needed to leave the table out
MAX_TILT = np.radians(15)  # table normal vs ARKit gravity (world +y is up)


def _plane(pc):
    """RANSAC plane (n, d) with n.x + d = 0, n pointing up, on a world cloud; None unless roughly horizontal."""
    if len(pc.points) < 100:
        return None
    (a, b, c, d), _ = pc.segment_plane(0.01, 3, 200)
    n = np.array([a, b, c]); s = np.sign(b) or 1.0
    return (n * s, d * s) if abs(b) > np.cos(MAX_TILT) else None


def _split_table(pc, Tw, plane, mode, margin, rng):
    """pc (camera frame; world = Tw @ p) minus table points ("drop") or with only TABLE_FRAC of them kept."""
    pts = np.asarray(pc.points)
    tab = np.abs((pts @ Tw[:3, :3].T + Tw[:3, 3]) @ plane[0] + plane[1]) < margin
    keep, nobj = ~tab, int((~tab).sum())
    if mode == "downweight":
        idx = np.flatnonzero(tab)
        keep[rng.choice(idx, min(len(idx), int(TABLE_FRAC / (1 - TABLE_FRAC) * nobj)), replace=False)] = True
    return pc.select_by_index(np.flatnonzero(keep)), nobj


def _icp_t(src, tgt, init, iters=15):
    """Point-to-plane ICP for translation only (rotation of `init` kept)."""
    T = init.copy()
    q, n = np.asarray(tgt.points), np.asarray(tgt.normals)
    p = np.asarray(src.points)
    for _ in range(iters):
        c = np.asarray(_reg.evaluate_registration(src, tgt, 0.05, T).correspondence_set)
        if len(c) < 10:
            break
        nn = n[c[:, 1]]
        w = p[c[:, 0]] @ T[:3, :3].T + T[:3, 3]
        A = nn.T @ nn
        if np.linalg.eigvalsh(A)[0] < 1e-6 * len(c):
            break
        dt = np.linalg.solve(A, (nn * np.sum(nn * (q[c[:, 1]] - w), 1, keepdims=True)).sum(0))
        T[:3, 3] += dt
        if np.linalg.norm(dt) < 1e-5:
            break
    return T


# Full 6-DoF ICP is the default. rot="arkit" keeps ARKit's relative rotation and solves translation only.
def refine_scan(path: str | Path, window: int = WINDOW, method: str = "geometric", table: str = "keep",
                table_margin_m: float = TABLE_MARGIN, rot: str = "icp"):
    """-> (poses Nx4x4 camera-to-world ARKit convention, bad Nx bool, fitness N, rmse N, why N list[str]).

    method: "geometric" (point-to-plane ICP), "colored" (multi-scale colored ICP) or "features"
    (SIFT + depth PnP-RANSAC, polished by colored ICP, colored ICP fallback below MIN_INLIERS).
    table: "keep"; "drop" (remove points within table_margin_m of the RANSAC table plane from source and
    model); "downweight" (keep a random TABLE_FRAC of them). The plane is fit on the local model each frame,
    must be within MAX_TILT of ARKit gravity, and with < MIN_OBJ object points or a still-degenerate support
    that frame falls back to "keep" (reason "table-fallback ..." in `why`).
    rot: "icp" or "arkit" (ARKit relative rotation, translation re-solved).
    bad = degenerate geometry (support < MIN_SUPPORT) | fitness/rmse failure | low feature inliers |
    implausible step (> 5 cm or 10 deg). `why` lists the reasons ("" if fine).
    """
    assert method in METHODS and table in TABLES and rot in ("icp", "arkit"), (method, table, rot)
    fr = Scan(path, poses="arkit").frames
    ark = [f.T_cw for f in fr]  # OpenCV-convention ARKit poses
    scales = (VOXEL,) if method == "geometric" else SCALES
    base = scales.index(VOXEL)
    pcs = [[_cloud(f, v, color=method != "geometric") for v in scales] for f in fr]
    feats = _features(fr) if method == "features" else None
    depths = [f.masked_depth(max_depth=MAX_DEPTH) for f in fr] if feats else None
    ok = lambda r: r[1] >= MIN_FITNESS and r[2] <= MAX_RMSE

    def solve(i, P, model, seeds, T_prev):
        def register(init):
            if method == "geometric":
                return _icp(P[0], model[0], init)
            return _eval(P[base], model[base], _colored(P, model, init))
        reasons, feat_ok, res = [], False, None
        if feats:
            best, nin = None, 0
            for j in {i - 1, max(0, i - KEY_GAP)}:
                M, n = _pnp(fr[j], depths[j], feats[j], fr[i], feats[i])
                if n > nin:
                    best, nin = T[j] @ np.linalg.inv(M), n
            if best is not None and (lambda d: d[0] > MAX_STEP_T or d[1] > MAX_STEP_R)(_motion_diff(T_prev, best)):
                nin = 0  # implausible jump (repeated texture / mismatches): treat as failed
            if nin >= MIN_INLIERS:
                pol = _colored(P[1:], model[1:], best, scales=SCALES[1:], iters=(10, 10))
                dt, dr = _motion_diff(best, pol)
                res = _eval(P[base], model[base], pol if dt < 0.015 and dr < np.radians(2) else best)
                feat_ok = ok(res)
            if not feat_ok:
                reasons.append(f"inliers={nin}" if nin < MIN_INLIERS else "feature pose misfit")
        if not feat_ok:
            ev = _reg.evaluate_registration(P[base], model[base], 0.05, seeds[0])
            ark_res = (seeds[0], ev.fitness, ev.inlier_rmse)
            res = register(seeds[0])
            dt, dr = _motion_diff(seeds[0], res[0])
            if not ok(res) or dt > DISAGREE_T or dr > DISAGREE_R:
                cands = [res] + [register(s) for s in seeds[1:]]
                res = max(cands, key=lambda r: (ok(r), r[1] - 10 * r[2]))
            # point-to-plane ICP slides along the flat table, so only override ARKit when it fits
            # clearly worse than the ICP result (or fails outright)
            if ok(ark_res) and not (ark_res[2] - res[2] > MARGIN_RMSE or res[1] - ark_res[1] > MARGIN_FIT):
                res = ark_res
        if rot == "arkit":
            Tr = res[0].copy(); Tr[:3, :3] = seeds[0][:3, :3]
            res = _eval(P[base], model[base], _icp_t(P[base], model[base], Tr))
        sup = 1.0 if feat_ok else _support(P[base], model[base], res[0])  # textured feature poses don't need geometry
        return res, feat_ok, reasons, sup

    T = [ark[0]]
    world = [[copy.deepcopy(c).transform(ark[0]) for c in pcs[0]]]  # per-frame lists of per-scale world clouds
    vel = np.eye(4)  # last accepted frame-to-frame motion (T_{i-1,i})
    bad, fit, rmse, why = [False], [1.0], [0.0], [""]
    for i in range(1, len(fr)):
        model = [_merge([w[s] for w in world]) for s in range(len(scales))]
        rel_ark = np.linalg.inv(ark[i - 1]) @ ark[i]
        seeds = [T[-1] @ rel_ark, T[-1], T[-1] @ vel]  # ARKit, identity, constant velocity
        notes = []
        attempts = [(pcs[i], model)]
        if table != "keep":
            plane = _plane(model[base])
            if plane is None:
                notes.append("table-fallback: no horizontal plane")
            else:
                rng = np.random.default_rng(i)
                P = [_split_table(c, seeds[0], plane, table, table_margin_m, rng) for c in pcs[i]]
                M = [_split_table(c, np.eye(4), plane, table, table_margin_m, rng) for c in model]
                if min(P[base][1], M[base][1]) < MIN_OBJ:
                    notes.append(f"table-fallback: {min(P[base][1], M[base][1])} object points")
                else:
                    attempts.insert(0, ([p[0] for p in P], [m[0] for m in M]))
        for k, (Pk, Mk) in enumerate(attempts):
            res, feat_ok, reasons, sup = solve(i, Pk, Mk, seeds, T[-1])
            if sup >= MIN_SUPPORT or k == len(attempts) - 1:
                break
            notes.append(f"table-fallback: support={sup:.3f}")
        reasons = notes + reasons
        good = ok(res)
        Ti = res[0] if good else T[-1] @ vel
        if not good:
            reasons.append(f"fit={res[1]:.2f} rmse={res[2]*100:.1f}cm")
        bad_r = [r for r in reasons if not r.startswith("table-fallback")]
        if sup < MIN_SUPPORT:
            bad_r.append(f"degenerate support={sup:.3f}"); reasons.append(bad_r[-1])
        st, sr = _motion_diff(T[-1], Ti)
        if st > MAX_STEP_T or sr > MAX_STEP_R:
            bad_r.append(f"step={st*100:.1f}cm/{np.degrees(sr):.1f}deg"); reasons.append(bad_r[-1])
        if good:
            vel = np.linalg.inv(T[-1]) @ Ti
        bad.append(bool(bad_r)); fit.append(res[1]); rmse.append(res[2]); why.append("; ".join(reasons))
        T.append(Ti)
        world = (world + [[copy.deepcopy(c).transform(Ti) for c in pcs[i]]])[-window:]
    poses = np.array(T) @ np.linalg.inv(_ARKIT_TO_CV)
    return poses, np.array(bad), np.array(fit), np.array(rmse), why


def _merge(clouds):
    pc = clouds[0]
    for c in clouds[1:]:
        pc = pc + c
    return pc


# --- evaluation helpers -------------------------------------------------------------------

def world_cloud(f: Frame, T_arkit: np.ndarray, stride: int = 2, rgb: bool = False):
    """High-conf depth back-projected to world with an ARKit-convention pose (and RGB at depth res)."""
    d = f.masked_depth(max_depth=MAX_DEPTH)
    v, u = np.nonzero(d)
    keep = (v % stride == 0) & (u % stride == 0)
    v, u = v[keep], u[keep]
    z = d[v, u].astype(np.float64)
    K = f.K_depth
    p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], 1)
    P = (p @ (T_arkit @ _ARKIT_TO_CV).T)[:, :3]
    if not rgb:
        return P
    from PIL import Image
    w, h = f.depth_size
    img = np.asarray(Image.open(f._dir / "rgb" / f"{f._name}.jpg").convert("RGB").resize((w, h)))
    return P, img[v, u]


def cloud_nn(fa: Frame, Ta: np.ndarray, fb: Frame, Tb: np.ndarray) -> float:
    """Median distance from frame a's world cloud to frame b's world cloud (both ARKit-convention poses).
    Median, not mean: the clouds only partly overlap, so non-overlapping points would dominate a mean."""
    return float(np.median(cKDTree(world_cloud(fb, Tb)).query(world_cloud(fa, Ta))[0]))


def path_length(poses: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def gap_nn(scan: Scan, poses: np.ndarray, gap: float = 1.0, n: int = 8) -> float:
    """Mean cloud_nn over ~n frame pairs `gap` seconds apart."""
    ts = np.array([f.timestamp for f in scan.frames])
    pairs = [(i, int(np.searchsorted(ts, ts[i] + gap))) for i in range(len(ts))]
    pairs = [(i, j) for i, j in pairs if j < len(ts)]
    sel = pairs[:: max(1, len(pairs) // n)]
    return float(np.mean([cloud_nn(scan.frames[i], poses[i], scan.frames[j], poses[j]) for i, j in sel]))


def write_jsonl(path: Path, poses, bad, fit, rmse, why=None, method: str = "geometric") -> Path:
    """`method` is the output suffix: a method name or a config label like "features_drop"."""
    out = path / f"frames_refined_{method}.jsonl"
    why = why or [""] * len(poses)
    rows = [json.loads(l) for l in (path / "frames.jsonl").read_text().splitlines() if l.strip()]
    with open(out, "w") as f:
        for r, T, b, ft, rm, w in zip(rows, poses, bad, fit, rmse, why):
            r.update(transform=T.tolist(), refined_bad=bool(b), icp_fitness=float(ft), icp_rmse=float(rm), refined_why=w)
            f.write(json.dumps(r) + "\n")
    return out


def check(root: Path = Path(__file__).resolve().parents[2] / "data" / "scans") -> None:
    from .loader import arkit_runaway
    for n, want in (("20261017_202402", 1), ("20261017_201428", 1), ("20261017_201507", 1), ("20261017_201530", 1),
                    ("20261002_210342", 0), ("20261017_202343", 0), ("20261017_203105", 0), ("20261017_203137", 0)):
        assert arkit_runaway(root / f"{n}_scan") == bool(want), n
    for name, limit in (("20261017_202343_scan", 0.05), ("20261017_202402_scan", 0.3)):
        scan = Scan(root / name, poses="arkit")
        poses = refine_scan(root / name)[0]
        nn = cloud_nn(scan.frames[0], poses[0], scan.frames[-1], poses[-1])
        print(f"{name}: refined first-vs-last NN {nn:.3f} m")
        assert nn < limit, (name, nn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="?")
    ap.add_argument("--method", default="geometric", choices=METHODS)
    ap.add_argument("--table", default="keep", choices=TABLES)
    ap.add_argument("--table-margin", type=float, default=TABLE_MARGIN)
    ap.add_argument("--rot", default="icp", choices=("icp", "arkit"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(); print("ok")
    else:
        p = Path(a.scan)
        tag = a.method + ("" if a.table == "keep" else "_" + a.table) + ("_rotarkit" if a.rot == "arkit" else "")
        print(write_jsonl(p, *refine_scan(p, method=a.method, table=a.table, table_margin_m=a.table_margin, rot=a.rot), method=tag))
