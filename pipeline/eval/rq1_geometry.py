"""Geometry helpers for the RQ1 harness (single overhead RGB-D frame vs multi-frame orbit scan).

Read-only reuse of foodscan (Support, volume) and the synth_volume harness (container template, hull,
top envelope).  All clouds are in the *dish frame* (z up, z = 0 on the table, metres) so the GT table plane
is Z_PLANE.  Frames use the exported (noisy) ARKit poses straight from frames.jsonl; those line up with
gt/seg_points.npz to ~2 mm (the seg export's ICP correction is not applied).

Frame labels: v12 `gt/seg_ids/NNNNNN.png` (item ids 1..251, 253 clutter, 254 container, all at RGB resolution);
v9/v10 have no seg_ids, so `gt/masks/NNNNNN.png` (item ids) + `gt/container/NNNNNN.png` (255 = container) are
merged into the same id convention.  Labels are sampled at the depth pixel centres (nearest).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.eval import synth_volume_methods as M  # noqa: E402
from pipeline.foodscan.container import Support  # noqa: E402
from pipeline.foodscan.volume import height_map_volume as fs_hmv  # noqa: E402
from pipeline.foodscan.volume import points_above_plane  # noqa: E402

Z_PLANE = M.Z_PLANE
CONTAINER_ID = 254
VOXEL = 0.002
METHODS = ["plane", "container", "hull", "gtcont"]
MIN_CONT_PTS = 300


# ------------------------------------------------------------------------------------------- frames

def load_frames(scan_dir: Path) -> list[dict]:
    """frames.jsonl -> [{index, K (depth res, pixel-centre convention), T (ARKit c2w), size (w, h)}]."""
    out = []
    for line in Path(scan_dir, "frames.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        K = np.array(r["intrinsics"], np.float64)
        dw, dh = r["depthWidth"], r["depthHeight"]
        K = np.diag([dw / r["imageWidth"], dh / r["imageHeight"], 1.0]) @ K
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        out.append({"index": r["index"], "K": K, "T": np.array(r["transform"], np.float64), "size": (dw, dh)})
    return out


def backproject(depth: np.ndarray, conf: np.ndarray | None, K: np.ndarray, T_c2w: np.ndarray,
                R: np.ndarray, t: np.ndarray, min_conf: int = 1, dmin: float = 0.1, dmax: float = 3.0):
    """Depth (h, w, z-depth, m) -> dish-frame points (N, 3) and the (v, u) pixel of each point.
    Camera convention is ARKit (+y up, -z forward); dish = (p_arkit - t) @ R."""
    h, w = depth.shape
    ok = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    if conf is not None:
        ok &= conf >= min_conf
    v, u = np.nonzero(ok)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, -y, -z], 1)
    pw = pc @ T_c2w[:3, :3].T + T_c2w[:3, 3]
    return (pw - t) @ R, v, u


def sample_labels(lab_full: np.ndarray, v: np.ndarray, u: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest label at the centre of each depth pixel from a full-resolution id image."""
    H, W = lab_full.shape
    rr = np.minimum(((v + 0.5) * H / size[1]).astype(int), H - 1)
    cc = np.minimum(((u + 0.5) * W / size[0]).astype(int), W - 1)
    return lab_full[rr, cc].astype(np.int32)


def load_label_image(scan_dir: Path, idx: int) -> np.ndarray:
    d = Path(scan_dir) / "gt"
    name = f"{idx:06d}.png"
    if (d / "seg_ids" / name).exists():
        return np.array(Image.open(d / "seg_ids" / name))
    m = np.array(Image.open(d / "masks" / name)).astype(np.uint8)
    c = d / "container" / name
    if c.exists():
        cm = np.array(Image.open(c)) > 0
        m = np.where((m == 0) & cm, CONTAINER_ID, m).astype(np.uint8)
    return m


def clean_mask(depth: np.ndarray, lab_grid: np.ndarray | None, jump: float = 0.01) -> np.ndarray:
    """Pixels to keep after a per-frame cleaning that any real pipeline would do before fusing:
    drop pixels on a depth discontinuity (> `jump` m to a 4-neighbour: LiDAR edge flyers / bleed) and pixels
    whose label differs from a 4-neighbour (mixed boundary pixels, where the oracle label is unreliable)."""
    keep = np.ones(depth.shape, bool)
    d = np.where(np.isfinite(depth), depth, 0.0)
    for ax in (0, 1):
        for sh in (1, -1):
            nb = np.roll(d, sh, axis=ax)
            bad = (np.abs(d - nb) > jump) & (nb > 0) & (d > 0)
            if ax == 0:
                bad[0 if sh == 1 else -1, :] = False
            else:
                bad[:, 0 if sh == 1 else -1] = False
            keep &= ~bad
            if lab_grid is not None:
                nl = np.roll(lab_grid, sh, axis=ax)
                diff = nl != lab_grid
                if ax == 0:
                    diff[0 if sh == 1 else -1, :] = False
                else:
                    diff[:, 0 if sh == 1 else -1] = False
                keep &= ~diff
    return keep


def gt_target(gt_xyz: np.ndarray, voxel: float = 0.003) -> o3d.geometry.PointCloud:
    """Noise-free GT surface (dish frame, all labels) as an ICP target with normals."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_xyz)).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
    return pc


def icp_align(pts: np.ndarray, target: o3d.geometry.PointCloud, max_dist: float = 0.015, iters: int = 40):
    """Oracle pose refinement: rigid point-to-plane ICP of one frame's cloud onto the GT surface.  Returns
    (aligned points, 4x4 correction, fitness).  Upper bound of what foodscan.refine_poses could recover."""
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(0.004)
    res = o3d.pipelines.registration.registration_icp(
        src, target, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    T = np.asarray(res.transformation)
    return pts @ T[:3, :3].T + T[:3, 3], T, float(res.fitness)


def frame_cloud(scan_dir: Path, fr: dict, R: np.ndarray, t: np.ndarray, min_conf: int = 1,
                depth_dir: str = "depth", clean: bool = True, target: o3d.geometry.PointCloud | None = None):
    """One frame -> (pts dish (N,3), labels (N,)) from depth/conf + oracle ids (see `clean_mask`)."""
    i, (w, h) = fr["index"], fr["size"]
    d = np.fromfile(Path(scan_dir) / depth_dir / f"{i:06d}.f32", "<f4").reshape(h, w)
    cp = Path(scan_dir) / "conf" / f"{i:06d}.u8"
    c = np.fromfile(cp, np.uint8).reshape(h, w) if cp.exists() else None
    if clean:
        vv, uu = np.mgrid[:h, :w]
        grid = sample_labels(load_label_image(scan_dir, i), vv.ravel(), uu.ravel(), fr["size"]).reshape(h, w)
        k = clean_mask(d, grid)
        d = np.where(k, d, 0.0).astype(np.float32)
    pts, v, u = backproject(d, c, fr["K"], fr["T"], R, t, min_conf)
    lab = sample_labels(load_label_image(scan_dir, i), v, u, fr["size"])
    if target is not None and len(pts) > 200:
        pts, _, _ = icp_align(pts, target)
    return pts, lab


# ------------------------------------------------------------------------------------------- selection

def even_indices(n: int, k: int, offset: int = 0) -> list[int]:
    """k frame indices evenly spaced over n orbit frames (phase `offset` frames)."""
    k = max(1, min(k, n))
    return sorted({int((round(i * n / k) + offset) % n) for i in range(k)})


def topdown_scores(frames: list[dict], R: np.ndarray) -> np.ndarray:
    """cos of the angle between each camera's viewing axis and straight down (dish -z); 1 = overhead."""
    s = []
    for f in frames:
        fwd_ark = -f["T"][:3, 2]
        s.append(float(-(R.T @ fwd_ark)[2]))
    return np.array(s)


def topdown_index(frames: list[dict], R: np.ndarray) -> int:
    return int(np.argmax(topdown_scores(frames, R)))


def tilt_deg(score: float) -> float:
    return float(np.degrees(np.arccos(np.clip(score, -1, 1))))


# ------------------------------------------------------------------------------------------- fusion

def voxel_fuse(pts: np.ndarray, lab: np.ndarray, voxel: float = VOXEL):
    """Per-(voxel, label) centroid downsample.  No TSDF."""
    if len(pts) == 0:
        return pts, lab
    ij = np.floor(pts / voxel).astype(np.int64)
    key = np.c_[ij, lab.astype(np.int64)]
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    out = np.stack([np.bincount(inv, weights=pts[:, a], minlength=n) / cnt for a in range(3)], 1)
    first = np.zeros(n, np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return out, lab[first]


# ------------------------------------------------------------------------------------------- supports

def fitted_support(cont_pts: np.ndarray, min_pts: int = MIN_CONT_PTS) -> Support | None:
    """Height field of the container from the *visible* container points of an arm's cloud: top envelope,
    linear interpolation over the visible hull (holes under food are interpolated), plane (0) elsewhere."""
    cont_pts = cont_pts[np.isfinite(cont_pts).all(1)] if len(cont_pts) else cont_pts
    if len(cont_pts) < min_pts:
        return None
    env = M.top_envelope(cont_pts, tol=0.002)
    try:
        dense = M.densify_surface(env)
    except Exception:  # noqa: BLE001  (degenerate triangulation)
        return None
    if len(dense) < 200:
        return None
    return Support.from_points(dense, Z_PLANE, cell=0.002)


def gt_support(container_pts_gt: np.ndarray) -> Support | None:
    try:
        tmpl, centre = M.make_template(container_pts_gt)
    except Exception:  # noqa: BLE001
        return None
    return tmpl.support(M.gt_pose(centre), Z_PLANE)


# ------------------------------------------------------------------------------------------- volumes

def _fs_vol(sup: Support | None, pts: np.ndarray) -> float:
    if sup is None:
        return float("nan")
    a = sup.points_above(pts, min_height=M.MIN_H)
    return sup.height_map_volume(a) if len(a) else 0.0


def item_volumes(pts: np.ndarray, sup_fit: Support | None, sup_gt: Support | None,
                 min_points: int = M.MIN_POINTS) -> dict[str, float]:
    """plane: height map vs the table plane; container: vs the fitted container support; hull: convex hull
    of the points plus their drops onto the fitted support; gtcont: vs the GT container surface (diagnostic)."""
    out = {m: float("nan") for m in METHODS}
    pts = pts[np.isfinite(pts).all(1)] if len(pts) else pts
    if len(pts) < min_points:
        return out
    a = points_above_plane(pts, Z_PLANE, min_height=M.MIN_H)
    out["plane"] = fs_hmv(a, Z_PLANE) if len(a) else 0.0
    out["container"] = _fs_vol(sup_fit, pts)
    if sup_fit is not None:
        b = sup_fit.points_above(pts, min_height=M.MIN_H)
        out["hull"] = M.hull_support_volume(b, sup_fit) if len(b) >= 4 else 0.0
    out["gtcont"] = _fs_vol(sup_gt, pts)
    return out


def visibility(gt_surface: np.ndarray, arm_pts: np.ndarray, tol: float = 0.003, max_gt: int = 4000,
               seed: int = 0) -> float:
    """Fraction of the item's GT surface samples within `tol` of the arm's item points."""
    from scipy.spatial import cKDTree
    if len(gt_surface) == 0:
        return float("nan")
    if len(arm_pts) == 0:
        return 0.0
    if len(gt_surface) > max_gt:
        gt_surface = gt_surface[np.random.default_rng(seed).choice(len(gt_surface), max_gt, replace=False)]
    d, _ = cKDTree(arm_pts).query(gt_surface, distance_upper_bound=tol)
    return float(np.mean(np.isfinite(d)))
