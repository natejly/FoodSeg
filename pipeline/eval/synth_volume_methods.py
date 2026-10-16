"""Per-item volume estimators + container-support construction for the synthetic volume harness.

Everything here works in a z-up frame whose z = 0 is the (GT or fitted) table plane, in metres; volumes
are returned in mL (= cm^3).  The foodscan code is imported read-only and used as-is:

    foodscan.container.ContainerTemplate  from_points / register (ICP) / support
    foodscan.container.Support            points_above / height_map_volume / height_at
    foodscan.volume                       points_above_plane / height_map_volume / convex_hull_volume

Estimators (all take the item's points; NaN = not computable):

    foodscan         ICP-registered container template -> Support height field -> points_above(1 mm)
                     -> Support.height_map_volume (point-only analogue of the mesh-prism integral that
                     run_meal.py uses; the harness has no TSDF mesh).
    foodscan_cinit   same, but ICP is initialised at the centroid of the container-labelled points
                     (`register(init_xy=...)`) instead of run_meal's centroid of all off-plane points.
    foodscan_gtreg   same, but the template is placed at its ground-truth pose (isolates ICP error).
    foodscan_bbox    same height-map integral over *every* above-support scene point inside the item's
                     padded (4 mm) bounding box: what run_meal's `crop_mesh_to_points` (bbox crop of
                     the whole mesh, not a label crop) actually integrates -> shows neighbour leakage.
    hull             convex hull of the item's above-support points plus their drops onto the support.
    voxel            2 mm column fill: footprint closed + hole-filled, top = mean height per cell,
                     bottom = support height at the cell centre.
    table            no container: points above the flat table plane, foodscan.volume.height_map_volume.
    hull_table       foodscan.volume.convex_hull_volume against the flat table plane.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import open3d as o3d  # noqa: E402

from pipeline.foodscan.container import ContainerTemplate, Support  # noqa: E402
from pipeline.foodscan.volume import Plane  # noqa: E402
from pipeline.foodscan.volume import convex_hull_volume as fs_convex_hull_volume  # noqa: E402
from pipeline.foodscan.volume import height_map_volume as fs_height_map_volume  # noqa: E402
from pipeline.foodscan.volume import points_above_plane  # noqa: E402

METHODS = ["foodscan", "voxel", "hull", "foodscan_cinit", "foodscan_gtreg", "foodscan_bbox", "table", "hull_table"]
PRIMARY = ["foodscan", "hull", "voxel"]
MIN_POINTS = 20
CELL = 0.002
MIN_H = 0.001           # run_meal.py: support.points_above(ipts, min_height=0.001)
BBOX_PAD = 0.004        # volume.crop_mesh_to_points default pad
Z_PLANE = Plane(np.array([0.0, 0.0, 1.0]), 0.0)


# --------------------------------------------------------------------------- template / support

def top_envelope(pts: np.ndarray, cell: float = CELL, tol: float = 0.001) -> np.ndarray:
    """Points within `tol` of the highest point of their xy cell: what a top-down scan of the empty
    container sees (drops the underside / outer wall samples of the GT surface)."""
    if len(pts) == 0:
        return pts
    ij = np.floor(pts[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    zmax = np.full(inv.max() + 1, -np.inf)
    np.maximum.at(zmax, inv, pts[:, 2])
    return pts[pts[:, 2] >= zmax[inv] - tol]


def densify_surface(env: np.ndarray, spacing: float = CELL) -> np.ndarray:
    """GT container samples are ~5 mm apart; a real empty-container scan is ~1.5 mm.  Linear-interpolate
    the top envelope onto a regular grid so the oracle template has scan-like density."""
    if len(env) < 50:
        return env
    f = LinearNDInterpolator(env[:, :2], env[:, 2])
    r = np.hypot(env[:, 0], env[:, 1]).max()
    g = np.arange(-r, r + spacing, spacing)
    X, Y = np.meshgrid(g, g)
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    z = f(xy)
    ok = np.isfinite(z)
    return np.c_[xy[ok], z[ok]]


def table_ring(r_in: float, r_out: float = 0.15, spacing: float = 0.003) -> np.ndarray:
    g = np.arange(-r_out, r_out + spacing, spacing)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    m = (r > r_in) & (r < r_out)
    return np.c_[X[m], Y[m], np.zeros(m.sum())]


def make_template(container_pts: np.ndarray, plane: Plane = Z_PLANE) -> tuple[ContainerTemplate, np.ndarray]:
    """Empty-container 'scan' stand-in -> foodscan ContainerTemplate (built by its own from_points).
    Returns (template, xy_centre) where the template -> scene ground-truth pose is translate(centre).
    Input: GT container surface samples in the table-plane frame."""
    env = densify_surface(top_envelope(container_pts))
    if len(env) < 200:
        raise ValueError("container samples too sparse")
    rmax = float(np.hypot(env[:, 0], env[:, 1]).max())
    allp = np.vstack([env, table_ring(rmax - 0.002)])
    dev = np.abs(allp[:, 2]) > 0.0015                 # replicate from_points' centring (same input)
    if dev.sum() < 100:
        raise ValueError("no container geometry above the plane")
    centre = allp[dev, :2].mean(0)
    tmpl = ContainerTemplate.from_points(allp, plane, name="gt_container", band=0.15)
    return tmpl, centre


def gt_pose(centre: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:2, 3] = centre
    return T


def build_supports(container_pts: np.ndarray, scene_pts: np.ndarray, plane: Plane = Z_PLANE,
                   init_xy: np.ndarray | None = None) -> dict:
    """{'icp': Support|None (run_meal-style init), 'icp_c': Support|None (init at `init_xy`), 'gt': Support|None
    (true pose), 'info': {...}} for one scan."""
    out: dict = {"icp": None, "icp_c": None, "gt": None, "info": {}}
    try:
        tmpl, centre = make_template(container_pts, plane)
    except Exception as e:  # noqa: BLE001
        out["info"]["template_error"] = str(e)
        return out
    out["gt"] = tmpl.support(gt_pose(centre), plane)
    try:
        if len(scene_pts) < 200:
            raise ValueError("scene too small to register")
        T = tmpl.register(scene_pts, plane)
        out["icp"] = tmpl.support(T, plane)
        out["info"].update(icp_fitness=float(tmpl.last_fitness), icp_rmse_mm=float(tmpl.last_rmse * 1e3),
                           covered_fraction=float(getattr(tmpl, "covered_fraction", np.nan)))
        out["info"]["icp_shift_mm"] = float(np.linalg.norm((T - gt_pose(centre))[:2, 3]) * 1e3)
        out["info"]["support_rmse_mm"] = support_rmse_mm(out["icp"], out["gt"], container_pts)
        if init_xy is not None:
            T = tmpl.register(scene_pts, plane, init_xy=init_xy)
            out["icp_c"] = tmpl.support(T, plane)
            out["info"].update(icp_c_fitness=float(tmpl.last_fitness),
                               icp_c_shift_mm=float(np.linalg.norm((T - gt_pose(centre))[:2, 3]) * 1e3),
                               support_c_rmse_mm=support_rmse_mm(out["icp_c"], out["gt"], container_pts))
    except Exception as e:  # noqa: BLE001
        out["info"]["icp_error"] = str(e)
    return out


def support_rmse_mm(a: Support, b: Support, container_pts: np.ndarray) -> float:
    """RMS difference (mm) of two supports' heights over the container's xy footprint."""
    env = top_envelope(container_pts)
    if len(env) == 0:
        return float("nan")
    p = np.c_[env[:, :2], np.zeros(len(env))]
    return float(np.sqrt(np.mean((a.height_at(p) - b.height_at(p)) ** 2)) * 1e3)


# --------------------------------------------------------------------------- estimators

def _finite_pts(p: np.ndarray) -> np.ndarray:
    return p[np.isfinite(p).all(1)] if len(p) else p


def voxel_fill_volume(pts: np.ndarray, support: Support | None, cell: float = CELL, close_iter: int = 2) -> float:
    """Column fill on a `cell` grid between the support height and the item's mean top height."""
    if len(pts) == 0:
        return 0.0
    o = pts[:, :2].min(0)
    ij = np.floor((pts[:, :2] - o) / cell).astype(int)
    shape = ij.max(0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    n = shape[0] * shape[1]
    cnt = np.bincount(flat, minlength=n).reshape(shape)
    top = np.bincount(flat, weights=pts[:, 2], minlength=n).reshape(shape)
    obs = cnt > 0
    top = np.where(obs, top / np.maximum(cnt, 1), 0.0)
    pad = close_iter + 1
    m = np.pad(obs, pad)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=np.ones((3, 3), bool), iterations=close_iter))
    m = m[pad:-pad, pad:-pad]
    # heights for filled cells: nearest observed cell
    _, (iy, ix) = ndimage.distance_transform_edt(~obs, return_indices=True)
    top = top[iy, ix]
    ii, jj = np.nonzero(m)
    cx = o[0] + (ii + 0.5) * cell
    cy = o[1] + (jj + 0.5) * cell
    base = support.height_at(np.c_[cx, cy, np.zeros(len(cx))]) if support is not None else np.zeros(len(cx))
    h = np.clip(top[ii, jj] - base, 0, None)
    return float(h.sum() * cell * cell * 1e6)


def hull_support_volume(pts: np.ndarray, support: Support | None) -> float:
    """Convex hull of the points plus their drops onto the support surface."""
    if len(pts) < 4:
        return float("nan")
    base_z = support.height_at(np.c_[pts[:, :2], np.zeros(len(pts))]) if support is not None else np.zeros(len(pts))
    base = np.c_[pts[:, :2], np.minimum(base_z, pts[:, 2])]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pts, base]).astype(np.float64)))
    try:
        hull, _ = pc.compute_convex_hull()
        return float(hull.get_volume() * 1e6)
    except Exception:  # noqa: BLE001  (qhull on coplanar / collinear points)
        return float("nan")


def estimate_item(pts: np.ndarray, scene: np.ndarray, supports: dict, plane: Plane = Z_PLANE,
                  min_points: int = MIN_POINTS) -> dict[str, float]:
    """All methods for one item.  `pts` = the item's points, `scene` = the whole cloud (for foodscan_bbox)."""
    out = {m: float("nan") for m in METHODS}
    pts = _finite_pts(pts)
    if len(pts) < min_points:
        return out
    sup, sup_gt = supports.get("icp"), supports.get("gt")

    def fs(s: Support | None, p: np.ndarray) -> float:
        if s is None:
            return float("nan")
        a = s.points_above(p, min_height=MIN_H)
        return s.height_map_volume(a) if len(a) else 0.0

    out["foodscan"] = fs(sup, pts)
    out["foodscan_cinit"] = fs(supports.get("icp_c"), pts)
    out["foodscan_gtreg"] = fs(sup_gt, pts)
    if sup is not None:
        lo, hi = pts.min(0) - BBOX_PAD, pts.max(0) + BBOX_PAD
        inb = ((scene >= lo) & (scene <= hi)).all(1)
        out["foodscan_bbox"] = fs(sup, scene[inb])
        a = sup.points_above(pts, min_height=MIN_H)
        out["voxel"] = voxel_fill_volume(a, sup) if len(a) else 0.0
        out["hull"] = hull_support_volume(a, sup) if len(a) else 0.0
    a = points_above_plane(pts, plane, min_height=MIN_H)
    out["table"] = fs_height_map_volume(a, plane) if len(a) else 0.0
    try:
        out["hull_table"] = fs_convex_hull_volume(a, plane) if len(a) >= 4 else 0.0
    except Exception:  # noqa: BLE001
        pass
    return out
