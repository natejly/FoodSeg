"""Per-item ground-truth statistics derived from the GT surface samples (gt/points.npz).

Shared by finalize (new scans) and seg_export (backfills older scans, adds fusion-dependent stats).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from .density import mass_from_volume

TABLE_ID = 252
CONTAINER_ID = 254
CELL = 0.002


def table_points(W2A: np.ndarray, container_xyz: np.ndarray, r_out: float = 0.27, cell: float = 0.002) -> np.ndarray:
    """Jittered-grid samples of the table plane (dish-frame z = 0) outside the container footprint,
    returned in the ARKit frame. `container_xyz` are the container's GT samples (ARKit)."""
    rng = np.random.default_rng(1)
    g = np.arange(-r_out, r_out, cell)
    X, Y = np.meshgrid(g, g)
    xy = np.stack([X.ravel(), Y.ravel()], 1) + rng.uniform(0, cell, (X.size, 2))
    loc_c = (container_xyz - W2A[:3, 3]) @ W2A[:3, :3]
    r_c = np.hypot(loc_c[:, 0], loc_c[:, 1]).max() if len(loc_c) else 0.0
    r = np.hypot(xy[:, 0], xy[:, 1])
    xy = xy[(r < r_out) & (r > r_c)]
    loc = np.concatenate([xy, np.zeros((len(xy), 1))], 1)
    return (loc @ W2A[:3, :3].T + W2A[:3, 3]).astype(np.float32)


def with_table(xyz: np.ndarray, label: np.ndarray, W2A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append table samples (label 252) unless the cloud already has them."""
    if (label == TABLE_ID).any():
        return xyz, label
    t = table_points(W2A, xyz[label == CONTAINER_ID])
    return (np.concatenate([xyz, t]).astype(np.float32),
            np.concatenate([label, np.full(len(t), TABLE_ID, np.uint8)]))


def _column_volume_ml(loc: np.ndarray, plane_z: float) -> float:
    """Volume (mL) of the item above z = plane_z in the dish frame, by filling each 2 mm xy column
    between its lowest and highest surface sample (approximate for concave items)."""
    if not len(loc):
        return 0.0
    ij = np.floor(loc[:, :2] / CELL).astype(np.int64)
    key = (ij[:, 0] - ij[:, 0].min()) * (ij[:, 1].max() - ij[:, 1].min() + 1) + (ij[:, 1] - ij[:, 1].min())
    order = np.argsort(key)
    key, z = key[order], loc[order, 2]
    first = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    zmin, zmax = np.minimum.reduceat(z, first), np.maximum.reduceat(z, first)
    h = np.clip(zmax - np.maximum(zmin, plane_z), 0, None)
    return float(h.sum() * CELL * CELL * 1e6)


def container_support_fn(cont_loc: np.ndarray, radius: float = 0.006, tol: float = 0.003):
    """support_fn(xy, zmin) -> support height per column (dish frame, table at z = 0): the highest container
    surface sample within `radius` (xy) that is not above the item's lowest sample in that column (+tol),
    else the table plane (items lying off the container, or on the table)."""
    if not len(cont_loc):
        return lambda xy, zmin: np.zeros(len(xy))
    tree = cKDTree(cont_loc[:, :2])
    cz = cont_loc[:, 2]

    def fn(xy, zmin):
        out = np.zeros(len(xy))
        for i, nb in enumerate(tree.query_ball_point(xy, radius)):
            if nb:
                z = cz[nb]
                z = z[z <= zmin[i] + tol]
                if len(z):
                    out[i] = max(0.0, z.max())
        return out
    return fn


def envelope_volume(item_loc: np.ndarray, support_fn, grid: float = CELL) -> tuple[float, float]:
    """(envelope_ml, hull_ml) from ALL GT surface samples of one item (dish frame, z up, table at 0).
    Envelope: footprint on a `grid` raster (closed by 1 cell), top height per cell (max z; empty footprint
    cells take a 3x3 max-filter of their neighbours), integrated from the support height to the top.
    This is what a scan sees (top surface closed to the container): it includes voids in piles/leaves.
    Hull: convex hull volume of the samples."""
    if len(item_loc) < 4:
        return 0.0, 0.0
    ij = np.floor(item_loc[:, :2] / grid).astype(np.int64)
    ij -= ij.min(0) - 2
    shape = tuple(ij.max(0) + 3)
    top = np.full(shape, -np.inf)
    bot = np.full(shape, np.inf)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), item_loc[:, 2])
    np.minimum.at(bot, (ij[:, 0], ij[:, 1]), item_loc[:, 2])
    foot = np.isfinite(top)
    st = np.ones((3, 3), bool)
    foot = ndimage.binary_erosion(ndimage.binary_dilation(foot, st), st)
    filt = ndimage.maximum_filter(top, size=3)
    filt_bot = ndimage.minimum_filter(bot, size=3)
    top = np.where(np.isfinite(top), top, filt)
    bot = np.where(np.isfinite(bot), bot, filt_bot)
    foot &= np.isfinite(top)
    # the max of a sloped surface over a cell sits above its mean by ~ (|gx|+|gy|)*cell/2: remove that bias
    gx, gy = np.gradient(np.where(foot, top, np.nan), grid)
    slope = np.nan_to_num(np.abs(gx) + np.abs(gy))
    top = np.maximum(top - 0.5 * grid * slope, bot)
    ci = np.argwhere(foot)
    xy = (ci + np.floor(item_loc[:, :2].min(0) / grid) - 2 + 0.5) * grid
    sup = support_fn(xy, bot[foot])
    h = np.clip(top[foot] - sup, 0, None)
    env = float(h.sum() * grid * grid * 1e6)
    try:
        hull = float(ConvexHull(item_loc).volume * 1e6)
    except Exception:       # degenerate (coplanar) sample set
        hull = 0.0
    return env, hull


def item_gt_stats(gt: dict, xyz: np.ndarray, label: np.ndarray) -> None:
    """Fill mass_g/mass_source, bbox_arkit, n_gt_points, volume_above_{container,table}_ml,
    volume_envelope_ml, hull_volume_ml, envelope_ratio, bulk_density_g_ml in place."""
    W2A = np.array(gt["world_to_arkit"])
    cont = xyz[label == CONTAINER_ID]
    cl = (cont - W2A[:3, 3]) @ W2A[:3, :3] if len(cont) else np.zeros((0, 3))
    rim_z = float(cl[:, 2].max()) if len(cl) else 0.0
    sup_fn = container_support_fn(cl)
    for it in gt["items"]:
        if it.get("mass_g") is None or "mass_source" not in it:
            it["mass_g"], it["mass_source"] = mass_from_volume(it)
        for k in ("volume_envelope_ml", "hull_volume_ml", "envelope_ratio", "bulk_density_g_ml"):
            it.setdefault(k, None)
        p = xyz[label == it["id"]]
        it["n_gt_points"] = int(len(p))
        if not len(p):
            continue
        it["bbox_arkit"] = [p.min(0).tolist(), p.max(0).tolist()]
        loc = (p - W2A[:3, 3]) @ W2A[:3, :3]
        it["volume_above_container_ml"] = _column_volume_ml(loc, rim_z)
        it["volume_above_table_ml"] = _column_volume_ml(loc, 0.0)
        it["volume_envelope_ml"], it["hull_volume_ml"] = envelope_volume(loc, sup_fn)
        v = it.get("volume_ml")
        it["envelope_ratio"] = it["volume_envelope_ml"] / v if v else None
        if it.get("mass_g") is not None and it["volume_envelope_ml"] > 0:
            it["bulk_density_g_ml"] = it["mass_g"] / it["volume_envelope_ml"]
