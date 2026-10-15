"""Per-item ground-truth statistics derived from the GT surface samples (gt/points.npz).

Shared by finalize (new scans) and seg_export (backfills older scans, adds fusion-dependent stats).
"""
from __future__ import annotations

import numpy as np

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


def item_gt_stats(gt: dict, xyz: np.ndarray, label: np.ndarray) -> None:
    """Fill mass_g/mass_source, bbox_arkit, n_gt_points, volume_above_{container,table}_ml in place."""
    W2A = np.array(gt["world_to_arkit"])
    cont = xyz[label == CONTAINER_ID]
    cl = (cont - W2A[:3, 3]) @ W2A[:3, :3] if len(cont) else np.zeros((0, 3))
    rim_z = float(cl[:, 2].max()) if len(cl) else 0.0
    for it in gt["items"]:
        if it.get("mass_g") is None or "mass_source" not in it:
            it["mass_g"], it["mass_source"] = mass_from_volume(it)
        p = xyz[label == it["id"]]
        it["n_gt_points"] = int(len(p))
        if not len(p):
            continue
        it["bbox_arkit"] = [p.min(0).tolist(), p.max(0).tolist()]
        loc = (p - W2A[:3, 3]) @ W2A[:3, :3]
        it["volume_above_container_ml"] = _column_volume_ml(loc, rim_z)
        it["volume_above_table_ml"] = _column_volume_ml(loc, 0.0)
