import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.boundary_refine import Graph, RefineParams, refine_labels  # noqa: E402


def two_blobs(seed=0, jitter=0.004):
    """Flat 2 mm grid, 120 x 60 mm; left half (x<60 mm) item 1 with red-ish colour, right half item 2 blue-ish.
    The lift boundary is jittered by +-4 mm and has votes that are noisy near it."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(0, 0.12, 0.002), np.arange(0, 0.06, 0.002))
    xyz = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    gt = np.where(xyz[:, 0] < 0.06, 1, 2)
    rgb = np.where((gt == 1)[:, None], [[0.8, 0.2, 0.2]], [[0.2, 0.2, 0.8]]) + rng.normal(0, 0.01, (len(xyz), 3))
    normal = np.tile([0, 0, 1.0], (len(xyz), 1))
    # jittered lift boundary per row (y) -> wrong labels near the true edge
    off = np.round(rng.uniform(-jitter, jitter, 30) / 0.002) * 0.002
    row = np.round(xyz[:, 1] / 0.002).astype(int)
    lift = np.where(xyz[:, 0] < 0.06 + off[row], 1, 2)
    return xyz, rgb, normal, gt, lift


def votes_for(lift, conf=0.6):
    v = np.zeros((len(lift), 3), np.float32)           # columns ids [0, 1, 2]
    v[np.arange(len(lift)), lift] = conf
    v[np.arange(len(lift)), 3 - lift] = 1 - conf
    return v, np.array([0, 1, 2])


def test_recovers_jittered_boundary():
    xyz, rgb, nrm, gt, lift = two_blobs()
    assert (lift != gt).sum() > 20
    votes, ids = votes_for(lift)
    g = Graph(xyz, rgb, nrm, xyz[:, 2], 12, 8)
    out, info = refine_labels(lift, votes, ids, g, RefineParams(w=3.0, s_c=15, s_n=30, s_z=0, band_mm=6.0))
    assert (out != gt).sum() < 0.25 * (lift != gt).sum()
    assert info["n_changed"] > 0


def test_band_leaves_far_points_untouched():
    xyz, rgb, nrm, gt, lift = two_blobs(seed=1)
    lift = lift.copy()
    far = np.flatnonzero(np.abs(xyz[:, 0] - 0.06) > 0.02)
    flip = far[::97]
    lift[flip] = 3 - lift[flip]                         # isolated errors far from any boundary stay
    # make them genuine boundaries of their own (band is relative to the lift labels), so check outside every band
    votes, ids = votes_for(lift)
    g = Graph(xyz, rgb, nrm, xyz[:, 2], 12, 8)
    p = RefineParams(w=3.0, s_c=15, s_n=30, s_z=0, band_mm=3.0)
    out, _ = refine_labels(lift, votes, ids, g, p)
    from foodscan.boundary_refine import _band
    free = _band(g, lift, 3.0)
    assert (out[~free] == lift[~free]).all()
    out_none, _ = refine_labels(lift, votes, ids, g, RefineParams(w=3.0, s_c=15, s_n=30, s_z=0, band_mm=0.0001))
    assert (out_none == lift).all() or (out_none != lift).sum() < 5


def test_grid_union_contains_default_and_tag40():
    from foodscan.boundary_refine import RefineParams, grid_union, tag40
    g = grid_union()
    assert RefineParams() in g and len(g) == len(set(g))
    assert tag40("v13/s000003") == "v13__s000003"
