"""Tests for the RQ1 harness (pipeline/eval/rq1_geometry.py, rq1_frame_vs_scan.py).

    cd /home/kyle/Desktop/FoodSeg && conda run -n foodseg python -m pytest pipeline/tests/test_rq1.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.eval import rq1_frame_vs_scan as RQ  # noqa: E402
from pipeline.eval import rq1_geometry as G  # noqa: E402

W, H = 256, 192
CAM_Z = 0.40


def render_hemisphere(radius: float, fx: float = 420.0, cam_xyz=(0.0, 0.0, CAM_Z)):
    """Fake z-depth frame of a hemisphere (centre on the plane z = 0) over the table, seen from a camera looking
    straight down (ARKit camera with identity rotation looks along -z).  Returns depth, K, T, hit-mask."""
    K = np.array([[fx, 0, W / 2 - 0.5], [0, fx, H / 2 - 0.5], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = cam_xyz
    v, u = np.mgrid[:H, :W]
    d = np.stack([(u - K[0, 2]) / fx, -(v - K[1, 2]) / fx, -np.ones_like(u, float)], -1)   # ARKit ray dirs, z-depth 1
    o = np.array(cam_xyz)
    # sphere centred at origin
    a = (d ** 2).sum(-1)
    b = 2 * (d @ o)
    c = o @ o - radius ** 2
    disc = b * b - 4 * a * c
    t_s = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    hit_z = o[2] + t_s * d[..., 2]
    sphere = np.isfinite(t_s) & (hit_z >= 0)
    t_plane = -o[2] / d[..., 2]
    t = np.where(sphere, t_s, t_plane)
    return t.astype(np.float32), K, T, sphere        # ray param == z-depth because d_z = -1


def test_hemisphere_plane_closing_within_5pct():
    r = 0.04
    depth, K, T, sph = render_hemisphere(r)
    pts, v, u = G.backproject(depth, None, K, T, np.eye(3), np.zeros(3))
    item = pts[sph[v, u]]
    assert len(item) > 500
    vol = G.item_volumes(item, None, None)["plane"]
    true = 2 / 3 * np.pi * r ** 3 * 1e6
    assert abs(vol - true) / true < 0.05, (vol, true)


def test_hemisphere_container_closing_on_flat_support():
    r = 0.04
    depth, K, T, sph = render_hemisphere(r)
    pts, v, u = G.backproject(depth, None, K, T, np.eye(3), np.zeros(3))
    lab = np.where(sph[v, u], 1, 254).astype(np.int32)
    # table pixels play the container; a flat container surface at z = 0 gives the same answer as the plane
    sup = G.fitted_support(pts[lab == 254])
    assert sup is not None
    vol = G.item_volumes(pts[lab == 1], sup, sup)
    true = 2 / 3 * np.pi * r ** 3 * 1e6
    for m in ("plane", "container", "gtcont"):
        assert abs(vol[m] - true) / true < 0.05, (m, vol[m], true)


def test_low_confidence_pixels_are_dropped():
    depth, K, T, sph = render_hemisphere(0.04)
    conf = np.full(depth.shape, 2, np.uint8)
    conf[:, : W // 2] = 0
    pts, v, u = G.backproject(depth, conf, K, T, np.eye(3), np.zeros(3), min_conf=1)
    assert (u >= W // 2).all() and len(pts) == (H * W) // 2


def test_even_indices():
    assert G.even_indices(24, 1) == [0]
    assert G.even_indices(24, 2) == [0, 12]
    assert G.even_indices(24, 4) == [0, 6, 12, 18]
    assert G.even_indices(24, 8) == [0, 3, 6, 9, 12, 15, 18, 21]
    assert G.even_indices(24, 12) == list(range(0, 24, 2))
    assert G.even_indices(24, 24) == list(range(24))
    assert G.even_indices(8, 4) == [0, 2, 4, 6]
    assert G.even_indices(24, 4, offset=1) == [1, 7, 13, 19]
    assert G.even_indices(8, 99) == list(range(8))        # k clipped to n


def test_topdown_selection():
    R = np.eye(3)
    frames = []
    for tilt in (60, 20, 5, 40):
        a = np.radians(tilt)
        # camera looking down, tilted by `tilt` around x: forward (ARKit -z column) = (0, sin, -cos)
        fwd = np.array([0.0, np.sin(a), -np.cos(a)])
        up = np.array([0.0, np.cos(a), np.sin(a)])
        right = np.cross(fwd, up)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = right, up, -fwd
        frames.append({"T": T})
    assert G.topdown_index(frames, R) == 2
    assert abs(G.tilt_deg(G.topdown_scores(frames, R)[2]) - 5) < 1e-6


def test_voxel_fuse_keeps_labels_and_merges_duplicates():
    p = np.array([[0.0, 0, 0], [0.0004, 0, 0], [0.0, 0, 0.0004], [0.1, 0, 0], [0.1, 0, 0]])
    lab = np.array([1, 1, 2, 1, 1])
    q, ql = G.voxel_fuse(p, lab, 0.002)
    assert len(q) == 3 and sorted(ql.tolist()) == [1, 1, 2]


def test_aggregate_and_bootstrap_shapes():
    rows = []
    rng = np.random.default_rng(0)
    for s in range(6):
        for i in range(1, 4):
            true = 20.0 * i
            for arm, noise in (("overhead", 0.6), ("K24", 0.1)):
                rows.append(dict(batch="b", scan=f"s{s}", ok=1, arm=arm, draw=0, item_id=i, kind="piece", category="x",
                                 container="plate", true_ml=true, n_pts=100, vis=0.5, sup_src="fit", tilt=0.0,
                                 plane=true * (1 + noise * rng.choice([-1, 1])), container_m=true, hull=true, gtcont=true))
    agg = RQ.aggregate(rows, "plane")
    b = RQ.paired_bootstrap(agg, "b", "overhead", "K24", n_boot=200)
    assert b["n"] == 18 and b["d_med_ape"] > 30 and b["lo"] > 0
