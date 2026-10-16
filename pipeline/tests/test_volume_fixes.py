"""Unit tests for the volume-step fixes found by the synthetic volume harness.

Run: conda run -n foodseg python -m pytest pipeline/tests/test_volume_fixes.py
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from pipeline.foodscan.container import ContainerTemplate, Support
from pipeline.foodscan.volume import (Plane, crop_mesh_by_label, crop_mesh_to_points, fit_support_plane_info,
                                      label_mesh_vertices, mesh_prism_volume)

Z = Plane(np.array([0.0, 0.0, 1.0]), 0.0)


def _flat_support() -> Support:
    r = np.random.default_rng(0).uniform(-0.05, 0.05, (2000, 2))
    return Support.from_points(np.c_[r, np.zeros(len(r))], Z)


def cylinder(R, H, sp=0.0015, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(-R, R + sp, sp)
    X, Y = np.meshgrid(g, g)
    m = X ** 2 + Y ** 2 <= R * R
    top = np.c_[X[m], Y[m], np.full(m.sum(), H)]
    A, Zs = np.meshgrid(np.arange(0, 2 * np.pi, sp / R), np.arange(0.0015, H, sp))
    side = np.c_[R * np.cos(A).ravel(), R * np.sin(A).ravel(), Zs.ravel()]
    p = np.vstack([top, side])
    p[:, :2] += rng.uniform(-sp / 5, sp / 5, (len(p), 2))
    return p, np.pi * R * R * H * 1e6


def hemisphere(R, sp=0.0015, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(int(2 * np.pi * R * R / sp ** 2), 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v[:, 2] = np.abs(v[:, 2])
    p = v * R
    return p[p[:, 2] > 0.001], 2 / 3 * np.pi * R ** 3 * 1e6


@pytest.mark.parametrize("name,make", [("cyl3cm", lambda: cylinder(0.015, 0.03)),
                                        ("cyl6cm", lambda: cylinder(0.03, 0.03)),
                                        ("hemi4cm", lambda: hemisphere(0.02)),
                                        ("hemi6cm", lambda: hemisphere(0.03))])
def test_height_map_volume_within_3pct(name, make):
    p, true = make()
    sup = _flat_support()
    new = sup.height_map_volume(p)
    assert abs(new / true - 1) < 0.03, (name, new, true)


def test_height_map_volume_small_cylinder_and_legacy_flag():
    """1 cm cylinder: the new estimate is closer than the legacy 'every touched cell counts' one
    (legacy +4..13 % depending on sampling; hemispheres: legacy about -10 %, from unfilled holes)."""
    p, true = cylinder(0.005, 0.02)
    sup = _flat_support()
    new, old = sup.height_map_volume(p), sup.height_map_volume(p, legacy=True)
    assert abs(new / true - 1) < 0.05
    ph, th = hemisphere(0.03)
    assert abs(sup.height_map_volume(ph) / th - 1) < abs(sup.height_map_volume(ph, legacy=True) / th - 1)


def test_height_map_volume_fills_interior_hole():
    p, true = cylinder(0.03, 0.03)
    keep = np.hypot(p[:, 0], p[:, 1]) > 0.008          # LiDAR dropout in the top centre
    keep |= p[:, 2] < 0.029
    sup = _flat_support()
    assert abs(sup.height_map_volume(p[keep]) / true - 1) < 0.03
    assert sup.height_map_volume(p[keep], legacy=True) / true < 0.97


# ---- label crop ---------------------------------------------------------------------------------

def _two_boxes():
    """Touching blocks: A is 1x1x1 cm (x in [0, .01]), B is 3x1x1.2 cm (x in [.01, .04]) -> 1 mL and 3.6 mL."""
    a = o3d.geometry.TriangleMesh.create_box(0.01, 0.01, 0.01).subdivide_midpoint(3)
    b = o3d.geometry.TriangleMesh.create_box(0.03, 0.01, 0.012).subdivide_midpoint(4)
    b.translate((0.01, 0, 0))
    return a + b


def test_label_crop_excludes_neighbour_where_bbox_leaks():
    mesh = _two_boxes()
    v = np.asarray(mesh.vertices)
    n_a = len(o3d.geometry.TriangleMesh.create_box(1, 1, 1).subdivide_midpoint(3).vertices)
    labels_v = np.where(np.arange(len(v)) < n_a, 1, 2)             # A's vertices come first in a + b
    vl = label_mesh_vertices(mesh, v, labels_v, radius=0.003)       # labelled cloud = the vertices
    a_pts = v[(labels_v == 1) & (v[:, 2] > 0.001)]
    sub_label, info = crop_mesh_by_label(mesh, vl, 1, a_pts, min_faces=20)
    vol_label = mesh_prism_volume(sub_label, Z)
    vol_bbox = mesh_prism_volume(crop_mesh_to_points(mesh, a_pts), Z)
    assert info["crop"] == "label" and info["n_faces"] >= 20
    assert abs(vol_label - 1.0) < 0.15, vol_label
    assert vol_bbox > 1.3, vol_bbox          # bbox pad (4 mm) pulls in B's top faces


def test_label_crop_falls_back_to_bbox_with_few_faces():
    mesh = _two_boxes()
    v = np.asarray(mesh.vertices)
    vl = np.zeros(len(v), np.int64)          # nothing labelled -> label crop is empty
    sub, info = crop_mesh_by_label(mesh, vl, 1, v[v[:, 0] < 0.03], min_faces=20)
    assert info["crop"] == "bbox" and info["label_faces"] == 0 and len(sub.triangles) > 0


# ---- container ICP gate -------------------------------------------------------------------------

def _bowl(R=0.06, depth=0.03, sp=0.002):
    g = np.arange(-R, R + sp, sp)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    m = r <= R
    z = depth * (r[m] / R) ** 2 - 0.0     # parabolic bowl surface rising to the rim
    return np.c_[X[m], Y[m], z + 0.005]


def _template_and_scene(shift=(0.03, -0.02)):
    bowl = _bowl()
    rng = np.random.default_rng(1)
    tab = rng.uniform(-0.15, 0.15, (6000, 2))
    tab = tab[np.hypot(*tab.T) > 0.065]
    table = np.c_[tab, np.zeros(len(tab))]
    tmpl = ContainerTemplate.from_points(np.vstack([bowl, table]), Z, band=0.15)
    scene = np.vstack([bowl + np.array([*shift, 0]), table + np.array([*shift, 0]) * 0])
    return tmpl, scene


def test_icp_gate_passes_on_matching_template():
    tmpl, scene = _template_and_scene()
    tmpl.register(scene, Z)
    assert tmpl.last_ok, (tmpl.last_fitness, tmpl.last_rmse)


def test_icp_gate_flags_wrong_template():
    tmpl, scene = _template_and_scene()
    rng = np.random.default_rng(2)
    # deliberately wrong template: a tall narrow cone, nothing like the scanned bowl
    ang = rng.uniform(0, 2 * np.pi, 3000)
    rad = rng.uniform(0, 0.02, 3000)
    wrong = ContainerTemplate(points=np.c_[rad * np.cos(ang), rad * np.sin(ang), 0.08 * (1 - rad / 0.02)], name="wrong")
    wrong.register(scene, Z)
    assert not wrong.last_ok


# ---- plane fit ----------------------------------------------------------------------------------

def test_plane_fit_with_outliers_recovers_plane():
    rng = np.random.default_rng(3)
    tilt = np.radians(1.2)
    n = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    u = np.cross(n, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    ab = rng.uniform(-0.2, 0.2, (4000, 2))
    pts = ab[:, :1] * u + ab[:, 1:] * v + rng.normal(0, 0.0005, (4000, 1)) * n
    k = rng.choice(4000, 200, replace=False)          # 5 % outliers 20 mm above
    pts[k] += 0.02 * n
    plane, info = fit_support_plane_info(pts, up_hint=np.array([0.0, 0.0, 1.0]), max_tilt_deg=5.0)
    ang = np.degrees(np.arccos(np.clip(plane.normal @ n, -1, 1)))
    assert ang < 0.5, ang
    assert not info["gravity_fallback"]
    assert abs(plane.height(pts[np.setdiff1d(np.arange(4000), k)]).mean()) < 0.001


def test_plane_fit_gravity_fallback_flagged():
    rng = np.random.default_rng(4)
    ab = rng.uniform(-0.2, 0.2, (3000, 2))
    tilt = np.radians(4.0)
    pts = np.c_[ab, ab[:, 0] * np.tan(tilt)]            # table tilted 4 deg vs gravity z
    plane, info = fit_support_plane_info(pts, up_hint=np.array([0.0, 0.0, 1.0]))
    assert info["gravity_fallback"] and info["tilt_deg"] > 3.5
    assert np.allclose(plane.normal, [0, 0, 1])
