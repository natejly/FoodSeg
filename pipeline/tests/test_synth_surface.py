"""Blender / taichi-free checks of synth.surface_detail (real geometry for soft foods) and its looks tables.

    cd pipeline && python -m pytest tests/test_synth_surface.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from synth import looks
from synth.geometry import mesh_volume
from synth.surface_detail import displace, fbm


def _heap(r=0.05, h=0.03):
    """A mound resting on a floor: an ellipsoid cut at z = 0 (open bottom closed flat), 1.2 mm edges."""
    m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    V = np.asarray(m.vertices) * [r, r, h]
    keep = V[:, 2] > -1e-9
    V[~keep, 2] = 3e-4                                  # squash the lower half onto the dish (pressed, as mpm does)
    return V, np.asarray(m.faces)


@pytest.mark.parametrize("kind", ["grain", "porridge", "puree", "curd", "liquid"])
def test_displace_preserves_volume_and_topology(kind):
    V, F = _heap()
    floor = np.zeros(len(V))
    g = looks.GEOM[kind]
    V2, info = displace(V, F, g, floor, seed=3)
    assert V2.shape == V.shape                          # same topology: closed mesh stays closed
    assert abs(mesh_volume(V2, F) / mesh_volume(V, F) - 1) < 0.01
    assert info["ok"] and abs(info["vol_change"]) < 0.01
    assert info["rms_mm"] > 0.15                        # actually irregular
    assert V2[:, 2].min() >= -1e-9                      # nothing below the dish


def test_displace_deterministic_and_seeded():
    V, F = _heap()
    a, _ = displace(V, F, looks.GEOM["puree"], np.zeros(len(V)), seed=5)
    b, _ = displace(V, F, looks.GEOM["puree"], np.zeros(len(V)), seed=5)
    c, _ = displace(V, F, looks.GEOM["puree"], np.zeros(len(V)), seed=6)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_fbm_unit_variance():
    P = np.random.default_rng(0).uniform(0, 0.1, (5000, 3))
    n = fbm(P, 0.01, 4, 1)
    assert abs(n.std() - 1) < 1e-6 and abs(n.mean()) < 0.2


def test_looks_have_geom_and_optics():
    for name in list(looks.KIND) + ["unknown food"]:
        L = looks.look(name, np.random.default_rng(1))
        assert L["geom"]["amp_mm"] <= 3.0 and L["optics"]["class"] in looks.OPTICS
        o = L["optics"]
        assert 0 <= o["sss"] <= 1 and o["scale"] <= 0.005
        assert o["rough"][1] < o["rough"][0]            # wet is smoother than dry
        assert 0.8 <= o["radius"][0] / 1.0 and o["radius"][2] <= o["radius"][0]   # no long red-only radii
