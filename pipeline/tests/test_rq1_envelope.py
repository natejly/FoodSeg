"""GT-envelope column fill: reproduces a known cylinder envelope; a pile of spheres has envelope >> solid."""
import numpy as np

from pipeline.eval import rq1_envelope as E


def _cyl_surface(r, h, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    a_side, a_cap = 2 * np.pi * r * h, np.pi * r * r
    tot = a_side + 2 * a_cap
    n_s, n_c = int(n * a_side / tot), int(n * a_cap / tot)
    th = rng.uniform(0, 2 * np.pi, n_s)
    side = np.c_[r * np.cos(th), r * np.sin(th), rng.uniform(0, h, n_s)]
    caps = []
    for z in (0.0, h):
        rr, t = r * np.sqrt(rng.uniform(0, 1, n_c)), rng.uniform(0, 2 * np.pi, n_c)
        caps.append(np.c_[rr * np.cos(t), rr * np.sin(t), np.full(n_c, z)])
    return np.vstack([side] + caps)


def test_cylinder_envelope_within_3pct():
    r, h = 0.03, 0.04
    v = E.column_fill_envelope(_cyl_surface(r, h), None)
    true = np.pi * r * r * h * 1e6
    assert abs(v - true) / true < 0.03, (v, true)


def test_sphere_pile_envelope_exceeds_solid():
    r = 0.01
    rng = np.random.default_rng(1)
    pts, n_sph = [], 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(3):          # cubic stacking of touching spheres, 3 layers
                c = np.array([ix, iy, iz]) * 2 * r + [r, r, r]
                u = rng.normal(size=(1500, 3))
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                pts.append(c + r * u)
                n_sph += 1
    solid = n_sph * 4 / 3 * np.pi * r ** 3 * 1e6
    env = E.column_fill_envelope(np.vstack(pts), None)
    assert env / solid > 1.3, (env, solid)
