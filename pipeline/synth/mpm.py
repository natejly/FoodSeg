"""Settle a hard-preset scene with MLS-MPM (Taichi, GPU): soft food flows, slumps and heaps by
consistency, dropped scans sink into it and push it aside, and everything conforms to everything.

    python -m synth.mpm SCENE_DIR/spec.json [--time 1.0] [--dx 0.002]     (synth env)

Rigid-body drops (Bullet) left soft food as rigid domes that items sat on top of. Here every
food is a particle continuum on one grid, with a constitutive model per consistency
(synth.looks.PHYSICS): Drucker-Prager sand for grains, weakly compressible viscous fluid for
liquids / oatmeal / yogurt, snow-style plasticity for mash and curds, stiff elastic for scans.

In:  the spec's served shapes (procedural pieces / toppings, closed meshes in world
     coordinates) and its dropped scans (CoACD convex parts, posed at init_location /
     init_euler), the dish (procedural or scanned) and table as a heightfield collider.
Out: each soft item's settled surface (marching cubes on its particles' density) replaces its
     mesh; each scan gets the rigid pose best fitting its particles (Kabsch) as `matrix_world`,
     applied by render.py to the full-resolution mesh. Labels are exact (every particle knows
     its item); volumes are those of the output meshes.
"""

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import taichi as ti

from .collision import to_blender
from .geometry import mesh_volume, topping_mesh
from .looks import ITEM_PHYSICS
from .surface_detail import displace

MODELS = {"elastic": 0, "sand": 1, "plastic": 2, "fluid": 3, "rigid": 4}
MAX_BODIES = 64
MAX_MATERIALS = 64
GRAVITY = 9.81
FRICTION = 0.6                   # food on the dish / table (Coulomb, at the grid)


# ---- particles from closed meshes -------------------------------------------

def _column_hits(V: np.ndarray, F: np.ndarray, x0: float, y0: float, h: float, nx: int, ny: int):
    """Where vertical lines x0 + i h, y0 + j h cross the triangles: (i, j, z) arrays."""
    T = V[F]
    lo = np.maximum(np.floor((T[:, :, :2].min(1) - [x0, y0]) / h).astype(np.int64) + 1, 0)
    hi = np.minimum(np.floor((T[:, :, :2].max(1) - [x0, y0]) / h).astype(np.int64), [nx - 1, ny - 1])
    ni, nj = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    ok = (ni > 0) & (nj > 0)
    T, lo, ni, nj = T[ok], lo[ok], ni[ok], nj[ok]
    cnt = ni * nj
    tri = np.repeat(np.arange(len(T)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    i, j = lo[tri, 0] + k // nj[tri], lo[tri, 1] + k % nj[tri]
    a, b, c = T[tri, 0], T[tri, 1], T[tri, 2]
    px, py = x0 + i * h - a[:, 0], y0 + j * h - a[:, 1]
    e1, e2 = b - a, c - a
    d = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (px * e2[:, 1] - e2[:, 0] * py) / d
        v = (e1[:, 0] * py - px * e1[:, 1]) / d
        inside = (np.abs(d) > 1e-16) & (u >= 0) & (v >= 0) & (u + v <= 1)   # edge-on: d = 0, nan
        z = a[:, 2] + u * e1[:, 2] + v * e2[:, 2]
    return i[inside], j[inside], z[inside]


def fill_points(V: np.ndarray, F: np.ndarray, h: float, origin: np.ndarray | None = None) -> np.ndarray:
    """Points on a cubic lattice of spacing h inside a closed mesh (vertical-ray parity); the
    lattice (through `origin` if given) is offset by an irrational fraction so rays miss edges
    and vertices."""
    lo, hi = V.min(0), V.max(0)
    base = lo if origin is None else origin + np.floor((lo - origin) / h) * h
    x0, y0, z0 = base + h * (0.5 + np.array([1e-4 * np.pi, 1e-4 * np.e, 1e-4 * np.sqrt(2)]))
    nx, ny = int((hi[0] - x0) / h) + 1, int((hi[1] - y0) / h) + 1
    i, j, z = _column_hits(V, F, x0, y0, h, nx, ny)
    if not len(z):
        return np.zeros((0, 3))
    col = i * ny + j
    o = np.lexsort((z, col))
    col, z = col[o], z[o]
    starts = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
    size = np.diff(np.r_[starts, len(col)])
    even = np.repeat(size % 2 == 0, size)              # odd counts: a grazing ray, skip column
    rank = np.arange(len(col)) - np.repeat(starts, size)
    enter = np.flatnonzero(even & (rank % 2 == 0))
    za, zb, cc = z[enter], z[enter + 1], col[enter]
    k0 = np.ceil((za - z0) / h).astype(np.int64)
    n = np.maximum(np.floor((zb - z0) / h).astype(np.int64) - k0 + 1, 0)
    seg = np.repeat(np.arange(len(n)), n)
    kk = k0[seg] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    c = cc[seg]
    return np.stack([x0 + (c // ny) * h, y0 + (c % ny) * h, z0 + kk * h], 1)


def convex_points(P: np.ndarray, h: float, origin: np.ndarray | None = None) -> np.ndarray:
    from scipy.spatial import ConvexHull
    return fill_points(P, ConvexHull(P).simplices, h, origin)


def shell_points(P: np.ndarray, h: float, layers: int = 2) -> np.ndarray:
    """Keep the points of a lattice-aligned point set within `layers` lattice steps of its surface
    (a rigid body needs no filled interior: the grid only sees its skin)."""
    from scipy.ndimage import binary_erosion
    lo = P.min(0)
    idx = np.round((P - lo) / h).astype(np.int64) + layers + 1
    occ = np.zeros(idx.max(0) + layers + 2, bool)
    occ[tuple(idx.T)] = True
    inner = binary_erosion(occ, iterations=layers, border_value=0)
    return P[~inner[tuple(idx.T)]]


def euler_xyz(e) -> np.ndarray:
    """Blender XYZ Euler -> rotation matrix (R = Rz Ry Rx)."""
    cx, cy, cz = np.cos(e)
    sx, sy, sz = np.sin(e)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_local(path: str, fmt: str, up_axis: str = "Z") -> tuple[np.ndarray, np.ndarray]:
    """A mesh in the coordinates Blender gives it after import (render.load_mesh), unscaled."""
    if fmt == "npz":
        z = np.load(path)
        return z["V"].astype(np.float64), z["F"].astype(np.int64)
    import trimesh
    m = trimesh.load(path, force="mesh")
    V = np.asarray(m.vertices, np.float64)
    if fmt in ("gltf", "glb"):
        V = to_blender(V, "gltf")
    elif up_axis != "Z":                              # Y-up OBJ: Blender's (x, -z, y)
        V = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
    return V, np.asarray(m.faces, np.int64)


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R A_i - B_i| for centred point sets."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


# ---- the dish and table as a heightfield ------------------------------------

def dish_heightfield(spec: dict, scene_dir: Path, x0: float, y0: float, h: float, nx: int, ny: int) -> np.ndarray:
    """Top surface z of dish / table over the grid columns (table z = 0)."""
    c = spec["container"]
    if c.get("asset"):                                # as render.py places a scanned dish
        a = c["asset"]
        V, F = load_local(a["mesh"], a["fmt"], a.get("up_axis", "Z"))
        lo, hi = V.min(0), V.max(0)
        R = euler_xyz([0.0, 0.0, a.get("yaw", 0.0)])
        V = V @ R.T
        ctr = R @ ((lo + hi) / 2)                     # Blender: centre of the rotated local AABB
        V = V - [ctr[0], ctr[1], V[:, 2].min() - 0.0003]
    else:
        z = np.load(scene_dir / c["mesh"])
        V, F = z["V"].astype(np.float64), z["F"].astype(np.int64)
    H = np.zeros(nx * ny)
    i, j, zz = _column_hits(V, F, x0, y0, h, nx, ny)
    np.maximum.at(H, i * ny + j, zz)
    return H.reshape(nx, ny)


# ---- the solver --------------------------------------------------------------

@ti.data_oriented
class Solver:
    """MLS-MPM (quadratic B-splines, APIC), sized once for the largest scene."""

    def __init__(self, max_particles: int = 1_500_000, max_grid=(208, 208, 208)):
        self.x = ti.Vector.field(3, ti.f32, max_particles)
        self.v = ti.Vector.field(3, ti.f32, max_particles)
        self.C = ti.Matrix.field(3, 3, ti.f32, max_particles)
        self.F = ti.Matrix.field(3, 3, ti.f32, max_particles)
        self.J = ti.field(ti.f32, max_particles)           # fluid volume ratio
        self.mat = ti.field(ti.i32, max_particles)
        self.body = ti.field(ti.i32, max_particles)        # rigid body index, -1: continuum
        self.bn = ti.field(ti.f32, MAX_BODIES)             # per body: particle count, sums, momenta
        self.bx = ti.Vector.field(3, ti.f32, MAX_BODIES)
        self.bv = ti.Vector.field(3, ti.f32, MAX_BODIES)
        self.bL = ti.Vector.field(3, ti.f32, MAX_BODIES)
        self.bI = ti.Matrix.field(3, 3, ti.f32, MAX_BODIES)
        self.bw = ti.Vector.field(3, ti.f32, MAX_BODIES)
        self.gv = ti.Vector.field(3, ti.f32, max_grid)
        self.gm = ti.field(ti.f32, max_grid)
        self.H = ti.field(ti.f32, max_grid[:2])
        self.prm = ti.field(ti.f32, (64, 8))               # per material: model mu lam rho visc K a b
        self.max_particles, self.max_grid = max_particles, max_grid

    @ti.kernel
    def reset(self, n: ti.i32):
        for p in range(n):
            self.v[p] = ti.Vector.zero(ti.f32, 3)
            self.C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.F[p] = ti.Matrix.identity(ti.f32, 3)
            self.J[p] = 1.0

    @ti.kernel
    def clear(self, nx: ti.i32, ny: ti.i32, nz: ti.i32):
        for I in ti.grouped(ti.ndrange(nx, ny, nz)):
            self.gv[I] = ti.Vector.zero(ti.f32, 3)
            self.gm[I] = 0.0

    @ti.kernel
    def p2g(self, n: ti.i32, dt: ti.f32, dx: ti.f32, o: ti.types.vector(3, ti.f32), pvol: ti.f32):
        inv_dx = 1.0 / dx
        Id = ti.Matrix.identity(ti.f32, 3)
        for p in range(n):
            m = self.mat[p]
            model = ti.cast(self.prm[m, 0], ti.i32)
            mu, lam, rho = self.prm[m, 1], self.prm[m, 2], self.prm[m, 3]
            Xp = (self.x[p] - o) * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            Cp = self.C[p]
            tau = ti.Matrix.zero(ti.f32, 3, 3)
            if model == 4:                                  # rigid: motion projected in rigidify()
                pass
            elif model == 3:                                # weakly compressible viscous fluid
                self.J[p] *= 1.0 + dt * Cp.trace()
                Jp = self.J[p]
                tau = self.prm[m, 5] * Jp * (Jp - 1.0) * Id + self.prm[m, 4] * (Cp + Cp.transpose())
            else:
                Fp = (Id + dt * Cp) @ self.F[p]
                U, sig, V = ti.svd(Fp)
                if model == 1:                              # Drucker-Prager sand (Klar et al. 2016)
                    e = ti.Vector([ti.log(ti.max(ti.abs(sig[d, d]), 1e-4)) for d in ti.static(range(3))])
                    tr = e.sum()
                    eh = e - tr / 3.0
                    en = eh.norm() + 1e-12
                    if tr >= 0.0:                           # tension: grains separate
                        e = ti.Vector.zero(ti.f32, 3)
                    else:
                        dg = en + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * tr * self.prm[m, 6]
                        if dg > 0.0:
                            e = e - dg / en * eh
                    for d in ti.static(range(3)):
                        sig[d, d] = ti.exp(e[d])
                    tr2 = e.sum()
                    t = ti.Matrix.zero(ti.f32, 3, 3)
                    for d in ti.static(range(3)):
                        t[d, d] = 2.0 * mu * e[d] + lam * tr2
                    tau = U @ t @ U.transpose()
                    Fp = U @ sig @ V.transpose()
                else:
                    if model == 2:                          # plasticity: clamp principal stretches
                        for d in ti.static(range(3)):
                            sig[d, d] = ti.min(ti.max(sig[d, d], 1.0 - self.prm[m, 6]), 1.0 + self.prm[m, 7])
                        Fp = U @ sig @ V.transpose()
                    Jp = sig[0, 0] * sig[1, 1] * sig[2, 2]
                    tau = 2.0 * mu * (Fp - U @ V.transpose()) @ Fp.transpose() + lam * Jp * (Jp - 1.0) * Id
                self.F[p] = Fp
            mass = rho * pvol
            affine = -dt * pvol * 4.0 * inv_dx * inv_dx * tau + mass * Cp
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                off = ti.Vector([i, j, k])
                dpos = (off - fx) * dx
                wt = w[i][0] * w[j][1] * w[k][2]
                self.gv[base + off] += wt * (mass * self.v[p] + affine @ dpos)
                self.gm[base + off] += wt * mass

    @ti.kernel
    def grid_update(self, nx: ti.i32, ny: ti.i32, nz: ti.i32, dt: ti.f32, dx: ti.f32,
                    o: ti.types.vector(3, ti.f32), damp: ti.f32):
        for I in ti.grouped(ti.ndrange(nx, ny, nz)):
            if self.gm[I] > 0:
                v = self.gv[I] / self.gm[I]
                v[2] -= dt * GRAVITY
                v *= damp
                z = o[2] + I[2] * dx
                i0, j0 = ti.max(I[0], 1), ti.max(I[1], 1)
                i1, j1 = ti.min(I[0], nx - 2), ti.min(I[1], ny - 2)
                h = self.H[I[0], I[1]]
                if z <= h:                                  # dish / table: no entry, Coulomb friction
                    nrm = ti.Vector([-(self.H[i1 + 1, I[1]] - self.H[i0 - 1, I[1]]) / (2 * dx),
                                     -(self.H[I[0], j1 + 1] - self.H[I[0], j0 - 1]) / (2 * dx), 1.0]).normalized()
                    vn = v.dot(nrm)
                    if vn < 0:
                        vt = v - vn * nrm
                        vtn = vt.norm()
                        if vtn <= -FRICTION * vn:
                            v = ti.Vector.zero(ti.f32, 3)
                        else:
                            v = vt * (1.0 + FRICTION * vn / vtn)
                for d in ti.static(range(3)):              # domain walls
                    if (I[d] < 3 and v[d] < 0) or (I[d] > (nx, ny, nz)[d] - 4 and v[d] > 0):
                        v[d] = 0.0
                self.gv[I] = v

    @ti.kernel
    def g2p(self, n: ti.i32, dt: ti.f32, dx: ti.f32, o: ti.types.vector(3, ti.f32)) -> ti.f32:
        inv_dx = 1.0 / dx
        vmax = 0.0
        for p in range(n):
            Xp = (self.x[p] - o) * inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            nv = ti.Vector.zero(ti.f32, 3)
            nC = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                off = ti.Vector([i, j, k])
                dpos = (off - fx) * dx
                gv = self.gv[base + off]
                wt = w[i][0] * w[j][1] * w[k][2]
                nv += wt * gv
                nC += 4.0 * inv_dx * inv_dx * wt * gv.outer_product(dpos)
            self.v[p], self.C[p] = nv, nC
            if self.body[p] < 0:
                self.x[p] += dt * nv
                ti.atomic_max(vmax, nv.norm())
        return vmax

    @ti.kernel
    def body_sums(self, n: ti.i32, nb: ti.i32):
        for b in range(nb):
            self.bn[b], self.bx[b], self.bv[b] = 0.0, ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
            self.bL[b], self.bI[b] = ti.Vector.zero(ti.f32, 3), ti.Matrix.zero(ti.f32, 3, 3)
        for p in range(n):
            b = self.body[p]
            if b >= 0:
                self.bn[b] += 1.0
                self.bx[b] += self.x[p]
                self.bv[b] += self.v[p]

    @ti.kernel
    def body_momenta(self, n: ti.i32, nb: ti.i32):
        for b in range(nb):
            self.bx[b] /= ti.max(self.bn[b], 1.0)
            self.bv[b] /= ti.max(self.bn[b], 1.0)
        Id = ti.Matrix.identity(ti.f32, 3)
        for p in range(n):
            b = self.body[p]
            if b >= 0:
                r = self.x[p] - self.bx[b]
                self.bL[b] += r.cross(self.v[p])
                self.bI[b] += r.dot(r) * Id - r.outer_product(r)

    @ti.kernel
    def rigidify(self, n: ti.i32, nb: ti.i32, dt: ti.f32) -> ti.f32:
        """Scans are rigid bodies two-way coupled to the continuum: their particles' grid
        velocities are replaced by the rigid motion with the same linear / angular momentum."""
        for b in range(nb):
            self.bw[b] = (self.bI[b] + 1e-10 * ti.Matrix.identity(ti.f32, 3)).inverse() @ self.bL[b]
        vmax = 0.0
        for p in range(n):
            b = self.body[p]
            if b >= 0:
                w = self.bw[b]
                r = self.x[p] - self.bx[b]
                vel = self.bv[b] + w.cross(r)
                self.v[p] = vel
                self.C[p] = ti.Matrix([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
                self.x[p] += dt * vel
                ti.atomic_max(vmax, vel.norm())
        return vmax

    def run(self, X: np.ndarray, mat: np.ndarray, body: np.ndarray, params: list[dict], H: np.ndarray,
            origin: np.ndarray, dims: tuple[int, int, int], dx: float, pvol: float, sim_time: float,
            log=print) -> np.ndarray:
        n, nb = len(X), int(body.max()) + 1
        assert n <= self.max_particles and nb <= MAX_BODIES and len(params) <= MAX_MATERIALS and all(d <= m for d, m in zip(dims, self.max_grid))
        prm = np.zeros((64, 8), np.float32)
        c_max, visc_max = 1.0, 0.0
        for m, P in enumerate(params):
            model = MODELS[P["model"]]
            if model == 4:
                prm[m] = [model, 0, 0, P["rho"], 0, 0, 0, 0]
                continue
            if model == 3:
                mu = lam = 0.0
                c = np.sqrt(P["K"] / P["rho"])
                prm[m] = [model, 0, 0, P["rho"], P["visc"], P["K"], 0, 0]
                visc_max = max(visc_max, P["visc"] / P["rho"])
            else:
                E, nu = P["E"], P["nu"]
                mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
                c = np.sqrt((lam + 2 * mu) / P["rho"])
                a = b = 0.0
                if model == 1:
                    s = np.sin(np.radians(P["phi_deg"]))
                    a = np.sqrt(2 / 3) * 2 * s / (3 - s)
                elif model == 2:
                    a, b = P["theta_c"], P["theta_s"]
                prm[m] = [model, mu, lam, P["rho"], 0, 0, a, b]
            c_max = max(c_max, c)
        c_max = max(c_max, 5.0)                    # rigid-only scenes: bound the step by the fall speed
        dt = 0.3 * dx / c_max
        if visc_max:
            dt = min(dt, 0.15 * dx * dx / visc_max)
        steps = int(np.ceil(sim_time / dt))
        self.prm.from_numpy(prm)
        Hf = np.zeros(self.max_grid[:2], np.float32)
        Hf[:dims[0], :dims[1]] = H
        self.H.from_numpy(Hf)
        pad = lambda a, shape, dtype: np.concatenate([a.astype(dtype), np.zeros((self.max_particles - n, *shape), dtype)])
        self.x.from_numpy(pad(X, (3,), np.float32))
        self.mat.from_numpy(pad(mat, (), np.int32))
        self.body.from_numpy(np.concatenate([body.astype(np.int32), np.full(self.max_particles - n, -1, np.int32)]))
        self.reset(n)
        o = ti.Vector(origin.astype(np.float32).tolist())
        t0, vmax = time.time(), 0.0
        for s in range(steps):
            frac = s / steps
            # light air damping while things fall, then settle (kills jiggle without freezing flow early)
            gamma = 2.0 if frac < 0.6 else 2.0 + 60.0 * (frac - 0.6) / 0.4
            self.clear(*dims)
            self.p2g(n, dt, dx, o, pvol)
            self.grid_update(*dims, dt, dx, o, float(np.exp(-gamma * dt)))
            vmax = self.g2p(n, dt, dx, o)
            if nb > 0:
                self.body_sums(n, nb)
                self.body_momenta(n, nb)
                vmax = max(vmax, self.rigidify(n, nb, dt))
            if s % 2000 == 0 or s == steps - 1:
                log(f"[mpm] step {s}/{steps} t={s * dt:.3f}s vmax={vmax:.3f} m/s ({time.time() - t0:.0f}s)")
        return self.x.to_numpy()[:n].astype(np.float64)


_SOLVER: Solver | None = None


def solver() -> Solver:
    global _SOLVER
    if _SOLVER is None:
        ti.init(arch=ti.gpu, device_memory_GB=0.9, offline_cache=True)
        _SOLVER = Solver()
    return _SOLVER


# ---- particles back to meshes ------------------------------------------------

def surface_mesh(P: np.ndarray, h: float, H_at, max_faces: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    """Closed surface around particles (rest spacing h): density on an h-grid, Gaussian
    smoothed, marching cubes at half the rest density; the largest components only; down-facing
    vertices close to the dish are pressed onto it (particles stop ~h/2 above a collider)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    import trimesh
    lo = P.min(0) - 4 * h
    idx = np.floor((P - lo) / h).astype(np.int64)
    shape = idx.max(0) + 5
    G = np.zeros(shape, np.float32)
    np.add.at(G, tuple(idx.T), 1.0)
    G = gaussian_filter(G, 1.5)
    Vm, Fm, Nm, _ = marching_cubes(G, level=0.5, spacing=(h, h, h))
    Vm = Vm + lo + 0.5 * h
    m = trimesh.Trimesh(Vm, Fm[:, ::-1], process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        vols = np.array([abs(p.volume) for p in parts])
        m = trimesh.util.concatenate([p for p, v in zip(parts, vols) if v > 0.02 * vols.max()])
    if len(m.faces) > max_faces:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    V, F = np.asarray(m.vertices, np.float64).copy(), np.asarray(m.faces, np.int64)
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    nz = trimesh.Trimesh(V, F, process=False).vertex_normals[:, 2]
    floor = H_at(V[:, 0], V[:, 1])
    press = (nz < -0.2) & (V[:, 2] - floor < 2.5 * h)
    V[press, 2] = floor[press] + 3e-4
    return V, F


# ---- how foods share a heap --------------------------------------------------

def relayout(X: np.ndarray, owner: np.ndarray, heap: dict, h: float) -> np.ndarray:
    """Re-assign the heap's particles (built as radial wedges, one food each) to a natural
    arrangement before simulating; returns the new owner ids. Layouts (spec_hard):
      sides    foods side by side across the heap along wavy lines, in the given volume shares;
      patches  irregular patches: nearest of a few warped seeds, two per food;
      layers   the last food (the runniest) poured over part of the others: the top layer of
               the heap within a wobbly disc; the rest split as `sides`."""
    rng = np.random.default_rng(heap["seed"])
    ids = list(heap["items"])
    sel = np.flatnonzero(np.isin(owner, ids))
    P = X[sel]
    c = np.asarray(heap["centre"], float)
    xy = P[:, :2] - c
    rf = heap["rf"]
    new = owner.copy()

    def wavy(Q, n_foods, fracs):
        a = rng.uniform(0, 2 * np.pi)
        d, perp = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        t = Q @ perp / rf
        s = Q @ d + 0.18 * rf * np.sin(rng.uniform(2, 4) * t + rng.uniform(0, 2 * np.pi)) \
            + 0.08 * rf * np.sin(rng.uniform(5, 8) * t + rng.uniform(0, 2 * np.pi))
        cuts = np.quantile(s, np.cumsum(fracs)[:-1] / np.sum(fracs))
        return np.searchsorted(cuts, s)

    if heap["layout"] == "sides":
        k = wavy(xy, len(ids), heap["fractions"])
    elif heap["layout"] == "patches":
        seeds = rng.uniform(-0.8, 0.8, (2 * len(ids), 2)) * rf
        warp = np.stack([np.sin(xy[:, 1] / rf * 5 + 1.3), np.sin(xy[:, 0] / rf * 5 + 0.7)], 1) * 0.12 * rf
        k = np.argmin((((xy + warp)[:, None, :] - seeds[None]) ** 2).sum(-1), axis=1) % len(ids)
    else:                                                  # layers
        k = np.zeros(len(P), int)
        if len(ids) > 2:
            k = wavy(xy, len(ids) - 1, heap["fractions"][:-1])
        col = np.floor(P[:, :2] / (2 * h)).astype(np.int64)
        key = col[:, 0] * 100003 + col[:, 1]
        order = np.argsort(key)
        _, start = np.unique(key[order], return_index=True)
        top = np.maximum.reduceat(P[order, 2], start)               # heap surface per 2h column
        ztop = np.empty(len(P))
        ztop[order] = np.repeat(top, np.diff(np.r_[start, len(P)]))
        off = rng.uniform(-0.3, 0.3, 2) * rf
        th = np.arctan2(xy[:, 1] - off[1], xy[:, 0] - off[0])
        r_lim = heap["top_r"] * (1 + 0.18 * np.sin(3 * th + rng.uniform(0, 6)) + 0.08 * np.sin(7 * th))
        on_top = (np.hypot(*(xy - off).T) < r_lim) & (P[:, 2] > ztop - heap["top_thick"])
        k[on_top] = len(ids) - 1
    if np.bincount(k, minlength=len(ids)).min() < 0.04 * len(P):     # a food nearly vanished: side by side
        k = wavy(xy, len(ids), heap["fractions"])
    new[sel] = np.asarray(ids)[k]
    return new


def compact_drops(X: np.ndarray, body: np.ndarray, floor_z: float, static: np.ndarray, cell: float,
                  gap: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Dropped bodies start high and staggered; most of that is empty air (and a huge grid). Lower each
    body, lowest first, until it is `gap` above the highest thing in each of its columns (the dish rim
    height, the soft food, the bodies already lowered). Returns new positions and per-body shifts."""
    nb = int(body.max()) + 1
    lo = X[:, :2].min(0)
    ij = np.floor((X[:, :2] - lo) / cell).astype(np.int64)
    ny = ij[:, 1].max() + 1
    key = ij[:, 0] * ny + ij[:, 1]
    hm = np.full(int(key.max()) + 1, floor_z)
    if static.any():
        np.maximum.at(hm, key[static], X[static, 2])
    X = X.copy()
    dz = np.zeros(nb)
    order = np.argsort([X[body == b, 2].min() for b in range(nb)])
    for b in order:
        sel = np.flatnonzero(body == b)
        d = float(np.min(X[sel, 2] - hm[key[sel]])) - gap
        if d > 0:
            X[sel, 2] -= d
            dz[b] = d
        np.maximum.at(hm, key[sel], X[sel, 2])
    return X, dz


class _Surface:
    """Height field of the settled food (max z of its mesh vertices per 1 mm bin, holes filled by the nearest bin, smoothed):
    what a topping lying on the food rests on. `top(c, x, y)` mirrors geometry.Mound.top."""

    def __init__(self, V: np.ndarray, centre, half: float, step: float = 0.001):
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        self.x0, self.y0, self.step = centre[0] - half, centre[1] - half, step
        n = int(2 * half / step) + 1
        G = np.full((n, n), -np.inf)
        ij = np.floor((V[:, :2] - [self.x0, self.y0]) / step).astype(int)
        ok = ((ij >= 0) & (ij < n)).all(1)
        np.maximum.at(G, (ij[ok, 0], ij[ok, 1]), V[ok, 2])
        have = np.isfinite(G)
        self.coverage = float(have.mean())
        if have.sum() < 20:
            self.G = None
            return
        idx = distance_transform_edt(~have, return_distances=False, return_indices=True)
        G = G[idx[0], idx[1]]
        self.G = gaussian_filter(G, 1.5)

    def top(self, c, x, y):
        from scipy.ndimage import map_coordinates
        x, y = np.asarray(x, float), np.asarray(y, float)
        pts = np.stack([(x - self.x0) / self.step, (y - self.y0) / self.step])
        return map_coordinates(self.G, pts.reshape(2, -1), order=1, mode="nearest").reshape(x.shape)


def _is_topping(it: dict) -> bool:
    sh = it.get("shape") or {}
    return sh.get("kind") == "topping" and sh.get("on_item") is not None and sh.get("thick") is not None


# ---- a scene -----------------------------------------------------------------

def settle(spec: dict, scene_dir: Path, sim_time: float = 1.0, dx: float = 0.002, log=print,
           dump: bool = False) -> dict:
    """Simulate the scene's soft food and dropped items together; rewrite spec items in place
    (soft: settled mesh + volume; dropped: matrix_world). No soft food: spec unchanged.
    dump: also save the particles (start, end, owner id) to mpm_particles.npz."""
    items = spec["items"]
    soft = [it for it in items if it.get("physics") and not it["dynamic"] and not it.get("cloth")]
    # a sauce / cream blob lying on a food is not simulated (as particles it splashed off the heap and floated, its
    # shadow detached): it is re-laid on the settled surface below once that is known
    tops = [it for it in soft if _is_topping(it)]
    soft = [it for it in soft if not _is_topping(it)]
    drop = [it for it in items if it["dynamic"]]
    if not soft and not drop:
        return spec
    assert len(drop) <= MAX_BODIES and len(soft) + len(drop) <= MAX_MATERIALS - 1, "too many items for the MPM solver"
    t0 = time.time()
    S = solver()
    cj = spec["container"]
    soft_mesh = {it["id"]: np.load(scene_dir / it["mesh"]) for it in soft}
    soft_mesh = {k: (z["V"].astype(np.float64), z["F"].astype(np.int64)) for k, z in soft_mesh.items()}

    while True:                                          # coarsen the lattice until particles and grid fit
        hp = dx / 2                                      # particle spacing: 8 per cell
        X, mat, owner, body, params = [], [], [], [], []
        jit = np.random.default_rng(0)
        frames = {}
        for it in soft:
            P = fill_points(*soft_mesh[it["id"]], hp)
            P = P + jit.uniform(-0.35, 0.35, P.shape) * hp      # off the lattice: no grid print on the surface
            params.append(it["physics"])
            X.append(P), mat.append(np.full(len(P), len(params) - 1)), owner.append(np.full(len(P), it["id"]))
            body.append(np.full(len(P), -1))
        for it in drop:
            s = it["to_m"] * it["scale"]
            if it.get("collision"):
                zc = np.load(it["collision"])
                parts = np.split(zc["V"].astype(np.float64), np.cumsum(zc["n"])[:-1])
                org = np.concatenate(parts).min(0) * s          # one lattice: overlapping parts merge
                P = np.concatenate([convex_points(q * s, hp, org) for q in parts if len(q) >= 4])
                P = np.unique(np.floor((P - org) / hp).astype(np.int64), axis=0) * hp + org + hp * 0.5
            else:
                V, F = load_local(it["mesh"], it["fmt"], it.get("up_axis", "Y"))
                P = fill_points(V * s, F, hp) if it["fmt"] == "npz" else convex_points(V * s, hp)
            if len(P) < 8:                                   # thinner than the lattice: its hull
                V, _ = load_local(it["mesh"], it["fmt"], it.get("up_axis", "Y"))
                P = convex_points(V * s, hp / 2)
            c = P.mean(0)                                    # centre of the filled body
            n_full = len(P)
            Q = shell_points(P, hp) if n_full > 64 else P    # skin only; mass kept by scaling density
            R0 = euler_xyz(it["init_euler"])
            L0 = np.asarray(it["init_location"], float)
            frames[it["id"]] = (s, c, R0)
            params.append({**ITEM_PHYSICS, "rho": ITEM_PHYSICS["rho"] * n_full / len(Q)})
            X.append((Q - c) @ R0.T + L0), mat.append(np.full(len(Q), len(params) - 1))
            owner.append(np.full(len(Q), it["id"])), body.append(np.full(len(Q), len(frames) - 1))
        X, mat, owner, body = np.concatenate(X), np.concatenate(mat), np.concatenate(owner), np.concatenate(body)
        dz_body = np.zeros(len(drop))
        if len(drop):          # skip the empty fall: lower each drop (lowest first) to 1 cm above what is under it
            X, dz_body = compact_drops(X, body, float(cj.get("H") or 0.0), body < 0, 3 * dx)
        if spec.get("heap") and spec["heap"]["layout"] != "wedges":
            owner = relayout(X, owner, spec["heap"], hp)
            mat_of = {it["id"]: k for k, it in enumerate(soft)}
            sel = np.isin(owner, spec["heap"]["items"])
            mat[sel] = [mat_of[o] for o in owner[sel]]
        reach = cj["R"] + cj.get("rim_w", 0.0) + 0.02
        lo = np.minimum(X.min(0), [-reach, -reach, 0.0]) - [4 * dx, 4 * dx, 0.0]
        hi = np.maximum(X.max(0), [reach, reach, 0.0]) + [4 * dx, 4 * dx, 0.03]
        lo[2] = -3 * dx
        dims = tuple(int(np.ceil(v)) + 1 for v in (hi - lo) / dx)
        if len(X) <= S.max_particles and all(d <= m for d, m in zip(dims, S.max_grid)):
            break
        log(f"[mpm] {len(X)} particles, grid {dims} at dx {dx * 1e3:.1f} mm: too big, coarsening")
        dx *= 1.15

    H = dish_heightfield(spec, scene_dir, lo[0], lo[1], dx, dims[0], dims[1])
    log(f"[mpm] {len(X)} particles ({len(soft)} soft, {len(drop)} dropped), grid {dims}, dx {dx * 1e3:.1f} mm")
    Xf = S.run(X, mat, body, params, H, lo, dims, dx, hp ** 3, sim_time, log)
    if dump:
        np.savez_compressed(scene_dir / "mpm_particles.npz", x0=X.astype(np.float32), x1=Xf.astype(np.float32),
                            owner=owner.astype(np.int16))

    # soft items: settled surfaces; the floor for pressing bottoms onto the dish, at h resolution
    nxh, nyh = int((hi[0] - lo[0]) / hp) + 1, int((hi[1] - lo[1]) / hp) + 1
    Hh = dish_heightfield(spec, scene_dir, lo[0], lo[1], hp, nxh, nyh)
    H_at = lambda x, y: Hh[np.clip(np.round((x - lo[0]) / hp).astype(int), 0, nxh - 1),
                           np.clip(np.round((y - lo[1]) / hp).astype(int), 0, nyh - 1)]
    raw = {}
    for it in soft:
        raw[it["id"]] = surface_mesh(Xf[owner == it["id"]], hp, H_at)
    for it in soft:
        V, F = raw[it["id"]]
        extra = {}
        g = (it.get("look") or {}).get("geom")
        if g and not os.environ.get("SYNTH_NO_DISP"):   # irregular real geometry, baked into the saved mesh
            others = np.concatenate([raw[k][0] for k in raw if k != it["id"]] or [np.zeros((0, 3))])
            seed = zlib.crc32(str(spec.get("scene_id", "")).encode()) % 100000 * 100 + int(it["id"])
            V2, info = displace(V, F, g, H_at(V[:, 0], V[:, 1]), seed, others)
            log(f"[mpm] item {it['id']} {it.get('name')}: displaced rms {info['rms_mm']:.2f} mm max {info['max_mm']:.1f} mm, "
                f"volume {info['vol_change'] * 100:+.3f} %, flipped {info['flipped_frac'] * 100:.2f} %"
                + ("" if info["ok"] else "  REJECTED (kept smooth)"))
            extra = {"disp": info, "geom_baked": bool(info["ok"]), "volume_ml_raw": float(mesh_volume(V, F) * 1e6)}
            V = V2
        mesh = f"mpm_{it['id']}.npz"
        np.savez_compressed(scene_dir / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
        it.update(mesh=mesh, served_mesh=it["mesh"], volume_ml=float(mesh_volume(V, F) * 1e6), mpm=True, **extra)
    final = {it["id"]: np.load(scene_dir / it["mesh"])["V"].astype(np.float64) for it in soft}
    for it in tops:
        sh = it["shape"]
        base = np.concatenate([V for V in final.values()] or [np.zeros((0, 3))])
        half = 1.6 * sh["radius"] + 0.02
        surf = _Surface(base, sh["centre"], half)
        if surf.G is None or surf.coverage < 0.15:
            log(f"[mpm] topping {it['id']} {it.get('name')}: no settled surface under it, kept as served")
            continue
        V, F = topping_mesh(None, surf, tuple(sh["centre"]), sh["radius"], sh["thick"], seed=sh["seed"], lift=-4e-4)
        mesh = f"mpm_{it['id']}.npz"
        np.savez_compressed(scene_dir / mesh, V=V.astype(np.float32), F=F.astype(np.int32))
        it.update(mesh=mesh, served_mesh=it["mesh"], volume_ml=float(mesh_volume(V, F) * 1e6), mpm=True)
        log(f"[mpm] topping {it['id']} {it.get('name')}: laid on the settled surface (z {V[:, 2].min() * 1e3:.1f}.."
            f"{V[:, 2].max() * 1e3:.1f} mm), volume {it['volume_ml']:.1f} mL")
    # dropped items: rigid pose of their particles
    for b, it in enumerate(drop):
        sel = owner == it["id"]
        A, B = X[sel], Xf[sel]
        g0, g1 = A.mean(0), B.mean(0)
        Rf = kabsch(A - g0, B - g1)
        s, c, R0 = frames[it["id"]]
        M = np.eye(4)
        M[:3, :3] = Rf @ R0 * s                          # world = M @ mesh-local (unscaled)
        L0 = np.asarray(it["init_location"], float) - [0.0, 0.0, dz_body[b]]   # where its filled centroid started
        M[:3, 3] = g1 + Rf @ (L0 - g0 - R0 @ c)
        rms = float(np.sqrt((((A - g0) @ Rf.T - (B - g1)) ** 2).sum(1).mean()))
        it.update(matrix_world=M.tolist(), dynamic=False, mpm=True, mpm_fit_rms_m=rms)
    spec["physics"] = {"engine": "mpm", "dx_m": dx, "sim_time_s": sim_time, "particles": int(len(X)),
                       "wall_s": time.time() - t0}
    log(f"[mpm] settled in {time.time() - t0:.0f}s")
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--dx", type=float, default=0.002)
    args = ap.parse_args()
    p = Path(args.spec)
    spec = settle(json.loads(p.read_text()), p.parent, args.time, args.dx)
    p.write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist()))


if __name__ == "__main__":
    main()
