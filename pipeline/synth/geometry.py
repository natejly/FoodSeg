"""Procedural solids of revolution: plates, bowls, and food that fills a container.

Everything is in a Z-up frame with the table at z = 0 and the container axis on
z, in metres. Meshes are (vertices, faces) numpy arrays, watertight and outward
oriented, so their volume is exact (`mesh_volume`).

A container is described by its inner profile z_in(r) for r in [0, R] (the
surface food sits on), a flat rim of width `rim_w` at height H = z_in(R), and an
outer wall down to a foot. `fill_mesh` builds the food that fills a container up
to height h with a mound on top, the bowl case that single-plane methods get wrong.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x / 2.x

# ---- lathe -----------------------------------------------------------------

def lathe(profile: np.ndarray, n_theta: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Revolve an (r, z) polyline that starts and ends on the axis (r = 0) about z.

    Returns a closed triangle mesh, faces oriented outward.
    """
    profile = np.asarray(profile, float)
    keep = np.r_[True, np.any(np.diff(profile, axis=0) != 0, axis=1)]
    profile = profile[keep]                      # drop repeated points (degenerate faces)
    assert profile[0, 0] == 0 and profile[-1, 0] == 0, "profile must start and end on the axis"
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    verts, rings = [], []
    for r, z in profile:
        if r == 0:
            rings.append([len(verts)])
            verts.append([0.0, 0.0, z])
        else:
            rings.append(list(range(len(verts), len(verts) + n_theta)))
            verts.extend(np.stack([r * c, r * s, np.full(n_theta, z)], 1))
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(n_theta):
            k = (j + 1) % n_theta
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                faces.append([a[0], b[j], b[k]])
            elif len(b) == 1:
                faces.append([a[j], b[0], a[k]])
            else:
                faces.append([a[j], b[j], b[k]])
                faces.append([a[j], b[k], a[k]])
    V, F = np.asarray(verts), np.asarray(faces, dtype=np.int64)
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh (divergence theorem), m^3 if V is in m."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def write_obj(path, V: np.ndarray, F: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F + 1:
            f.write(f"f {t[0]} {t[1]} {t[2]}\n")


# ---- containers --------------------------------------------------------------

@dataclass
class Container:
    """Axisymmetric plate or bowl. All lengths in metres.

    kind "bowl":  z_in(r) = floor + (H - floor) (r / R)^q
    kind "plate": flat well z = floor for r < well_r, smoothstep up to H at R
    """
    kind: str
    R: float              # inner radius at the rim
    H: float              # rim height above the table
    floor: float          # inner floor height above the table (base thickness)
    q: float = 2.0        # bowl wall exponent (2 = paraboloid, large = flat floor + steep wall)
    well_r: float = 0.0   # plate: radius of the flat well
    rim_w: float = 0.012  # flat rim width
    wall: float = 0.004   # wall thickness
    foot_r: float = 0.0   # outer radius at the table (default: 0.6 R)

    def z_in(self, r: np.ndarray) -> np.ndarray:
        r = np.clip(np.asarray(r, float), 0, self.R)
        if self.kind == "bowl":
            return self.floor + (self.H - self.floor) * (r / self.R) ** self.q
        t = np.clip((r - self.well_r) / max(self.R - self.well_r, 1e-9), 0, 1)
        return self.floor + (self.H - self.floor) * t * t * (3 - 2 * t)

    def r_at_height(self, h: float) -> float:
        """Radius where the inner surface reaches height h (h in [floor, H])."""
        h = float(np.clip(h, self.floor, self.H))
        if self.kind == "bowl":
            return self.R * ((h - self.floor) / (self.H - self.floor)) ** (1 / self.q)
        rs = np.linspace(self.well_r, self.R, 4001)
        return float(np.interp(h, self.z_in(rs), rs))

    def profile(self, n: int = 64) -> np.ndarray:
        rs = np.linspace(0, self.R, n)
        if self.kind == "plate":   # keep a vertex at the well edge
            rs = np.unique(np.concatenate([rs, [self.well_r]]))
        inner = np.stack([rs, self.z_in(rs)], 1)
        Ro = self.R + self.rim_w
        foot = self.foot_r or 0.6 * self.R
        # outer wall: offset of the inner wall, ending at the foot ring on the table
        ro = np.linspace(Ro, foot, n)
        zo = np.interp(ro, rs + self.wall, self.z_in(rs) - self.wall)
        zo = np.clip(zo, 0.0, self.H)
        outer = np.stack([ro, zo], 1)
        return np.concatenate([inner, [[Ro, self.H]], outer[1:], [[foot, 0.0], [0.0, 0.0]]])

    def mesh(self, n_theta: int = 96) -> tuple[np.ndarray, np.ndarray]:
        return lathe(self.profile(), n_theta)

    def capacity_ml(self, h: float | None = None) -> float:
        """Volume between the inner surface and the plane z = h (default: the rim)."""
        h = self.H if h is None else h
        rs = np.linspace(0, self.r_at_height(h), 4001)
        dz = np.clip(h - self.z_in(rs), 0, None)
        return float(_trapz(2 * np.pi * rs * dz, rs) * 1e6)

    def surface_points(self, n: int = 20000, table_r: float = 0.15, rng=None) -> np.ndarray:
        """Points on the top surface (inner, rim) plus surrounding table, for ContainerTemplate."""
        rng = np.random.default_rng(rng)
        Ro = self.R + self.rim_w
        r = table_r * np.sqrt(rng.random(n))
        th = rng.random(n) * 2 * np.pi
        z = np.where(r <= self.R, self.z_in(r), np.where(r <= Ro, self.H, 0.0))
        # outer wall is visible from oblique views; approximate it as the rim-to-table drop
        return np.stack([r * np.cos(th), r * np.sin(th), z], 1)

    def to_json(self) -> dict:
        return asdict(self)


def sample_container(rng: np.random.Generator, kind: str | None = None) -> Container:
    """Plausible dinnerware. Ranges from common plates (Ø 20-28 cm, 1.5-3 cm tall)
    and cereal/soup bowls (Ø 12-18 cm, 4-8 cm tall)."""
    kind = kind or rng.choice(["plate", "bowl"])
    if kind == "bowl":
        R = rng.uniform(0.055, 0.085)
        return Container("bowl", R=R, H=rng.uniform(0.04, 0.075), floor=rng.uniform(0.006, 0.012),
                         q=rng.uniform(1.6, 4.0), rim_w=rng.uniform(0.003, 0.008),
                         wall=rng.uniform(0.003, 0.006), foot_r=R * rng.uniform(0.45, 0.65))
    R = rng.uniform(0.085, 0.125)
    return Container("plate", R=R, H=rng.uniform(0.014, 0.03), floor=rng.uniform(0.004, 0.008),
                     well_r=R * rng.uniform(0.6, 0.85), rim_w=rng.uniform(0.008, 0.025),
                     wall=rng.uniform(0.003, 0.005), foot_r=R * rng.uniform(0.55, 0.75))


def container_from_json(d: dict) -> Container:
    """Accepts a spec's container dict (extra keys like colour or mesh path are ignored)."""
    names = {f.name for f in fields(Container)}
    return Container(**{k: v for k, v in d.items() if k in names})


# ---- container-conforming food ----------------------------------------------

def fill_mesh(c: Container, h: float, mound: float, wobble: float = 0.0, seed: int = 0,
              n_r: int = 48, n_theta: int = 96, lift: float = 5e-4) -> tuple[np.ndarray, np.ndarray]:
    """Food filling container `c` to height h with a mound of height `mound` on top.

    Bottom = the container's inner surface (lifted by `lift` so the two don't
    coincide). Top = h + mound (1 - rho^2) + wobble * low-frequency noise, which
    vanishes at the fill edge so the solid stays closed.
    """
    rf = c.r_at_height(h)
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rs = np.linspace(0, rf, n_r)
    rho = rs / rf
    # low-frequency angular/radial bumps
    k = rng.integers(2, 6, 3)
    ph = rng.random(3) * 2 * np.pi
    amp = rng.normal(0, 1, 3)
    # rho^k cos(k theta) is smooth at the axis (no crease); sin(.) adds radial ripples
    bump = sum(a * rho[:, None] ** kk * np.cos(kk * th[None] + p) * np.sin(np.pi * rho[:, None] * (i + 1))
               for i, (a, kk, p) in enumerate(zip(amp, k, ph))) / 3
    top = h + mound * (1 - rho[:, None] ** 2) + wobble * bump * (1 - rho[:, None] ** 2)
    bot = np.broadcast_to(c.z_in(rs)[:, None] + lift, top.shape)
    top = np.maximum(top, bot + 1e-4)
    top[-1] = bot[-1]          # meet at the edge ring

    def grid(z, flip):
        V = [[0, 0, z[0, 0]]]
        for i in range(1, n_r):
            V += [[rs[i] * np.cos(t), rs[i] * np.sin(t), z[i, j]] for j, t in enumerate(th)]
        F = []
        for j in range(n_theta):
            F.append([0, 1 + j, 1 + (j + 1) % n_theta])
        for i in range(1, n_r - 1):
            a, b = 1 + (i - 1) * n_theta, 1 + i * n_theta
            for j in range(n_theta):
                k2 = (j + 1) % n_theta
                F += [[a + j, b + j, b + k2], [a + j, b + k2, a + k2]]
        F = np.asarray(F)
        return np.asarray(V, float), (F[:, ::-1] if flip else F)

    Vt, Ft = grid(top, False)
    Vb, Fb = grid(bot, True)
    # share the outer ring: map the bottom's last ring onto the top's
    nb = len(Vb)
    last = np.arange(nb - n_theta, nb)
    remap = np.arange(nb) + len(Vt)
    remap[last] = np.arange(len(Vt) - n_theta, len(Vt))
    keep = np.ones(nb, bool)
    keep[last] = False
    # compact bottom vertex indices
    idx = np.cumsum(keep) - 1 + len(Vt)
    remap[keep] = idx[keep]
    V = np.concatenate([Vt, Vb[keep]])
    F = np.concatenate([Ft, remap[Fb]])
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


# ---- simple standalone food -------------------------------------------------

def ellipsoid_mesh(a: float, b: float, c: float, n: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Ellipsoid with semi-axes a, b, c, resting on z = 0."""
    t = np.linspace(0, np.pi, n)
    prof = np.stack([np.sin(t), -np.cos(t)], 1)
    prof[0, 0] = prof[-1, 0] = 0.0
    V, F = lathe(prof, 2 * n)
    V = V * np.array([a, b, c])
    V[:, 2] += c
    return V, F


def box_mesh(x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    V = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    V = (V - [0.5, 0.5, 0]) * [x, y, z]
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
