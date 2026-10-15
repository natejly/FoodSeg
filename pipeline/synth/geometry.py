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
    square: float = 0.0   # 0 = round; else superellipse exponent n (3.5-6): the rim / outer wall are pushed out to
                          # a rounded square, the (flat) well stays round, so z_in / r_at_height still hold

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
        if self.square <= 0:
            return lathe(self.profile(), n_theta)
        V, F = lathe(self.profile(36), 128)    # <= ~20k faces: Bullet's mesh collider stalls on much more (rim corners need azimuth steps)
        return self.squarify(V), F

    def squarify(self, V: np.ndarray) -> np.ndarray:
        """Push the lathe's rim and outer wall outwards towards a superellipse outline (same z); the
        inside of the well, where food rests, is flat or unchanged."""
        n = float(self.square)
        r = np.hypot(V[:, 0], V[:, 1])
        th = np.arctan2(V[:, 1], V[:, 0])
        s_th = (np.abs(np.cos(th)) ** n + np.abs(np.sin(th)) ** n) ** (-1.0 / n)     # 1 on the axes, max on diagonals
        r0, r1 = 0.55 * self.well_r, self.R + self.rim_w
        t = np.clip((r - r0) / max(r1 - r0, 1e-9), 0, 1)
        g = t * t * (3 - 2 * t)
        k = 1 + (s_th - 1) * g
        W = V.copy()
        W[:, 0] *= k
        W[:, 1] *= k
        return W

    def outline_norm(self) -> float:
        """Exponent for the p-norm used by the render shader (outline-following rim bands); 2 = round."""
        return float(self.square) if self.square > 0 else 2.0

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


def sample_container(rng: np.random.Generator, kind: str | None = None, p_square: float = 0.0) -> Container:
    """Plausible dinnerware. Ranges from common plates (Ø 20-28 cm, 1.5-3 cm tall)
    and cereal/soup bowls (Ø 12-18 cm, 4-8 cm tall)."""
    kind = kind or rng.choice(["plate", "bowl"])
    if kind == "bowl" and rng.random() < 0.3:       # wide shallow pasta / cafeteria bowl
        R = rng.uniform(0.085, 0.105)
        return Container("bowl", R=R, H=rng.uniform(0.032, 0.05), floor=rng.uniform(0.006, 0.01),
                         q=rng.uniform(2.6, 4.5), rim_w=rng.uniform(0.004, 0.01),
                         wall=rng.uniform(0.003, 0.005), foot_r=R * rng.uniform(0.4, 0.6))
    if kind == "bowl":
        R = rng.uniform(0.055, 0.085)
        return Container("bowl", R=R, H=rng.uniform(0.04, 0.075), floor=rng.uniform(0.006, 0.012),
                         q=rng.uniform(1.6, 4.0), rim_w=rng.uniform(0.003, 0.008),
                         wall=rng.uniform(0.003, 0.006), foot_r=R * rng.uniform(0.45, 0.65))
    R = rng.uniform(0.085, 0.125)
    sq = float(rng.uniform(3.5, 6.0)) if rng.random() < p_square else 0.0
    return Container("plate", R=R * (0.93 if sq else 1.0), H=rng.uniform(0.014, 0.03), floor=rng.uniform(0.004, 0.008),
                     well_r=R * (0.93 if sq else 1.0) * rng.uniform(0.6, 0.85), rim_w=rng.uniform(0.008, 0.025),
                     wall=rng.uniform(0.003, 0.005), foot_r=R * rng.uniform(0.55, 0.75) * (0.93 if sq else 1.0), square=sq)


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


# ---- harder scenes: touching pieces, toppings -------------------------------

def grid_solid(top: np.ndarray, bot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed solid between two (n, m, 3) vertex grids with matching boundaries.

    The grid is a topological box: top sheet, bottom sheet and four side strips, so
    the mesh is closed even when rows collapse to a point (a wedge apex) or the two
    sheets meet (a mound edge); degenerate faces add no volume.
    """
    n, m = top.shape[:2]
    V = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)])
    t = np.arange(n * m).reshape(n, m)
    b = t + n * m
    q = lambda g: np.stack([g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]], -1).reshape(-1, 4)
    quads = [q(t), q(b)[:, ::-1]]
    loop = np.concatenate([t[:, 0], t[-1, 1:], t[-2::-1, -1], t[0, -2:0:-1]])   # CCW boundary
    loopb = loop + n * m
    a, c = np.r_[loop], np.roll(loop, -1)
    quads.append(np.stack([c, a, a + n * m, c + n * m], 1))
    Q = np.concatenate(quads)
    F = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    # weld coincident vertices (apex, rim, seam) and drop the faces that collapse: Bullet's
    # triangle-mesh collider stalls on zero-area triangles
    _, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    V, F = V[np.sort(first)], rank[inv.reshape(-1)][F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    F = F[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-14]
    if mesh_volume(V, F) < 0:
        F = F[:, ::-1].copy()
    return V, F


@dataclass
class Mound:
    """A heap of amorphous food in container `c`: its top over a disc of radius rf
    centred at `centre`, its bottom the container's inner surface.

    top = max(level, z_in) + (mound + wobble * bump) * (1 - rho^p), rho = r / rf, so on
    a bowl filled to `level` it meets the wall at rf, and on a plate (level None) it
    tapers to the plate surface. Pieces of one mound split by angle touch exactly.
    """
    rf: float
    mound: float
    p: float = 2.0
    level: float | None = None
    wobble: float = 0.0
    centre: tuple[float, float] = (0.0, 0.0)
    seed: int = 0
    lift: float = 5e-4
    relief: float = 0.0          # amplitude (m) of the food's own lumpy texture (grains, flakes, curds)
    relief_scale: float = 0.01   # its wavelength (m)
    # pieces of one heap: within `blend` (m) of the wedge edges (th_a, th_b callables) the top
    # eases to the heap's shared profile (mound, p), so neighbours meet without a cliff
    shared: tuple[float, float] | None = None
    edges: tuple | None = None
    blend: float = 0.015

    def _bump(self, rho, th):
        rng = np.random.default_rng(self.seed)
        k, ph, amp = rng.integers(2, 6, 3), rng.random(3) * 2 * np.pi, rng.normal(0, 1, 3)
        return sum(a * rho ** kk * np.cos(kk * th + p) * np.sin(np.pi * rho * (i + 1))
                   for i, (a, kk, p) in enumerate(zip(amp, k, ph))) / 3

    def base(self, c: Container, x, y):
        return c.z_in(np.hypot(x, y)) + self.lift

    def top(self, c: Container, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        lx, ly = x - self.centre[0], y - self.centre[1]
        rho = np.clip(np.hypot(lx, ly) / self.rf, 0, 1)
        th = np.arctan2(ly, lx)
        base = self.base(c, x, y)
        lev = base if self.level is None else np.maximum(self.level + self.lift, base)
        s = 1 - rho ** self.p
        z = lev + (self.mound + self.wobble * self._bump(rho, th)) * s
        if self.shared is not None and self.edges is not None:
            r = rho * self.rf
            ang = np.minimum(np.mod(th - self.edges[0](r), 2 * np.pi), np.mod(self.edges[1](r) - th, 2 * np.pi))
            w = np.clip(r * ang / self.blend, 0, 1)
            w = w * w * (3 - 2 * w)
            z_sh = lev + self.shared[0] * (1 - rho ** self.shared[1])
            z = z_sh + (z - z_sh) * w
        if self.relief:
            z = z + self.relief * relief_noise(x, y, self.seed, self.relief_scale) * np.sqrt(s)
        z = np.maximum(z, base + 1e-4)
        return np.where(rho >= 1, base, z)

    def piece(self, c: Container, th_a, th_b, n_r: int = 56, m: int = 64):
        """Closed mesh of the wedge th_a(r) <= theta <= th_b(r) (callables of local radius)."""
        rs = np.linspace(0, self.rf, n_r)
        u = np.linspace(0, 1, m)
        th = th_a(rs)[:, None] + (th_b(rs) - th_a(rs))[:, None] * u[None]
        x = self.centre[0] + rs[:, None] * np.cos(th)
        y = self.centre[1] + rs[:, None] * np.sin(th)
        top = np.stack([x, y, self.top(c, x, y)], -1)
        bot = np.stack([x, y, self.base(c, x, y)], -1)
        top[-1] = bot[-1]                      # the rim of the heap sits on the dish
        return grid_solid(top, bot)


def relief_noise(x, y, seed: int, scale: float, n: int = 16):
    """Isotropic lumpy field ~N(0, 1) in world x, y: random plane waves of wavelength ~scale."""
    rng = np.random.default_rng(seed + 7)
    a, lam, ph = rng.uniform(0, 2 * np.pi, n), scale * rng.uniform(0.6, 1.6, n), rng.uniform(0, 2 * np.pi, n)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return sum(np.cos(2 * np.pi / l * (x * np.cos(t) + y * np.sin(t)) + p) for t, l, p in zip(a, lam, ph)) \
        * np.sqrt(2 / n)


def split_angles(rng: np.random.Generator, n: int, rf: float, min_gap: float = 1.2, amp: float = 0.2,
                 bend: float = 0.3):
    """n wavy, curving radial boundaries splitting a disc into n wedges; returns callables th_k(r),
    k = 0..n (th_n = th_0 + 2 pi), so wedge k spans [th_k, th_{k+1}]. Each boundary strays at most
    amp + bend < min_gap / 2 from its base angle, so neighbours never cross."""
    if n == 1:
        a0 = rng.uniform(0, 2 * np.pi)
        return [lambda r, a=a0: np.full_like(r, a), lambda r, a=a0: np.full_like(r, a + 2 * np.pi)]
    while True:
        w = rng.dirichlet(np.ones(n) * 3) * 2 * np.pi
        if w.min() >= min_gap:
            break
    base = rng.uniform(0, 2 * np.pi) + np.r_[0, np.cumsum(w)[:-1]]
    waves = [(rng.uniform(-amp, amp), rng.uniform(0.5, 1.5) * rf, rng.uniform(0, 2 * np.pi), rng.uniform(-bend, bend))
             for _ in range(n)]
    # a curve (food pushed to one side) plus a wobble; both vanish at the centre
    fns = [lambda r, b=b, a=a, lam=lam, ph=ph, cb=cb: b + cb * (r / rf) ** 1.5 + a * np.sin(2 * np.pi * r / lam + ph) * (r / rf)
           for b, (a, lam, ph, cb) in zip(base, waves)]
    return fns + [lambda r, f=fns[0]: f(r) + 2 * np.pi]


def topping_mesh(c: Container, under: Mound, centre: tuple[float, float], radius: float, thick: float,
                 seed: int = 0, n_r: int = 20, m: int = 48, lift: float = 3e-4):
    """A sauce / cheese blob lying on `under`'s top surface: irregular disc footprint,
    thickness thick * (1 - rho^3) * (1 + small noise)."""
    rng = np.random.default_rng(seed)
    phi = np.linspace(0, 2 * np.pi, m)                     # seam duplicated: zero-volume wall
    k, ph, a = rng.integers(2, 5, 2), rng.random(2) * 2 * np.pi, rng.uniform(0.05, 0.2, 2)
    Rphi = radius * (1 + sum(ai * np.sin(ki * phi + pi) for ai, ki, pi in zip(a, k, ph)))
    rho = np.linspace(0, 1, n_r)
    x = centre[0] + rho[:, None] * Rphi[None] * np.cos(phi)[None]
    y = centre[1] + rho[:, None] * Rphi[None] * np.sin(phi)[None]
    zb = under.top(c, x, y) + lift
    s = (1 - rho ** 3)[:, None] * (1 + 0.15 * np.sin(3 * phi + ph[0])[None] * rho[:, None])
    top = np.stack([x, y, zb + thick * s], -1)
    bot = np.stack([x, y, zb], -1)
    top[-1] = bot[-1]
    return grid_solid(top, bot)


def leaf_sheet(rng: np.random.Generator, length: float, width: float, n_u: int = 12, n_v: int = 5):
    """One leaf as an open sheet (cloth-simulated in render.py, then given thickness):
    tapered outline, cupped across the midrib, bent along it, ruffled edge. Leaf frame:
    x along the midrib, z up. The tips stay > ~40 % of full width so no two vertices of a
    leaf come closer than cloth self-collision's distance (a pointed tip made it explode)."""
    u = np.linspace(0, 1, n_u)
    v = np.linspace(-1, 1, n_v)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    half_w = 0.5 * width * np.sin(np.pi * (0.12 + 0.76 * U)) ** 0.7
    ruffle = 1 + 0.12 * np.sin(rng.uniform(6, 14) * np.pi * U + rng.uniform(0, 6)) * np.abs(Vv)
    x = length * (U - 0.5)
    y = half_w * Vv * ruffle
    z = rng.uniform(0.1, 0.35) * half_w * Vv ** 2 + rng.uniform(-0.15, 0.15) * length * (U - 0.5) ** 2
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    idx = np.arange(n_u * n_v).reshape(n_u, n_v)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    return P, np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def salad_sheets(rng: np.random.Generator, n: int, centre, base_z: float, spread: float,
                 kind: str = "mixed", gap: tuple[float, float] = (0.012, 0.016)) -> tuple[np.ndarray, np.ndarray]:
    """n leaves stacked in a loose column above `centre` (they drape as cloth when simulated),
    random yaw / tilt each. Returns one mesh of disconnected sheets. Leaves stay >= 3 cm:
    shrinking them packs the 12 x 5 grid closer than cloth self-collision's distance and
    the simulation explodes (tried: 2 cm leaves sized to small bowls scattered everywhere)."""
    Vs, Fs, off, z = [], [], 0, base_z
    for _ in range(n):
        L = rng.uniform(0.035, 0.07) if kind != "arugula" else rng.uniform(0.03, 0.05)
        W = L * (rng.uniform(0.5, 0.8) if kind != "arugula" else rng.uniform(0.25, 0.4))
        P, F = leaf_sheet(rng, L, W)
        # leaves must not intersect at the start (cloth self-collision explodes on that):
        # mild tilt, and a vertical gap larger than a tilted leaf's height
        a, t = rng.uniform(0, 2 * np.pi), rng.normal(0, 0.15)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        z += rng.uniform(*gap)
        P = P @ (Rz @ Rx).T + np.r_[np.asarray(centre) + rng.normal(0, spread / 2, 2), z]
        Vs.append(P)
        Fs.append(F + off)
        off += len(P)
    return np.concatenate(Vs), np.concatenate(Fs)


def lathe_solid(profile: np.ndarray, n_theta: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Public alias for small props (mugs): lathe a closed (r, z) profile."""
    return lathe(profile, n_theta)
