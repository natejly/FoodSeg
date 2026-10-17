"""Real geometry for soft foods: irregular surfaces baked into the mesh that synth.mpm saves.

The MPM surface (marching cubes on particle density, Gaussian smoothed) is a smooth dome: its
silhouette, shadow terminators and speculars read as clay. `displace` pushes the vertices along
their normals by multi-octave noise (lumps, crumb, curds), spoon grooves, a spiral swirl and a
scoop, lets the pile's outline wander (lateral noise near the dish), and renormalises the
volume, all before the mesh is written, so masks, depth, ground-truth points and volumes of
every later stage come from the displaced mesh (labels exact by construction).

Per food parameters are `synth.looks.GEOM` (`look["geom"]`); see `looks.geom`. Pure numpy /
scipy / trimesh, no taichi or Blender, so it is unit-testable.
"""
import numpy as np

from .geometry import mesh_volume

U32 = np.uint32


def _hash3(ix, iy, iz, seed):
    h = ix.astype(U32) * U32(73856093) ^ iy.astype(U32) * U32(19349663) ^ iz.astype(U32) * U32(83492791) \
        ^ U32((seed * 2654435761) & 0xFFFFFFFF)
    h = (h ^ (h >> U32(13))) * U32(1274126177)
    h = h ^ (h >> U32(16))
    return (h & U32(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def value_noise(P: np.ndarray, wavelength: float, seed: int) -> np.ndarray:
    """Smooth 3D value noise in -1..1 (quintic interpolation), feature size `wavelength` (m)."""
    q = P / wavelength + (seed % 977) * 0.37
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    s = f * f * f * (f * (f * 6 - 15) + 10)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (s[:, 0] if dx else 1 - s[:, 0]) * (s[:, 1] if dy else 1 - s[:, 1]) * (s[:, 2] if dz else 1 - s[:, 2])
                out = out + w * _hash3(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return 2.0 * out - 1.0


def fbm(P: np.ndarray, wavelength: float, octaves: int, seed: int, gain: float = 0.55, lac: float = 2.1) -> np.ndarray:
    """Fractal sum of value noise, scaled to unit standard deviation over the points."""
    out, a, wl = np.zeros(len(P)), 1.0, wavelength
    for o in range(octaves):
        out += a * value_noise(P, wl, seed + 101 * o)
        a *= gain
        wl /= lac
    return out / max(out.std(), 1e-9)


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _vertex_area(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    return np.bincount(F.ravel(), weights=np.repeat(a / 3.0, 3), minlength=len(V)), a


def _polyline_dist(xy, pts):
    from scipy.spatial import cKDTree
    return cKDTree(pts).query(xy)[0]


def _top_points(V, nz, rng, n=1):
    """Random picks among the upward-facing vertices, biased to the higher ones (spoon marks sit on top)."""
    idx = np.flatnonzero(nz > 0.6)
    if len(idx) == 0:
        return np.zeros((0, 3))
    w = np.clip(V[idx, 2] - V[idx, 2].min(), 1e-4, None) ** 2
    return V[rng.choice(idx, n, p=w / w.sum())]


def spoon_marks(V, nz, g: dict, rng) -> np.ndarray:
    """Height field (m, + up) at the vertices: curved spoon grooves with shoulders, a spiral swirl and a scoop dent."""
    h = np.zeros(len(V))
    xy = V[:, :2]
    for _ in range(int(g.get("grooves", 0))):
        c = _top_points(V, nz, rng)
        if not len(c):
            break
        L = rng.uniform(0.025, 0.06)
        ang, curv = rng.uniform(0, 2 * np.pi), rng.uniform(-1.0, 1.0) / L
        s = np.linspace(0, L, 40)
        th = ang + curv * s
        pts = c[0, :2] + np.stack([np.cumsum(np.cos(th)) * (L / 40), np.cumsum(np.sin(th)) * (L / 40)], 1)
        d = _polyline_dist(xy, pts)
        w = rng.uniform(0.0025, 0.005)
        depth = rng.uniform(0.6, 1.0) * g.get("groove_mm", 1.5) / 1000.0
        taper = 1.0     # grooves fade at the polyline ends via the distance to the ends
        h -= depth * taper * (np.exp(-(d / w) ** 2) - 0.45 * np.exp(-(d / (2.4 * w)) ** 2))
    if g.get("swirl_mm", 0) > 0:                   # the spoon-turned spiral ridge on a mound of mash / yogurt
        c = _top_points(V, nz, rng)
        if len(c):
            dx, dy = xy[:, 0] - c[0, 0], xy[:, 1] - c[0, 1]
            r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
            pitch = rng.uniform(0.006, 0.010)
            env = np.exp(-(r / rng.uniform(0.025, 0.045)) ** 2)
            ph = 2 * np.pi * r / pitch + rng.choice([-1, 1]) * th + rng.uniform(0, 2 * np.pi)
            h += g["swirl_mm"] / 1000.0 * env * np.sign(np.sin(ph)) * np.abs(np.sin(ph)) ** 0.7
    if rng.random() < g.get("scoop", 0.0):         # a serving scoop taken out of the top, rim slightly raised
        c = _top_points(V, nz, rng)
        if len(c):
            R = rng.uniform(0.012, 0.024)
            r = np.hypot(xy[:, 0] - c[0, 0], xy[:, 1] - c[0, 1]) / R
            dent = np.clip(1 - r ** 2, 0, None) ** 1.5
            rim = np.exp(-((r - 1.05) / 0.25) ** 2)
            h += g.get("scoop_mm", 3.0) / 1000.0 * (-dent + 0.35 * rim)
    return h


def displace(V: np.ndarray, F: np.ndarray, g: dict, floor: np.ndarray, seed: int, others: np.ndarray | None = None,
             tol: float = 0.01) -> tuple[np.ndarray, dict]:
    """Irregular surface for one soft food. V, F: closed mesh (metres, outward faces); g: looks.geom
    parameters; floor: dish height under each vertex; others: vertices of neighbouring foods (the
    displacement fades near them so adjacent foods do not interpenetrate). Returns (V', info);
    the volume is within `tol` of the input (renormalised, else the amplitude is halved)."""
    import trimesh
    from trimesh.smoothing import filter_taubin
    rng = np.random.default_rng(seed)
    V0 = V.astype(np.float64)
    vol0 = mesh_volume(V0, F)
    pressed = (V0[:, 2] - floor) < 8e-4                      # bottom pressed onto the dish by mpm.surface_mesh
    m = trimesh.Trimesh(V0.copy(), F, process=False)
    w_top = _smooth(np.asarray(m.vertex_normals)[:, 2], 0.2, 0.5)     # relax the top surface only: flanks / outline stay put
    if g.get("smooth_iter", 0) > 0:                          # relax marching-cubes terraces (volume-preserving Taubin)
        filter_taubin(m, lamb=0.5, nu=-0.53, iterations=int(g["smooth_iter"]))
    Vs = np.asarray(m.vertices, np.float64).copy()
    dl = (Vs - V0) * w_top[:, None]                          # thin slivers / islands must not collapse under the relaxation
    nl = np.linalg.norm(dl, axis=1, keepdims=True)
    Vs = V0 + dl * np.minimum(1.0, 3e-3 / np.maximum(nl, 1e-12))
    Vs[:, 2] = np.where(pressed, V0[:, 2], np.maximum(Vs[:, 2], floor + 3e-4))
    m = trimesh.Trimesh(Vs, F, process=False)
    N = np.asarray(m.vertex_normals, np.float64)
    nz = N[:, 2]
    zh = np.clip(Vs[:, 2] - floor, 0.0, None)                # height above the dish
    amp = g["amp_mm"] / 1000.0
    w_body = _smooth(nz, -0.35, 0.1) * (0.35 + 0.65 * _smooth(nz, 0.1, 0.7))   # none on the bottom, 35 % on flanks
    w_lat = np.ones(len(Vs))                                 # the shared boundary of two foods stays where it is
    if others is not None and len(others):
        from scipy.spatial import cKDTree
        d_oth = cKDTree(others).query(Vs)[0]
        w_body *= _smooth(d_oth, 6e-4, 3.5e-3)
        w_lat = _smooth(d_oth, 2e-3, 9e-3)
    area_v, _ = _vertex_area(Vs, F)
    c_xy = Vs[:, :2].mean(0)                                # outline wander: a smooth radial warp (no fold-overs)
    nh = np.zeros((len(Vs), 3))
    nh[:, :2] = Vs[:, :2] - c_xy
    nh /= np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-9)
    near = _smooth(0.015 - zh, 0.0, 0.012) * (1 - _smooth(nz, 0.5, 0.9))      # flank verts within ~15 mm of the dish
    n_wl = g.get("wl_mm", (12.0, 5.0))
    base = 0.7 * fbm(Vs, n_wl[0] / 1000.0, g.get("octaves", 4), seed) + 0.5 * fbm(Vs, n_wl[1] / 1000.0, 2, seed + 7)
    base = np.clip(base / 1.2, -2.5, 2.5) / 2.5
    edge_n = fbm(Vs, g.get("edge_wl_mm", 28.0) / 1000.0, 2, seed + 13)
    edge_n = np.clip(edge_n, -2.5, 2.5) / 2.5
    spoon = spoon_marks(Vs, nz, g, rng)
    scale = 1.0
    mo = np.asarray(trimesh.Trimesh(V0, F, process=False).face_normals)
    for attempt in range(5):
        d = (amp * base + spoon) * scale
        d = np.minimum(d, 0.5 * zh + 5e-5)                    # never punch through to the dish where the food is thin
        d = np.maximum(d, -0.5 * zh)
        dN = d * w_body
        lat = g.get("edge_mm", 1.5) / 1000.0 * edge_n * near * w_lat * scale
        P = Vs + N * dN[:, None] + nh * lat[:, None]
        P[:, 2] = np.maximum(P[:, 2], floor + 3e-4)
        for _ in range(4):                                    # renormalise the volume by a uniform normal offset
            vol = mesh_volume(P, F)
            wv = w_body * np.minimum(zh / 3e-3, 1.0)
            den = float((area_v * wv).sum())
            if abs(vol / vol0 - 1) < tol * 0.2 or den <= 0:
                break
            P = P + N * (wv * (vol0 - vol) / den)[:, None]
            P[:, 2] = np.maximum(P[:, 2], floor + 3e-4)
        P[pressed, 2] = V0[pressed, 2]
        vol = mesh_volume(P, F)
        # quality: faces whose normal flipped against the input (fold-over / self-intersection)
        mn = np.asarray(trimesh.Trimesh(P, F, process=False).face_normals)
        flipped = float((np.einsum("ij,ij->i", mo, mn) < 0.0).mean())
        if abs(vol / vol0 - 1) < tol and flipped < 0.004:
            break
        scale *= 0.65
    ok = abs(vol / vol0 - 1) < tol and vol > 0
    info = {"vol_change": float(vol / vol0 - 1), "rms_mm": float(np.sqrt(((P - V0) ** 2).sum(1).mean()) * 1e3),
            "max_mm": float(np.linalg.norm(P - V0, axis=1).max() * 1e3), "flipped_frac": flipped,
            "amp_scale": scale, "ok": bool(ok and flipped < 0.01)}
    if not info["ok"]:
        return V0, info
    return P, info
