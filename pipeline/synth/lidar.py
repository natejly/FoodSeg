"""Turn a clean rendered depth map into something that looks like ARKit sceneDepth.

ARKit gives 256x192 float32 depth plus a 0/1/2 confidence map. Pixels it isn't
sure about still hold a value (often smeared from neighbours), they are just
marked low confidence. This model reproduces that, step by step (see
docs/synthetic_data.md, "Faking iPad LiDAR depth"):

  1. area-downsample the render to 256x192 (mixed pixels at edges, like dToF);
  2. sub-pixel lateral jitter;
  3. range-dependent axial noise: spatially correlated white noise + a bias field that is
     PERSISTENT per scan (a smooth 3-D field in world coordinates, sampled at each pixel's
     back-projected point, so multi-view fusion cannot average it away) + a small per-frame part;
  4. edge bleeding toward the RGB edge (Apple fuses RGB into the depth);
  5. dropout on dark, specular/wet and grazing-angle surfaces -> filled (smeared toward the
     farther neighbour, like ARKit's background bleed), conf 0; translucent foods (purees,
     porridge, soup) read slightly deep (subsurface delay);
  6. confidence 2/1/0 from depth discontinuities and incidence angle.

Every number lives in LidarParams. The defaults are PLACEHOLDERS from the
literature (Kim 2023: ~0.5 mm sigma, <=3 mm bias at 30-40 cm; Luetzenburg 2021);
refit them from the iPad calibration scans (docs/collection_protocol.md, cal_*)
and record the fitted values in gt.json via `asdict(params)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

D_W, D_H = 256, 192


@dataclass
class LidarParams:
    sigma0_m: float = 0.0008        # axial noise at z0
    sigma_z2: float = 0.004         # + sigma_z2 * (z - z0)^2
    z0_m: float = 0.30
    bias_sigma_m: float = 0.0012    # amplitude of the smooth bias field
    bias_scale_px: float = 25.0     # its spatial correlation length (depth pixels)
    lateral_px: float = 0.25        # sub-pixel jitter
    bleed_px: float = 1.0           # edge bleeding radius (depth pixels)
    dark_lum: float = 0.06          # luminance below which returns get weak
    p_drop_dark: float = 0.3
    grazing_deg: float = 75.0       # incidence angle beyond which returns drop
    p_drop_grazing: float = 0.5
    edge_jump_m: float = 0.01       # depth jump marking a discontinuity
    min_range_m: float = 0.18       # LiDAR bias explodes below ~20 cm
    smooth_factor: float = 0.4      # smooth_depth keeps this fraction of the white noise
    noise_blur_px: float = 1.0      # spatial correlation of the white noise (sigma, depth px); std is kept
    drop_mult: float = 1.0          # scales every dropout probability (per-scan)
    # persistent bias field (world frame); legacy per-frame field when persistent=False or no pose is given
    persistent: bool = True
    bias_amp_m: float = 0.0015      # rms amplitude of the persistent field (per scan 0.5-3 mm)
    bias_mix: tuple = (0.5, 0.5)    # share of the ~6 mm and ~30 mm spatial-scale parts (variance)
    bias_innov_frac: float = 0.2    # per-frame innovation, fraction of the amplitude
    field_seed: int = 0
    spec_lum: float = 0.95          # specular / wet highlight: luminance above this drops out ...
    p_drop_spec: float = 0.3        # ... with this probability
    far_bias: float = 0.7           # dropout fill: mean weight on the farther neighbour (vs nearest valid)
    translucent_bias_m: float = 0.002   # + depth on translucent foods (purees / porridge / soups)

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "LidarParams":
        """Per-scan noise parameters (record asdict() in gt.json)."""
        return cls(sigma0_m=float(rng.uniform(0.0008, 0.003)), noise_blur_px=float(rng.uniform(1.0, 1.5)),
                   drop_mult=float(rng.uniform(0.5, 2.0)), bias_amp_m=float(rng.uniform(0.0005, 0.003)),
                   bias_mix=(lambda a: (a, 1 - a))(float(rng.uniform(0.3, 0.7))),
                   field_seed=int(rng.integers(1 << 30)), translucent_bias_m=float(rng.uniform(0.001, 0.003)),
                   far_bias=float(rng.uniform(0.5, 0.9)))


# ---- persistent world-frame bias field ---------------------------------------------------------
_EXT = (0.25, 0.25, 0.25)          # field covers x,y in [-0.25, 0.25] m, z in [-0.02, 0.23] m around the dish
_Z0 = -0.02


@lru_cache(maxsize=8)
def _field_grid(seed: int, cell_m: float) -> np.ndarray:
    n = [int(np.ceil(2 * _EXT[0] / cell_m)) + 1, int(np.ceil(2 * _EXT[1] / cell_m)) + 1, int(np.ceil(_EXT[2] / cell_m)) + 1]
    g = np.random.default_rng([seed, int(cell_m * 1e4)]).normal(size=n).astype(np.float32)
    g = ndimage.gaussian_filter(g, 1.0, mode="wrap")          # correlation length ~ cell_m * 1.5
    return g / (g.std() + 1e-9)


def world_bias(xyz: np.ndarray, p: LidarParams) -> np.ndarray:
    """Unit-rms smooth bias at world points (..., 3), metres in/out, ~6 mm and ~30 mm scales."""
    out = 0.0
    for cell, w in zip((0.006, 0.030), p.bias_mix):
        g = _field_grid(p.field_seed, cell)
        idx = np.stack([(xyz[..., 0] + _EXT[0]) / cell, (xyz[..., 1] + _EXT[1]) / cell, (xyz[..., 2] - _Z0) / cell])
        out = out + np.sqrt(w) * ndimage.map_coordinates(g, idx.reshape(3, -1), order=1, mode="nearest").reshape(xyz.shape[:-1])
    return out.astype(np.float32)


def translucent_ids(items: list[dict]) -> list[int]:
    """Item labels (spec['items']) whose food reads deep under LiDAR: purees, porridge, soups."""
    from .looks import is_translucent
    return [int(it["id"]) for it in items if is_translucent(it.get("category") or it.get("name") or "")]


# ---- pose drift (tracking error of the exported camera poses) ----------------------------------
@dataclass
class PoseNoiseParams:
    enabled: bool = True
    trans_step_m: float = 0.0015     # per-frame random-walk translation step (rms)
    rot_step_deg: float = 0.1        # per-frame random-walk rotation step (rms)
    momentum: float = 0.6            # AR(1) smoothness of the walk (0 = white steps)
    jump_p: float = 0.05             # per-frame probability of a tracking jump
    jump_m: tuple = (0.005, 0.010)   # jump translation magnitude range

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "PoseNoiseParams":
        return cls(trans_step_m=float(rng.uniform(0.0005, 0.003)), rot_step_deg=float(rng.uniform(0.05, 0.2)),
                   momentum=float(rng.uniform(0.3, 0.8)))


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def perturb_poses(c2w_list, rng: np.random.Generator, params: PoseNoiseParams | None = None) -> list:
    """Camera-to-world poses as a tracker would report them: the true pose composed with an
    accumulated world-frame drift D_i (a smooth SE(3) random walk, occasional jumps).
    Export these; keep GT (depth_clean, masks, points) in the true frame."""
    params = params or PoseNoiseParams()
    T = [np.asarray(t, float) for t in c2w_list]
    if not params.enabled:
        return [t.copy() for t in T]
    R_d, t_d = np.eye(3), np.zeros(3)
    v_t, v_r = np.zeros(3), np.zeros(3)
    m = params.momentum
    out = []
    for i, Ti in enumerate(T):
        if i > 0:
            s = np.sqrt(1 - m * m)
            v_t = m * v_t + s * rng.normal(0, params.trans_step_m / np.sqrt(3), 3)
            v_r = m * v_r + s * rng.normal(0, np.deg2rad(params.rot_step_deg) / np.sqrt(3), 3)
            t_d = t_d + v_t
            R_d = _rodrigues(v_r) @ R_d
            if rng.random() < params.jump_p:
                d = rng.normal(size=3)
                t_d = t_d + d / np.linalg.norm(d) * rng.uniform(*params.jump_m)
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = R_d, t_d
        out.append(D @ Ti)
    return out


def _area_resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((w, h), Image.BOX))


def downsample_depth(depth: np.ndarray, w: int = D_W, h: int = D_H) -> np.ndarray:
    """Area-average valid depth into the LiDAR grid (0 = no return)."""
    valid = (depth > 0).astype(np.float32)
    s = _area_resize(depth * valid, w, h)
    n = _area_resize(valid, w, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(n > 0.5, s / np.maximum(n, 1e-6), 0.0)
    return d.astype(np.float32)


def incidence_deg(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Angle between the view ray and the surface normal from depth gradients, degrees."""
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    P = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cos = np.abs((n * ray).sum(-1))
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def simulate(depth_hr: np.ndarray, rgb: np.ndarray, K_hr: np.ndarray, rng: np.random.Generator,
             p: LidarParams = LidarParams(), c2w: np.ndarray | None = None, seg: np.ndarray | None = None,
             translucent: tuple | list = ()) -> dict[str, np.ndarray]:
    """depth_hr: render-res z-depth (m, 0 = none); rgb: render-res uint8; K_hr: render-res K.
    c2w: this frame's TRUE camera-to-world (Blender convention, world = spec frame); with it the bias
    field is persistent per scan (world frame), else the legacy per-frame field is used.
    seg: render-res label image and translucent: labels (translucent_ids) that read deep.

    Returns depth, smooth_depth (256x192 float32), conf (uint8) and clean (256x192, no noise).
    """
    H, W = depth_hr.shape
    clean = downsample_depth(depth_hr)
    S = np.diag([D_W / W, D_H / H, 1.0])
    K = S @ K_hr
    valid = clean > 0

    # 2. lateral jitter
    yy, xx = np.mgrid[0:D_H, 0:D_W].astype(np.float32)
    jx, jy = rng.normal(0, p.lateral_px, (2, D_H, D_W))
    d = ndimage.map_coordinates(clean, [yy + jy, xx + jx], order=0, mode="nearest")

    # 4. edge bleed: near discontinuities, blend toward the local median (RGB-guided smear)
    jump = ndimage.maximum_filter(clean, 3) - ndimage.minimum_filter(clean, 3) > p.edge_jump_m
    if p.bleed_px > 0:
        med = ndimage.median_filter(d, size=int(2 * p.bleed_px + 1))
        a = rng.uniform(0.3, 1.0, d.shape)
        d = np.where(jump, a * med + (1 - a) * d, d)

    # 3. axial noise: spatially correlated white part + bias field
    sig = p.sigma0_m + p.sigma_z2 * (d - p.z0_m) ** 2
    wn = rng.normal(0, 1, d.shape)
    if p.noise_blur_px > 0:
        wn = ndimage.gaussian_filter(wn, p.noise_blur_px)
        wn /= wn.std() + 1e-12
    white = wn * sig
    if p.persistent and c2w is not None:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cam = np.stack([(xx + 0.5 - cx) / fx * d, -(yy + 0.5 - cy) / fy * d, -d], -1)      # Blender camera axes
        c2w = np.asarray(c2w, float)
        xyz = cam @ c2w[:3, :3].T + c2w[:3, 3]
        bias = p.bias_amp_m * world_bias(xyz, p)
        if p.bias_innov_frac > 0:
            inn = ndimage.gaussian_filter(rng.normal(0, 1, d.shape), p.bias_scale_px / 4)
            bias = bias + p.bias_amp_m * p.bias_innov_frac * inn / (inn.std() + 1e-12)
    else:
        bias = ndimage.gaussian_filter(rng.normal(0, 1, d.shape), p.bias_scale_px)
        bias *= p.bias_sigma_m / (bias.std() + 1e-12)
    if seg is not None and len(translucent) and p.translucent_bias_m:
        sd_ = np.asarray(Image.fromarray(seg.astype(np.uint8)).resize((D_W, D_H), Image.NEAREST))
        bias = bias + p.translucent_bias_m * np.isin(sd_, list(translucent))
    noisy = d + white + bias
    smooth = d + p.smooth_factor * white + bias

    # 5. dropout: dark, specular / wet highlights, grazing angles
    lum = _area_resize(rgb.astype(np.float32).mean(-1) / 255.0, D_W, D_H)
    inc = incidence_deg(clean, K)
    m_ = p.drop_mult
    drop = ((lum < p.dark_lum) & (rng.random(d.shape) < min(1.0, m_ * p.p_drop_dark))) | \
           ((lum > p.spec_lum) & (rng.random(d.shape) < min(1.0, m_ * p.p_drop_spec))) | \
           ((inc > p.grazing_deg) & (rng.random(d.shape) < min(1.0, m_ * p.p_drop_grazing))) | \
           (clean < p.min_range_m) | ~valid
    if drop.any() and (~drop).any():
        # ARKit still reports a value, smeared from the neighbourhood and biased to the background:
        # blend of the nearest valid return and the farthest valid return within 2 px
        idx = ndimage.distance_transform_edt(drop, return_distances=False, return_indices=True)
        near = noisy[tuple(idx)]
        far = ndimage.maximum_filter(np.where(drop, -np.inf, noisy), size=5, mode="nearest")
        far = np.where(np.isfinite(far), far, near)
        w = np.clip(rng.normal(p.far_bias, 0.25, d.shape), 0, 1)
        fill = w * np.maximum(far, near) + (1 - w) * near + rng.normal(0, 5 * p.sigma0_m, d.shape)
        noisy = np.where(drop, fill, noisy)
        smooth = np.where(drop, fill, smooth)

    # 6. confidence
    conf = np.full(d.shape, 2, np.uint8)
    near_edge = ndimage.binary_dilation(jump, iterations=1)
    conf[near_edge | (inc > p.grazing_deg - 15)] = 1
    conf[drop | jump] = 0
    return {"depth": noisy.astype(np.float32), "smooth_depth": smooth.astype(np.float32),
            "conf": conf, "clean": clean}
