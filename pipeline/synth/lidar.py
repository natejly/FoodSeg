"""Turn a clean rendered depth map into something that looks like ARKit sceneDepth.

ARKit gives 256x192 float32 depth plus a 0/1/2 confidence map. Pixels it isn't
sure about still hold a value (often smeared from neighbours), they are just
marked low confidence. This model reproduces that, step by step (see
docs/synthetic_data.md, "Faking iPad LiDAR depth"):

  1. area-downsample the render to 256x192 (mixed pixels at edges, like dToF);
  2. sub-pixel lateral jitter;
  3. range-dependent axial noise, white + a smooth low-frequency bias field;
  4. edge bleeding toward the RGB edge (Apple fuses RGB into the depth);
  5. dropout on dark and grazing-angle surfaces -> filled from neighbours, conf 0;
  6. confidence 2/1/0 from depth discontinuities and incidence angle.

Every number lives in LidarParams. The defaults are PLACEHOLDERS from the
literature (Kim 2023: ~0.5 mm sigma, <=3 mm bias at 30-40 cm; Luetzenburg 2021);
refit them from the iPad calibration scans (docs/collection_protocol.md, cal_*)
and record the fitted values in gt.json via `asdict(params)`.
"""
from __future__ import annotations

from dataclasses import dataclass

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
             p: LidarParams = LidarParams()) -> dict[str, np.ndarray]:
    """depth_hr: render-res z-depth (m, 0 = none); rgb: render-res uint8; K_hr: render-res K.

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

    # 3. axial noise
    sig = p.sigma0_m + p.sigma_z2 * (d - p.z0_m) ** 2
    white = rng.normal(0, 1, d.shape) * sig
    bias = ndimage.gaussian_filter(rng.normal(0, 1, d.shape), p.bias_scale_px)
    bias *= p.bias_sigma_m / (bias.std() + 1e-12)
    noisy = d + white + bias
    smooth = d + p.smooth_factor * white + bias

    # 5. dropout
    lum = _area_resize(rgb.astype(np.float32).mean(-1) / 255.0, D_W, D_H)
    inc = incidence_deg(clean, K)
    drop = ((lum < p.dark_lum) & (rng.random(d.shape) < p.p_drop_dark)) | \
           ((inc > p.grazing_deg) & (rng.random(d.shape) < p.p_drop_grazing)) | \
           (clean < p.min_range_m) | ~valid
    if drop.any() and (~drop).any():
        # ARKit still reports a value: nearest valid neighbour plus extra noise
        idx = ndimage.distance_transform_edt(drop, return_distances=False, return_indices=True)
        fill = noisy[tuple(idx)] + rng.normal(0, 5 * p.sigma0_m, d.shape)
        noisy = np.where(drop, fill, noisy)
        smooth = np.where(drop, fill, smooth)

    # 6. confidence
    conf = np.full(d.shape, 2, np.uint8)
    near_edge = ndimage.binary_dilation(jump, iterations=1)
    conf[near_edge | (inc > p.grazing_deg - 15)] = 1
    conf[drop | jump] = 0
    return {"depth": noisy.astype(np.float32), "smooth_depth": smooth.astype(np.float32),
            "conf": conf, "clean": clean}
