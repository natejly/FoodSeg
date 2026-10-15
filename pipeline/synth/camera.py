"""Phone-camera (iPad / iPhone back camera) image pipeline applied to scene-linear renders (hard preset),
so the RGB looks captured, not rendered. Depth / seg are never touched; LiDAR sees the clean render.

`PhoneCamera.process(linear, key, rng, t)` is the whole sensor + ISP chain, in the order a phone does it:

  optics (linear)   lens softness, hand-shake motion blur, occasional defocus, lens shading, lateral CA
  auto-exposure     scan metering + AE target + slow handheld drift + partial re-metering on frame content
  sensor            shot + read noise (variance ~ ISO * signal) in linear, spatially correlated, with
                    low-frequency chroma noise (demosaic / binning)
  white balance     illuminant-estimation error + partial grey-world AWB on the frame (fooled by big
                    coloured surfaces, like the real thing), slow drift between frames
  tone mapping      local (guided-filter base/detail: shadow lift, detail boost = "smart HDR" look)
                    then a filmic global curve with toe / shoulder; hue-preserving, highlights roll to white
  colour            small colour-matrix error, vibrance + saturation
  NR + sharpening   luma denoise that smears low-contrast texture, strong chroma NR, unsharp mask (halos)
  JPEG              quality sampled per scan, 4:2:0 chroma subsampling (see `save_jpeg`)

All parameters are drawn once per scan (`PhoneCamera.sample`), recorded in gt.json (`to_json`), and
frame-to-frame variation is a smooth function of the frame index so a hand-held orbit drifts
plausibly. `develop` + `apply` are the older cheap path (kept for non-HDR frames / lidar guidance / tests).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def scene_key(frames: list[np.ndarray]) -> float:
    """Log-average luminance over a scan's linear frames (what an auto-exposure meters)."""
    lum = [0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2] for f in frames]
    return float(np.exp(np.mean([np.log(l.astype(np.float32) + 1e-4).mean() for l in lum])))


def develop(linear: np.ndarray, key: float, ev_bias: float = 0.0, white: float = 4.0,
            saturation: float = 1.1) -> np.ndarray:
    """Cheap clean development (no camera defects): auto-exposure maps the scan's log-average luminance
    to 0.18 (plus a bias in EV), an extended-Reinhard curve on luminance, mild saturation, sRGB.
    Used where a noise-free RGB is wanted (LiDAR RGB guidance); `PhoneCamera.process` is the camera."""
    x = linear.astype(np.float32) * (0.18 / max(key, 1e-6)) * 2.0 ** ev_bias
    L = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    Lo = L * (1 + L / white ** 2) / (1 + L)
    x = x * (Lo / np.maximum(L, 1e-6))[..., None]
    g = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = g + saturation * (x - g)
    return (srgb_oetf(x) * 255 + 0.5).astype(np.uint8)


@dataclass
class PhoneCamera:
    # ---- shared with the legacy `apply` path
    wb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)   # illuminant-estimation error, R G B (scan)
    noise_sigma: float = 1.5                               # legacy `apply`: 8-bit units at mid-grey
    blur_px_max: float = 2.0                               # hand-shake blur length scale at 1920 px width
    defocus_p: float = 0.15                                # share of frames slightly out of focus
    jpeg_quality: int = 90
    ev_bias: float = 0.0                                   # auto-exposure error of the scan, EV
    vignette: float = 0.2                                  # lens shading: brightness lost in the corners
    ca_px: float = 0.5                                     # lateral chromatic aberration at the corner, px at 1920
    sharpen: float = 0.5                                   # ISP unsharp-mask amount (halos on edges)
    # ---- `process` chain
    ae_target: float = 1.0          # AE brightness target, x the default (scene key -> mid grey)
    ae_follow: float = 0.35         # how much AE re-meters on each frame's own content (0 = fixed exposure)
    ae_drift_ev: float = 0.12       # slow handheld exposure wander amplitude, EV
    ae_phase: tuple[float, float, float, float] = (0.0, 1.0, 2.0, 3.0)   # drift phases (sum of sines)
    iso: float = 200.0              # sensor gain; shot-noise variance and read noise scale with it
    lens_sigma: float = 0.6         # optical + demosaic softness, px at 1920
    awb_strength: float = 0.5       # share of the frame's grey-world cast the AWB removes
    wb_drift: float = 0.015         # frame-to-frame AWB wander (fraction)
    ccm_err: tuple[float, ...] = (0.0,) * 6   # off-diagonal colour-matrix error
    lt_comp: float = 0.75           # local tone mapping: base-layer compression (1 = none)
    lt_detail: float = 1.15         # detail-layer gain
    contrast: float = 1.0           # global filmic contrast multiplier
    saturation: float = 1.03
    vibrance: float = 0.08
    nr_luma: float = 0.5            # luma denoise strength (texture smear)
    nr_chroma: float = 1.8          # chroma denoise sigma, px at 1920
    sharpen_sigma: float = 1.0
    motion_p: float = 0.35          # share of frames with visible hand-shake blur

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "PhoneCamera":
        base = dict(noise_sigma=float(rng.uniform(0.5, 3.0)), blur_px_max=float(rng.uniform(0.5, 4.0)),
                    defocus_p=float(rng.uniform(0.0, 0.3)), jpeg_quality=int(rng.integers(84, 96)),
                    ev_bias=float(rng.normal(0, 0.35)), vignette=float(rng.uniform(0.08, 0.3)),
                    ca_px=float(rng.uniform(0.2, 1.2)), sharpen=float(rng.uniform(0.2, 0.8)))
        # illuminant error lives mostly on the blue-amber axis (what AWB gets wrong), a little green-magenta
        ba, gm = rng.normal(0, 0.055), rng.normal(0, 0.025)
        base["wb_gain"] = (float(np.exp(ba + gm)), 1.0, float(np.exp(-ba + gm)))
        return cls(**base,
                   ae_target=float(np.exp(rng.normal(0, 0.12))), ae_follow=float(rng.uniform(0.15, 0.55)),
                   ae_drift_ev=float(rng.uniform(0.04, 0.2)),
                   ae_phase=tuple(float(v) for v in rng.uniform(0, 6.28, 4)),
                   iso=float(np.exp(rng.uniform(np.log(64), np.log(1000)))),
                   lens_sigma=float(rng.uniform(0.45, 0.9)),
                   awb_strength=float(rng.uniform(0.25, 0.7)), wb_drift=float(rng.uniform(0.005, 0.03)),
                   ccm_err=tuple(float(v) for v in rng.normal(0, 0.025, 6)),
                   lt_comp=float(rng.uniform(0.62, 0.9)), lt_detail=float(rng.uniform(1.0, 1.35)),
                   contrast=float(rng.uniform(0.95, 1.2)), saturation=float(rng.uniform(0.95, 1.1)),
                   vibrance=float(rng.uniform(0.0, 0.15)), nr_luma=float(rng.uniform(0.25, 0.8)),
                   nr_chroma=float(rng.uniform(2.5, 4.0)), sharpen_sigma=float(rng.uniform(0.8, 1.3)),
                   motion_p=float(rng.uniform(0.15, 0.5)))

    @classmethod
    def from_json(cls, d: dict) -> "PhoneCamera":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("wb_gain", "ae_phase", "ccm_err"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_json(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ full sensor + ISP chain
    def process(self, linear: np.ndarray, key: float, rng: np.random.Generator, t: int = 0) -> np.ndarray:
        """Scene-linear render (H, W, 3) -> uint8 sRGB as a phone camera would deliver it.
        `key` is the scan's log-average luminance (scene_key), `t` the frame index."""
        x = np.maximum(linear.astype(np.float32), 0)
        h, w = x.shape[:2]
        s = w / 1920.0
        sd = max(s, 0.5)
        # ---- optics (scene-linear)
        L = self.blur_px_max * rng.random() ** 2 * s * (1.0 if rng.random() < self.motion_p else 0.3)
        if L >= 0.8:
            x = _motion_blur(x, L, rng.uniform(0, np.pi))
        sg = self.lens_sigma * s
        if rng.random() < self.defocus_p:
            sg = float(np.hypot(sg, rng.uniform(0.6, 1.5) * sd))
        if sg > 0.25:
            x = ndimage.gaussian_filter(x, (sg, sg, 0), mode="nearest")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2)
        x *= (1 - self.vignette * r2 ** 1.5)[..., None]
        if self.ca_px > 0:
            for ch, sgn in ((0, 1.0), (2, -1.0)):
                k = 1 + sgn * self.ca_px * s / np.hypot(cx, cy)
                x[..., ch] = ndimage.map_coordinates(x[..., ch], [cy + (yy - cy) / k, cx + (xx - cx) / k],
                                                     order=1, mode="nearest")
        del yy, xx, r2
        Y0 = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        # ---- auto-exposure: scan meter + target + drift + partial re-metering on this frame
        fkey = float(np.exp(np.log(Y0[::4, ::4] + 1e-4).mean()))
        ph = self.ae_phase
        drift = self.ae_drift_ev * (np.sin(0.9 * t + ph[0]) + 0.6 * np.sin(2.3 * t + ph[1])) / 1.6
        ev = self.ev_bias + drift - self.ae_follow * np.log2(max(fkey, 1e-6) / max(key, 1e-6))
        ev += rng.normal(0, 0.06)                          # per-frame metering jitter (not smooth)
        ev = float(np.clip(ev, -1.2, 1.2))
        x *= np.float32(self.ae_target * 0.18 / max(key, 1e-6) * 2.0 ** ev)
        # ---- sensor noise in linear, spatially correlated; chroma part is coarser (binning / demosaic)
        g = 0.6e-4 * self.iso / 100.0                      # shot-noise variance per unit signal
        r = 1.2e-3 * np.sqrt(self.iso / 100.0)             # read-noise sd
        n = rng.standard_normal(x.shape, dtype=np.float32)
        lum = n.mean(-1, keepdims=True)
        chrm = n - lum
        lum = ndimage.gaussian_filter(lum, (0.6 * sd, 0.6 * sd, 0), mode="wrap")
        lum /= lum.std() + 1e-9
        chrm = ndimage.gaussian_filter(chrm, (1.4 * sd, 1.4 * sd, 0), mode="wrap")
        chrm /= chrm.std() + 1e-9
        n = 0.8 * lum + 0.12 * chrm
        x = x + n * np.sqrt(g * x + r * r)
        # ---- white balance: scan illuminant error, grey-world AWB on this frame, drift
        mc = np.maximum(x.reshape(-1, 3)[::17].mean(0), 1e-4)
        gw = (mc.mean() / mc) ** self.awb_strength
        dr = self.wb_drift * (np.sin(0.7 * t + ph[2]) + 0.5 * np.sin(1.9 * t + ph[3])) + rng.normal(0, 0.006)
        gains = np.asarray(self.wb_gain, np.float32) * gw * np.array([1 + dr, 1, 1 - dr], np.float32)
        gains = gains / (0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2])
        x = x * gains.astype(np.float32)
        ce = self.ccm_err
        M = np.array([[1 - ce[0] - ce[1], ce[0], ce[1]], [ce[2], 1 - ce[2] - ce[3], ce[3]],
                      [ce[4], ce[5], 1 - ce[4] - ce[5]]], np.float32)
        x = np.maximum(x @ M.T, 0)
        # ---- local + global tone mapping
        x = _tone_map(x, self.lt_comp, self.lt_detail, self.contrast)
        # ---- colour rendering (display-referred, gamma space)
        y = srgb_oetf(x)
        lu = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2])[..., None]
        ch = y - lu
        csat = np.sqrt((ch ** 2).sum(-1, keepdims=True) / 0.25).clip(0, 1)       # 0 grey .. 1 vivid
        y = lu + ch * (self.saturation + self.vibrance * (1 - csat))
        # ---- ISP denoise + sharpen in YCbCr (8-bit scale)
        y = np.clip(y, 0, 1) * 255
        Y = 0.299 * y[..., 0] + 0.587 * y[..., 1] + 0.114 * y[..., 2]
        Cb = (y[..., 2] - Y) * 0.564
        Cr = (y[..., 0] - Y) * 0.713
        if self.nr_chroma > 0:
            Cb = ndimage.gaussian_filter(Cb, self.nr_chroma * sd, mode="nearest")
            Cr = ndimage.gaussian_filter(Cr, self.nr_chroma * sd, mode="nearest")
        Ys = ndimage.gaussian_filter(Y, 1.1 * sd, mode="nearest")
        gm = np.hypot(*np.gradient(ndimage.gaussian_filter(Y, 0.8 * sd, mode="nearest")))
        g0 = 2.5 + 4.0 * np.sqrt(self.iso / 100.0)          # NR protects edges stronger than the noise level
        Y = Y + (self.nr_luma / (1 + (gm / g0) ** 2)) * (Ys - Y)
        if self.sharpen > 0:
            soft = ndimage.gaussian_filter(Y, self.sharpen_sigma * sd, mode="nearest")
            d = Y - soft
            Y = Y + self.sharpen * 1.3 * d * (np.abs(d) > 0.8)   # small core: no amplified grain in flats
        R_ = Y + Cr / 0.713
        B_ = Y + Cb / 0.564
        G_ = (Y - 0.299 * R_ - 0.114 * B_) / 0.587
        out = np.stack([R_, G_, B_], -1)
        return (np.clip(out, 0, 255) + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------ legacy 8-bit path
    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = rgb.astype(np.float32)
        s = img.shape[1] / 1920.0                          # effects are specified at full resolution
        L = rng.uniform(0, self.blur_px_max) * s
        if L >= 1.0:
            img = _motion_blur(img, L, rng.uniform(0, np.pi))
        if rng.random() < self.defocus_p:
            sig = rng.uniform(0.6, 1.5) * max(s, 0.5)
            img = ndimage.gaussian_filter(img, (sig, sig, 0))
        img *= np.asarray(self.wb_gain, np.float32)
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2)   # 0 centre, 1 corner
        img *= (1 - self.vignette * r2 ** 1.5)[..., None]
        if self.ca_px > 0:                                 # red and blue magnified slightly differently
            for ch, sgn in ((0, 1.0), (2, -1.0)):
                k = 1 + sgn * self.ca_px * s / np.hypot(cx, cy)
                img[..., ch] = ndimage.map_coordinates(img[..., ch], [cy + (yy - cy) / k, cx + (xx - cx) / k],
                                                       order=1, mode="nearest")
        sdv = self.noise_sigma * np.sqrt(0.3 + img / 128.0)  # shot (grows with signal) + read floor
        img += rng.normal(0, 1, img.shape).astype(np.float32) * sdv
        if self.sharpen > 0:
            soft = ndimage.gaussian_filter(img, (1.0 * max(s, 0.5), 1.0 * max(s, 0.5), 0))
            img = img + self.sharpen * (img - soft)
        return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------- helpers
def _motion_blur(img: np.ndarray, L: float, ang: float) -> np.ndarray:
    k = int(np.ceil(L)) | 1
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    for t in np.linspace(-L / 2, L / 2, 4 * k):
        ker[int(round(c + t * np.sin(ang))), int(round(c + t * np.cos(ang)))] = 1
    return ndimage.convolve(img, (ker / ker.sum())[..., None], mode="nearest")


def _resize(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize(size, Image.BILINEAR))


def _box(a: np.ndarray, r: int) -> np.ndarray:
    return ndimage.uniform_filter(a, 2 * r + 1, mode="nearest")


def _guided(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of I (self-guided filter, He et al.)."""
    m = _box(I, r)
    v = _box(I * I, r) - m * m
    a = v / (v + eps)
    return _box(a, r) * I + _box(m - a * m, r)


def _filmic(x: np.ndarray) -> np.ndarray:
    """ACES-fit (Narkowicz) curve with a 0.8 pre-scale: toe, near-linear mid (0.18 -> ~0.2), shoulder."""
    x = x * 0.8
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def _tone_map(x: np.ndarray, comp: float, detail: float, contrast: float) -> np.ndarray:
    """Local base/detail compression in log2 luminance (guided filter on a 1/4-res copy, radius ~3 % of the
    width), then the filmic curve on luminance with hue-preserving ratios, blended towards the per-channel
    curve in the highlights so clipped colours roll to white instead of saturating."""
    h, w = x.shape[:2]
    Y = np.maximum(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2], 1e-5)
    lg = np.log2(Y)
    q = (max(w // 4, 8), max(h // 4, 8))
    base = _resize(_guided(_resize(lg, q), max(int(0.03 * q[0]), 2), 0.25), (w, h))
    pivot = np.log2(0.18)
    Yn = np.exp2(pivot + (base - pivot) * comp + (lg - base) * detail)
    if contrast != 1.0:
        Yn = 0.18 * (Yn / 0.18) ** contrast
    Yo = _filmic(Yn)
    a = x * (Yo / Y)[..., None]                            # hue-preserving
    b = _filmic(x * (Yn / Y)[..., None])                   # per-channel
    wgt = np.clip((Yo - 0.5) / 0.4, 0, 1)[..., None]
    return np.clip(a * (1 - wgt) + b * wgt, 0, 1)


def save_jpeg(arr: np.ndarray, path, quality: int) -> None:
    """JPEG as a phone writes it: 4:2:0 chroma subsampling."""
    from PIL import Image
    Image.fromarray(arr).save(path, quality=int(quality), subsampling=2)
