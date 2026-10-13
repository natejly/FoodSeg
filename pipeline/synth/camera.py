"""Phone-camera degradations applied to clean renders (hard preset), so RGB looks captured, not rendered.

Per scan (one capture = one camera state): white-balance error, sensor noise level,
JPEG quality. Per frame (hand-held orbit): motion blur along a random direction and
slight defocus. Parameters are drawn once per scan and recorded in gt.json; the clean
render is what the LiDAR model sees, so depth is unaffected.
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
    """Scene-linear render -> 8-bit sRGB the way a phone ISP roughly does: auto-exposure maps
    the scan's log-average luminance to 0.18 (plus a bias in EV), an extended-Reinhard curve on
    luminance rolls off highlights without bleaching colour, a mild saturation boost, sRGB."""
    x = linear.astype(np.float32) * (0.18 / max(key, 1e-6)) * 2.0 ** ev_bias
    L = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    Lo = L * (1 + L / white ** 2) / (1 + L)
    x = x * (Lo / np.maximum(L, 1e-6))[..., None]
    g = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = g + saturation * (x - g)
    return (srgb_oetf(x) * 255 + 0.5).astype(np.uint8)


@dataclass
class PhoneCamera:
    wb_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)   # residual white-balance error, R G B
    noise_sigma: float = 1.5                               # 8-bit units at mid-grey (shot + read noise)
    blur_px_max: float = 2.0                               # max motion-blur length at 1920 px width
    defocus_p: float = 0.15                                # share of frames slightly out of focus
    jpeg_quality: int = 90
    ev_bias: float = 0.0                                   # auto-exposure error, EV (HDR renders only)

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "PhoneCamera":
        return cls(wb_gain=tuple(float(g) for g in np.r_[rng.normal(1, 0.04), 1.0, rng.normal(1, 0.04)]),
                   noise_sigma=float(rng.uniform(0.5, 3.0)), blur_px_max=float(rng.uniform(0.5, 4.0)),
                   defocus_p=float(rng.uniform(0.0, 0.3)), jpeg_quality=int(rng.integers(80, 96)),
                   ev_bias=float(rng.normal(0, 0.35)))

    def to_json(self) -> dict:
        return asdict(self)

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        img = rgb.astype(np.float32)
        s = img.shape[1] / 1920.0                          # effects are specified at full resolution
        L = rng.uniform(0, self.blur_px_max) * s
        if L >= 1.0:                                       # linear motion blur kernel
            k = int(np.ceil(L)) | 1
            ker = np.zeros((k, k), np.float32)
            a = rng.uniform(0, np.pi)
            c = k // 2
            for t in np.linspace(-L / 2, L / 2, 4 * k):
                ker[int(round(c + t * np.sin(a))), int(round(c + t * np.cos(a)))] = 1
            img = ndimage.convolve(img, (ker / ker.sum())[..., None], mode="nearest")
        if rng.random() < self.defocus_p:
            sig = rng.uniform(0.6, 1.5) * max(s, 0.5)
            img = ndimage.gaussian_filter(img, (sig, sig, 0))
        img *= np.asarray(self.wb_gain, np.float32)
        # signal-dependent noise: variance grows with intensity (shot) plus a floor (read)
        sd = self.noise_sigma * np.sqrt(0.3 + img / 128.0)
        img += rng.normal(0, 1, img.shape).astype(np.float32) * sd
        return np.clip(img, 0, 255).astype(np.uint8)
