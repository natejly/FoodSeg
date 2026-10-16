"""Colour grading of scanned-food textures at render time (numpy only: runs inside Blender's python).

Why: the calibrated MetaFood3D textures (synth.scan_colour) reproduce the colour of the capture photos, which are
bright, warm and over-saturated next to a Nutrition5k photo (broccoli L 75 / chroma 62 against 31 / 25), and the
Lab optimal-transport fit is gamut-clipped to neon lime / orange.  The atlases also carry
  * a constant fill colour for the unused area: cut faces / caps whose UVs land there render black or brown;
  * grey 'hole fill' islands (untextured polygons): flat grey cut faces on cucumbers, steak, fish.
`prepare` fixes all three on the texel array before the material is built:
    fill       unused-area colour -> bled edge colours, then the food's mean colour;
    grey fill  neutral texels of a coloured texture -> the food's own colour (luminance structure kept);
    grade      lightness / chroma pulled towards what the real photos of that food look like (Nutrition5k,
               synth.colour_audit.n5k_target; generic caps for foods without a reference), hue nudged towards it,
               extreme chroma soft-clipped.  Only reduces excess / lifts collapsed colour, never recolours a food.
Statistics are in CIE Lab on sRGB-encoded texels (the bytes of the image file)."""
from __future__ import annotations

import numpy as np

from synth.colour_audit import chroma, hue_deg, lab_to_srgb, n5k_target, srgb_to_lab

RENDER_L = 0.66        # median (rendered / texture) lightness and chroma, measured on hard_v13 (87 scan items)
RENDER_C = 0.58
STRENGTH_L = 0.7
STRENGTH_C = 0.85
GENERIC_L, GENERIC_C = 66.0, 50.0          # caps on a texture's median lightness / chroma without a reference food
FILL_SHARE = 0.02      # an unused-area colour covers at least this much of the atlas
FILL_TOL = 10          # (0..255) per channel
GREY_C = 8.0           # neutral texel: Lab chroma below this ...
GREY_L = (22.0, 88.0)  # ... and lightness inside this range (not speculars / crevices)
GREY_REF_C = 18.0      # grey fill only in textures whose coloured texels have at least this median chroma
GREY_MAX_SHARE = 0.6


def wrap_deg(d: float) -> float:
    """Signed angle in (-180, 180]."""
    return (float(d) + 180.0) % 360.0 - 180.0


def detect_fill(rgb: np.ndarray) -> np.ndarray | None:
    """Mask (H, W) of the atlas' unused-area colour, or None.  rgb: uint8 (H, W, 3).  The colour is the most
    common 5-bit bin when it covers >= FILL_SHARE and forms solid areas (not scattered texels of a flat food)."""
    q = (rgb >> 3).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    cnt = np.bincount(key.ravel(), minlength=32768)
    top = int(cnt.argmax())
    if cnt[top] < FILL_SHARE * key.size:
        return None
    sel = key == top
    col = rgb[sel].mean(0)
    m = (np.abs(rgb.astype(np.float32) - col).max(-1) <= FILL_TOL)
    if m.mean() > 0.85:                       # a flat texture: nothing to separate
        return None
    inner = m.copy()
    inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    if inner.sum() < 0.8 * m.sum():           # speckle, not an area
        return None
    return m


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Binary dilation by n px (4-neighbourhood, no wrap): swallows the JPEG fringe around the islands."""
    m = mask.copy()
    for _ in range(n):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def fill_background(rgb: np.ndarray, mask: np.ndarray, grow: int = 24) -> np.ndarray:
    """Replace the masked texels: edge colours bled outwards over `grow` px (seams / bilinear taps stay clean),
    the remainder gets the median colour of the unmasked texels.  Returns uint8 (H, W, 3)."""
    out = rgb.copy()
    fg = ~mask
    if fg.sum() == 0:
        return out
    med = np.median(rgb[fg][::7], axis=0)
    known = fg.copy()
    cur = out.astype(np.float32)
    for _ in range(grow):
        acc = np.zeros_like(cur)
        n = np.zeros(mask.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = np.roll(known, (dy, dx), (0, 1))
            c = np.roll(cur, (dy, dx), (0, 1))
            acc += c * k[..., None]
            n += k
        new = (~known) & (n > 0)
        if not new.any():
            break
        cur[new] = acc[new] / n[new][:, None]
        known = known | new
    cur[~known] = med
    # the bled band fades into the median so far-away UVs read as the food's mean colour
    return np.clip(cur + 0.5, 0, 255).astype(np.uint8)


def texture_stats(lab: np.ndarray) -> dict:
    """Median lightness / chroma and chroma-weighted circular-mean hue of (N, 3) Lab texels."""
    c = chroma(lab)
    w = c + 1e-3
    h = np.arctan2((lab[:, 2] * 1.0), lab[:, 1])
    hm = np.degrees(np.arctan2((np.sin(h) * w).sum(), (np.cos(h) * w).sum())) % 360.0
    return dict(L50=float(np.median(lab[:, 0])), C50=float(np.median(c)), C90=float(np.percentile(c, 90)),
                hue=float(hm))


def grade_params(stats: dict, target: dict | None) -> dict:
    """Target statistics of the graded texture (see module doc) from the texture's `stats`.
    target: synth.colour_audit.n5k_target(...) dict (L, C_p50, C_p95, hue) of the real food, or None."""
    L50, C50 = stats["L50"], stats["C50"]
    if target is not None:
        Ls, Cs = target["L"] / RENDER_L, target["C_p50"] / RENDER_C
        Cmax = max(1.3 * Cs, 35.0)
    else:
        Ls, Cs = GENERIC_L, GENERIC_C
        Cmax = 65.0
    L_to = L50 + STRENGTH_L * (Ls - L50) if L50 > Ls else (L50 + 0.5 * (Ls - L50) if L50 < 0.6 * Ls else L50)
    if C50 > Cs:
        C_to = C50 + STRENGTH_C * (Cs - C50)
    elif target is not None and C50 < 0.5 * Cs:
        C_to = 0.75 * Cs                                  # collapsed colour (grey fish / steak): lift
    else:
        C_to = C50
    dh = 0.0
    if target is not None and C_to > 12.0 and C50 > 8.0:
        d = wrap_deg(target["hue"] - stats["hue"])
        if abs(d) <= 60.0:
            dh = 0.6 * d
    return dict(L_to=float(L_to), C_to=float(C_to), dh=float(dh), Cmax=float(Cmax),
                lift_hue=None if target is None or C50 >= 0.5 * Cs else float(target["hue"]))


def fill_grey(lab: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Neutral texels of a coloured texture take the food's colour (luminance structure kept, Lab arrays (N, 3)).
    Returns (lab, share replaced).  valid: bool (N,) texels that belong to the food."""
    c = chroma(lab)
    L = lab[:, 0]
    col = valid & (c >= GREY_C * 1.8) & (L > GREY_L[0]) & (L < 95)
    if col.sum() < 500 or np.median(c[col]) < GREY_REF_C:
        return lab, 0.0
    grey = valid & (c < GREY_C) & (L > GREY_L[0]) & (L < GREY_L[1])
    share = float(grey.sum() / max(valid.sum(), 1))
    if grey.sum() < 200 or share > GREY_MAX_SHARE:
        return lab, share
    cl = lab[col]
    ref = np.median(cl[cl[:, 0] >= np.median(cl[:, 0])], axis=0)        # the lighter half: cut flesh, not skin
    out = lab.copy()
    w = np.clip(1.0 - c[grey] / GREY_C, 0, 1)[:, None]                 # fully neutral texels: full replacement
    Lg = L[grey]
    new_L = ref[0] + 0.5 * (Lg - np.median(Lg))
    out[grey, 0] = (1 - w[:, 0]) * Lg + w[:, 0] * new_L
    out[grey, 1:] = (1 - w) * lab[grey, 1:] + w * ref[1:]
    return out, share


def grade_lab(lab: np.ndarray, p: dict, stats: dict) -> np.ndarray:
    """Apply `grade_params` output to (N, 3) Lab texels."""
    out = lab.copy()
    L50, C50 = stats["L50"], max(stats["C50"], 1e-3)
    out[:, 0] = np.clip(p["L_to"] + 0.9 * (lab[:, 0] - L50), 0, 100)
    lift = p["lift_hue"] is not None
    c = chroma(lab)
    h = np.arctan2(lab[:, 2], lab[:, 1])
    if lift:                                  # neutral texels take the reference hue
        w = np.clip(1.0 - c / 14.0, 0, 1)
        ht = np.radians(p["lift_hue"])
        h = np.arctan2((1 - w) * np.sin(h) + w * np.sin(ht), (1 - w) * np.cos(h) + w * np.cos(ht))
        g = p["C_to"] / C50
        c = np.maximum(c * np.clip(g, 0.25, 4.0), w * p["C_to"] * 0.8)
    else:
        c = c * np.clip(p["C_to"] / C50, 0.25, 4.0)
    h = h + np.radians(p["dh"])
    cm = p["Cmax"]
    c = cm * np.tanh(c / cm)                  # soft clip: neon extremes
    out[:, 1], out[:, 2] = c * np.cos(h), c * np.sin(h)
    return out


def prepare(rgb: np.ndarray, name: str, grade: bool = True, chunk: int = 1 << 20) -> tuple[np.ndarray, dict]:
    """uint8 (H, W, 3) texture -> repaired + graded uint8 (H, W, 3) and an info dict for the log."""
    info: dict = {"name": name}
    h, w = rgb.shape[:2]
    mask = detect_fill(rgb)
    work = rgb
    if mask is not None:
        work = fill_background(rgb, dilate(mask, 3))
        info["fill_share"] = round(float(mask.mean()), 3)
    valid = np.ones((h, w), bool) if mask is None else ~dilate(mask, 3)
    flat = work.reshape(-1, 3)
    vflat = valid.reshape(-1)
    lab = np.empty(flat.shape, np.float32)
    for i in range(0, len(flat), chunk):
        lab[i:i + chunk] = srgb_to_lab(flat[i:i + chunk])
    lab, gshare = fill_grey(lab, vflat)
    info["grey_share"] = round(gshare, 3)
    target = n5k_target(name) if grade else None
    info["target"] = target["n5k"] if target else None
    stats = texture_stats(lab[vflat][::5])
    info["stats"] = {k: round(v, 1) for k, v in stats.items()}
    if grade:
        p = grade_params(stats, target)
        info["params"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()}
        lab = grade_lab(lab, p, stats)
    out = np.empty(flat.shape, np.uint8)
    for i in range(0, len(flat), chunk):
        out[i:i + chunk] = np.clip(lab_to_srgb(lab[i:i + chunk]) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out.reshape(h, w, 3), info
