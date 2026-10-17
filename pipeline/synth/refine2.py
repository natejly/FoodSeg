"""Render-then-refine v2: food-masked, per-food-prompted, multi-view-consistent diffusion refinement.

    python -m synth.refine2 --src ../data/synth/hard_v11 --dst ../data/synth/hard_v11_ref2 \
        --strength 0.4 --prop-strength 0.25 [--scans s000000] [--frames 0 1 2]

Why: a per-content realism probe says the fakest pixels are the procedural soft foods (curries,
porridge, rice, purees) and tables; scanned foods and dishes are less fake. refine.py applies one low
strength (0.15-0.2) everywhere; here the food gets a much stronger pass and everything else is left alone.

Per frame (work resolution 768 px, SD1.5 Realistic Vision 5.1 + ControlNet tile + depth, as refine.py):
  1. Food classes = items grouped by name (seg ids 1..252 -> spec items). Each class gets its own pass
     with its own prompt ("close-up phone photo of <name> in a <dish> ...", negative prompt against
     painterly / CG looks). At most `--max-passes` classes (largest first, >= `--min-area` of the image).
  2. Masked SDEdit: the class mask (each *instance* eroded by `--erode` px so label boundaries stay put,
     then feathered) acts as a per-pixel strength. The whole latent is noised to `strength` and denoised,
     but at every step the latent OUTSIDE the mask is reset to the (re-noised) original latent
     (blended-latent-diffusion trick), so the outside is exactly the original and the inside is
     refined at strength 0.3-0.5 with full context.
  3. The change is a delta (refined - original), its low-frequency component inside each mask is removed
     (masked-normalised Gaussian, `--lf-sigma` px @ full-res: keeps the render's per-region colour and
     exposure, view-consistent), weighted by the feathered mask, upsampled and added to the full-res
     frame; phone grain is topped up under the mask only; JPEG.
  4. Multi-view consistency (--no-prop disables): frames are processed in order; for each new frame the
     deltas of the already refined frames are reprojected into it with the rendered depth and the TRUE
     camera poses (spec.json camera c2w via world_to_arkit; frames.jsonl is the fallback), keeping only
     pixels that are visible in the source (depth agreement) and carry the same seg label. The warped
     delta is added to the init image, so the next view starts from the same food texture and only
     needs a lower strength (`--prop-strength`) to harmonise it and fill what was not visible before.
     Nearest source camera wins per pixel (no averaging, which would wash detail out).
Outputs use the same root layout as refine.py (rgb/ new, everything else symlinked).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import refine as R0
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from synth import refine as R0

NEG = ("cartoon, 3d render, cgi, illustration, painting, drawing, digital art, concept art, plastic, waxy, "
       "airbrushed, glossy plastic, smooth, flawless, perfect, blurry, oversaturated, text, watermark, "
       "extra objects, deformed, bowl, plate, table")
SEG_FOOD_MAX = 252


# ---------------------------------------------------------------- scene bookkeeping
def class_prompt(name: str, spec: dict) -> str:
    n = name.replace("_", " ")
    dish = (spec.get("container") or {}).get("kind") or "plate"
    return (f"close-up phone photo of {n} on a {dish}, real home-cooked food, natural uneven texture, "
            f"moist, subtle highlights, natural light, sharp focus, photograph, 35mm, high detail")


def poses_of(src: Path, sid: str, spec: dict):
    """camera-to-world in ARKit world (true poses from the spec; the exported frames.jsonl may carry drift)."""
    cam = spec.get("camera") or {}
    if cam.get("c2w") and spec.get("world_to_arkit"):
        W = np.array(spec["world_to_arkit"], np.float64)
        return [W @ np.array(c, np.float64) for c in cam["c2w"]], "spec"
    fr = [json.loads(l) for l in open(src / "scans" / sid / "frames.jsonl")]
    return [np.array(f["transform"], np.float64) for f in fr], "frames.jsonl"


def intrinsics_of(src: Path, sid: str, spec: dict):
    K = (spec.get("camera") or {}).get("K")
    if K is None:
        K = json.loads(open(src / "scans" / sid / "frames.jsonl").readline())["intrinsics"]
    return np.array(K, np.float64)


def food_classes(seg_low: np.ndarray, spec: dict, erode_px: float, min_area: float, max_passes: int):
    """-> [(name, hard mask bool (eroded per instance))] largest first."""
    import cv2
    by_id = {int(it.get("id", k + 1)): str(it.get("name", "food")) for k, it in enumerate(spec["items"])}
    k = max(int(round(erode_px * 2)) | 1, 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cls = {}
    for l in np.unique(seg_low):
        if not (1 <= l <= SEG_FOOD_MAX):
            continue
        m = (seg_low == l).astype(np.uint8)
        if m.sum() < 30:
            continue
        # erode away from every boundary incl. the image border
        me = cv2.erode(np.pad(m, 1, constant_values=0), ker)[1:-1, 1:-1] > 0
        cls.setdefault(by_id.get(int(l), "food"), np.zeros(m.shape, bool))
        cls[by_id.get(int(l), "food")] |= me
    out = [(n, m) for n, m in cls.items() if m.mean() >= min_area]
    out.sort(key=lambda t: -t[1].sum())
    return out[:max_passes]


def feather(m: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma), 0, 1) if sigma > 0 else m.astype(np.float32)


# ---------------------------------------------------------------- masked SDEdit
def _encode(pipe, arr_u8, device="cuda"):
    import torch
    x = torch.from_numpy(arr_u8.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    v = pipe.vae
    old = next(v.parameters()).device
    v.to(device)
    try:
        with torch.no_grad():
            z = v.encode(x.to(device, v.dtype)).latent_dist.mean * v.config.scaling_factor
    finally:
        v.to(old)
    return z


def masked_img2img(pipe, init: np.ndarray, ctl_tile: np.ndarray, ctl_depth: np.ndarray, w: np.ndarray,
                   prompt: str, seed: int, strength: float, steps: int = 30, cfg: float = 5.0,
                   tile_scale: float = 0.5, depth_scale: float = 0.5) -> np.ndarray:
    """SDEdit at `strength` inside w>0, original latent kept outside (blended per step). Returns uint8."""
    import torch
    import torch.nn.functional as F
    h, wd = init.shape[:2]
    z0 = _encode(pipe, init).to("cuda", torch.float16)
    g = torch.Generator("cpu").manual_seed(seed + 7)
    eps = torch.randn(z0.shape, generator=g).to("cuda", z0.dtype)
    wl = torch.from_numpy(w)[None, None].float()
    wl = F.interpolate(wl, size=z0.shape[-2:], mode="area").to("cuda", z0.dtype)
    wl = torch.clamp(wl * 1.0, 0, 1)

    def cb(p, i, t, kw):
        ts = p.scheduler.timesteps
        lat = kw["latents"]
        bg = z0 if i + 1 >= len(ts) else p.scheduler.add_noise(z0, eps, ts[i + 1:i + 2])
        kw["latents"] = wl * lat + (1 - wl) * bg
        return kw

    gen = torch.Generator("cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=NEG, image=Image.fromarray(init),
               control_image=[Image.fromarray(ctl_tile), Image.fromarray(ctl_depth)], strength=strength,
               num_inference_steps=steps, guidance_scale=cfg, generator=gen,
               controlnet_conditioning_scale=[tile_scale, depth_scale], width=wd, height=h,
               callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    return np.asarray(out)


# ---------------------------------------------------------------- multi-view propagation
def warp_deltas(cam_j, srcs, dj, sj, K_low):
    """Delta of already refined frames -> frame j. cam_j: c2w; srcs: [dict(c2w, depth, seg, delta)].
    Returns (delta (h,w,3) float32, valid bool)."""
    import cv2
    h, w = dj.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    ok0 = np.isfinite(dj) & (dj > 0.05) & (sj > 0) & (sj <= SEG_FOOD_MAX)
    z = np.where(ok0, dj, 1.0)
    x = (u - K_low[0, 2]) / K_low[0, 0] * z
    y = (v - K_low[1, 2]) / K_low[1, 1] * z
    pw = np.stack([x, -y, -z, np.ones_like(z)], -1).reshape(-1, 4) @ cam_j.T
    out = np.zeros((h, w, 3), np.float32)
    done = np.zeros((h, w), bool)
    order = sorted(srcs, key=lambda s: np.linalg.norm(s["c2w"][:3, 3] - cam_j[:3, 3]))
    for s in order:
        pj = pw @ np.linalg.inv(s["c2w"]).T
        zi = -pj[:, 2]
        ui = (pj[:, 0] / np.maximum(zi, 1e-6) * K_low[0, 0] + K_low[0, 2]).reshape(h, w).astype(np.float32)
        vi = (-pj[:, 1] / np.maximum(zi, 1e-6) * K_low[1, 1] + K_low[1, 2]).reshape(h, w).astype(np.float32)
        zi = zi.reshape(h, w)
        inb = (ui >= 1) & (ui < w - 2) & (vi >= 1) & (vi < h - 2) & (zi > 0.05)
        zs = cv2.remap(s["depth"], ui, vi, cv2.INTER_NEAREST)
        ss = cv2.remap(s["seg"], ui, vi, cv2.INTER_NEAREST)
        m = ok0 & inb & ~done & (np.abs(zs - zi) < 0.012 * zi) & (ss == sj)
        if not m.any():
            continue
        samp = cv2.remap(s["delta"], ui, vi, cv2.INTER_LINEAR)
        out[m] = samp[m]
        done |= m
    return out, done


# ---------------------------------------------------------------- frame
def lf_remove(delta: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Remove from `delta` its low-frequency part inside the weight mask (masked-normalised Gaussian)."""
    import cv2
    if sigma <= 0:
        return delta
    num = cv2.GaussianBlur(delta * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    return delta - num / np.maximum(den, 1e-3)


def refine_frame2(pipe, orig, depth, seg, spec, seed, srcs, cam, K_full, a, ww):
    """-> (full-res uint8 result, dict(delta=low-res delta, depth_low, seg_low) for propagation, info)."""
    import cv2
    H, W = orig.shape[:2]
    wh = int(round(H * ww / W / 8)) * 8
    low = cv2.resize(orig, (ww, wh), interpolation=cv2.INTER_AREA)
    seg_l = cv2.resize(seg, (ww, wh), interpolation=cv2.INTER_NEAREST)
    dep_l = cv2.resize(depth, (ww, wh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    ctl_d = cv2.resize(R0.depth_control(depth, seg), (ww, wh), interpolation=cv2.INTER_AREA)
    K_low = K_full.copy()
    K_low[:2] *= ww / W
    classes = food_classes(seg_l, spec, a.erode, a.min_area, a.max_passes)
    info = dict(classes=[(n, float(m.mean())) for n, m in classes], propagated=0.0)
    lowf = low.astype(np.float32)
    P = np.zeros_like(lowf)
    if srcs and not a.no_prop and classes:
        P, valid = warp_deltas(cam, srcs, dep_l, seg_l, K_low)
        union = np.zeros(seg_l.shape, bool)
        for _, m in classes:
            union |= m
        valid &= union
        P[~valid] = 0
        info["propagated"] = float(valid.sum() / max(union.sum(), 1))
        strength = a.prop_strength
    else:
        valid = np.zeros(seg_l.shape, bool)
        strength = a.strength
    init = np.clip(lowf + P + 0.5, 0, 255).astype(np.uint8) if valid.any() else low
    D = np.zeros_like(lowf)
    Wsum = np.zeros(seg_l.shape, np.float32)
    for name, m in classes:
        wgt = feather(m, a.feather)   # eroded mask, softened: ~0 at the true label boundary
        wgt *= np.clip(1.0 - Wsum, 0, 1)
        ref = masked_img2img(pipe, init, low, ctl_d, wgt, class_prompt(name, spec), seed, strength, a.steps, a.cfg,
                             a.tile_scale, a.depth_scale)
        d = ref.astype(np.float32) - lowf        # includes the propagated part (init = low + P)
        d = lf_remove(d, wgt, a.lf_sigma * ww / W)
        D += wgt[..., None] * d
        Wsum += wgt
    if classes:
        D += P * np.clip(1.0 - Wsum, 0, 1)[..., None]
    # full-res
    Dup = cv2.resize(D, (W, H), interpolation=cv2.INTER_CUBIC)
    Wup = np.clip(cv2.resize(np.minimum(Wsum, 1.0), (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)
    res = orig.astype(np.float32) + Dup
    if a.grain and classes:
        tgt = R0._hf_noise(orig)
        cur = R0._hf_noise(np.clip(res, 0, 255).astype(np.uint8))
        add = np.sqrt(max(tgt ** 2 - cur ** 2, 0))
        if add > 0:
            rng = np.random.default_rng(seed)
            nz = rng.normal(0, add, (H, W, 1)).astype(np.float32) + rng.normal(0, add * 0.3, (H, W, 3)).astype(np.float32)
            res = res + nz * Wup[..., None]
    out = np.clip(res + 0.5, 0, 255).astype(np.uint8)
    keep = dict(depth=dep_l, seg=seg_l, delta=D.astype(np.float32))
    return out, keep, info


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--strength", type=float, default=0.4, help="food strength for a view with no refined neighbours")
    ap.add_argument("--prop-strength", type=float, default=0.25, help="strength when the init carries propagated detail")
    ap.add_argument("--scans", nargs="*")
    ap.add_argument("--frames", nargs="*", type=int)
    ap.add_argument("--work-w", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--tile-scale", type=float, default=0.5)
    ap.add_argument("--depth-scale", type=float, default=0.5)
    ap.add_argument("--lf-sigma", type=float, default=40.0, help="remove low-freq colour drift inside food (px @full-res)")
    ap.add_argument("--erode", type=float, default=3.0, help="erode each food instance (work px)")
    ap.add_argument("--feather", type=float, default=2.0, help="mask feather sigma (work px)")
    ap.add_argument("--min-area", type=float, default=0.01)
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="skip scans whose frames are all written (a partly done scan is redone: propagation needs its deltas)")
    ap.add_argument("--no-prop", action="store_true")
    ap.add_argument("--no-grain", dest="grain", action="store_false")
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--jpeg-q", type=int, default=90)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    sids = a.scans or sorted(p.name for p in (src / "scans").iterdir())
    R0.prepare_root(src, dst, sids)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pipe = R0.load_pipe(not a.no_offload)
    pipe.enable_attention_slicing("max")   # 6 GB card shared with Blender
    ww = a.work_w - a.work_w % 8
    t0, n = time.time(), 0
    for sid in sids:
        spec = json.load(open(src / "specs" / sid / "spec.json"))
        seed = R0.seed_of(sid)
        poses, psrc = poses_of(src, sid, spec)
        K = intrinsics_of(src, sid, spec)
        if a.resume and all((dst / "scans" / sid / "rgb" / j.name).exists() and (dst / "scans" / sid / "rgb" / j.name).stat().st_size > 0
                            for j in (src / "scans" / sid / "rgb").glob("*.jpg")):
            continue
        srcs = []
        for j in sorted((src / "scans" / sid / "rgb").glob("*.jpg")):
            k = int(j.stem)
            if a.frames and k not in a.frames:
                continue
            outp = dst / "scans" / sid / "rgb" / j.name
            orig = np.asarray(Image.open(j).convert("RGB"))
            raw = np.load(src / "raw" / sid / f"frame_{k:06d}.npz")
            for attempt in range(60):
                try:
                    res, keep, info = refine_frame2(pipe, orig, raw["depth"], raw["seg"], spec, seed, srcs, poses[k], K, a, ww)
                    break
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" not in str(e).lower() and "cusolver" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                        raise
                    torch.cuda.empty_cache()
                    time.sleep(20)
            else:
                raise RuntimeError("OOM retries exhausted")
            keep["c2w"] = poses[k]
            srcs.append(keep)
            Image.fromarray(res).save(outp, quality=a.jpeg_q)
            n += 1
            print(f"{sid} {k} {info}  poses={psrc}  {(time.time() - t0) / n:.1f}s/frame", flush=True)


if __name__ == "__main__":
    main()
