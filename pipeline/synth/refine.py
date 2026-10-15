"""Render-then-refine: a light structure-conditioned diffusion img2img pass over finished frames.

    python -m synth.refine --src ../data/synth/hard_v10 --dst ../data/synth/hard_v10_ref030 \
        --strength 0.30 [--scans s000000 s000001] [--frames 0 1 2] [--work-w 768]

What it does per frame (geometry and labels must stay put, appearance becomes photographic):
  1. take the final phone-style JPEG (scans/<sid>/rgb/NNNNNN.jpg) and downscale it to a working
     width (768 px; SD1.5 on a 6 GB card, and ~the scale the realism benchmark looks at);
  2. ControlNet-tile (the frame itself; pins layout + colour) and ControlNet-depth (inverse z-depth
     from raw/<sid>/frame_*.npz, background masked to far) condition an SD1.5-class photoreal
     checkpoint (Realistic Vision 5.1) at denoise `strength` (0.15-0.45), fixed seed per scan, prompt
     from the scene spec (food names + plate + "phone photo ...");
  3. the change the diffusion made is transferred back to the full-res frame as a *delta*:
         out = orig + up(refined_lowres - down(orig))            (`--delta`, default)
     so the original's phone noise / sharpening / full-res edges are kept and only the low/mid
     frequency appearance (food surface, colour, shading) is altered; then optional mild grain
     (matched to the original's high-frequency noise level) and JPEG at the original quality.
Outputs go to the new root with the same layout (rgb/ new; every other scan entry, raw/ and specs/
are symlinked) so synth.realism can be pointed at it unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"   # CreativeML OpenRAIL-M
CN_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"  # OpenRAIL
CN_TILE = "lllyasviel/control_v11f1e_sd15_tile"    # OpenRAIL
VAE = None  # the noVAE repo ships its own vae/ (sd-vae-ft-mse equivalent)

NEG = ("cartoon, 3d render, cgi, illustration, painting, plastic, waxy, smooth, blurry, "
       "oversaturated, text, watermark, extra objects, deformed")


def load_pipe(offload: bool = True):
    import torch
    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline,
                           UniPCMultistepScheduler)
    dt = torch.float16
    cn = [ControlNetModel.from_pretrained(CN_TILE, torch_dtype=dt),
          ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=dt, variant="fp16")]
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL, controlnet=cn, torch_dtype=dt, safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    pipe.enable_attention_slicing("max")   # fits next to a Blender job on a 6 GB card
    pipe.vae.enable_tiling()
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_prompt(spec: dict) -> str:
    names = []
    for it in spec.get("items", []):
        n = str(it.get("name", "")).replace("_", " ")
        if n and n not in names:
            names.append(n)
    foods = ", ".join(names[:5]) if names else "food"
    return (f"a phone photo of a plated meal: {foods}, on a plate on a table, natural light, "
            f"realistic food, appetizing, detailed natural food texture, sharp focus, photograph, 35mm")


def seed_of(sid: str) -> int:
    return int(hashlib.md5(sid.encode()).hexdigest()[:8], 16)


def depth_control(depth: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """ControlNet-depth wants MiDaS-style (near = bright). Inverse depth, robustly normalised."""
    d = depth.astype(np.float32).copy()
    ok = np.isfinite(d) & (d > 0.05)
    inv = np.zeros_like(d)
    inv[ok] = 1.0 / d[ok]
    if ok.sum() < 100:
        return np.zeros(d.shape + (3,), np.uint8)
    lo, hi = np.percentile(inv[ok], [2, 98])
    v = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    v = 0.15 + 0.85 * v
    v[~ok] = 0
    return np.repeat((v * 255).astype(np.uint8)[..., None], 3, 2)


def _hf_noise(a: np.ndarray, mask=None) -> float:
    import cv2
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float(np.median(np.abs(r)) * 1.4826)


def refine_frame(pipe, orig: np.ndarray, depth: np.ndarray, seg: np.ndarray, prompt: str, seed: int,
                 strength: float = 0.3, work_w: int = 768, steps: int = 30, cfg: float = 5.0,
                 tile_scale: float = 0.6, depth_scale: float = 0.5, delta: bool = True,
                 grain: bool = True, jpeg_q: int = 90) -> np.ndarray:
    import cv2
    import torch
    H, W = orig.shape[:2]
    ww = work_w - work_w % 8
    wh = int(round(H * ww / W / 8)) * 8
    low = cv2.resize(orig, (ww, wh), interpolation=cv2.INTER_AREA)
    dctl = cv2.resize(depth_control(depth, seg), (ww, wh), interpolation=cv2.INTER_AREA)
    gen = torch.Generator("cpu").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=NEG, image=Image.fromarray(low),
               control_image=[Image.fromarray(low), Image.fromarray(dctl)], strength=strength,
               num_inference_steps=steps, guidance_scale=cfg, generator=gen,
               controlnet_conditioning_scale=[tile_scale, depth_scale], width=ww, height=wh).images[0]
    ref = np.asarray(out).astype(np.float32)
    if delta:
        d = ref - low.astype(np.float32)
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
        res = orig.astype(np.float32) + d
    else:
        res = cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)
    if grain:  # diffusion output is cleaner than a phone frame: top the noise level back up
        target = _hf_noise(orig)
        cur = _hf_noise(np.clip(res, 0, 255).astype(np.uint8))
        add = np.sqrt(max(target ** 2 - cur ** 2, 0))
        if add > 0:
            rng = np.random.default_rng(seed)
            res = res + rng.normal(0, add, (H, W, 1)).astype(np.float32) * 1.0 + \
                rng.normal(0, add * 0.3, (H, W, 3)).astype(np.float32)
    return np.clip(res + 0.5, 0, 255).astype(np.uint8)


def anchor_lowfreq(orig: np.ndarray, ref: np.ndarray, sigma: float, alpha: float = 1.0) -> np.ndarray:
    """Keep the original's low-frequency colour / exposure (view-consistent: it comes from the render),
    take only the detail above `sigma` px (full-res) from the refinement:
        out = ref - alpha * G_sigma(ref - orig)
    alpha=1 removes all colour drift the diffusion added below the sigma scale; 0 is a no-op."""
    import cv2
    d = ref.astype(np.float32) - orig.astype(np.float32)
    return np.clip(ref.astype(np.float32) - alpha * cv2.GaussianBlur(d, (0, 0), sigma) + 0.5, 0, 255).astype(np.uint8)


def reanchor_root(src: Path, dst: Path, out: Path, sigma: float, alpha: float = 1.0):
    """Post-process an already refined root `dst` into `out` with anchor_lowfreq."""
    sids = sorted(p.name for p in (dst / "scans").iterdir())
    prepare_root(src, out, sids)
    for sid in sids:
        for j in sorted((dst / "scans" / sid / "rgb").glob("*.jpg")):
            o = np.asarray(Image.open(src / "scans" / sid / "rgb" / j.name).convert("RGB"))
            r = np.asarray(Image.open(j).convert("RGB"))
            Image.fromarray(anchor_lowfreq(o, r, sigma, alpha)).save(out / "scans" / sid / "rgb" / j.name, quality=90)


def prepare_root(src: Path, dst: Path, sids):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "specs"):
        if not os.path.lexists(dst / name):
            os.symlink((src / name).resolve(), dst / name)
    for sid in sids:
        sd = dst / "scans" / sid
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "rgb").mkdir(exist_ok=True)
        for p in (src / "scans" / sid).iterdir():
            if p.name != "rgb" and not os.path.lexists(sd / p.name):
                os.symlink(p.resolve(), sd / p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--scans", nargs="*")
    ap.add_argument("--frames", nargs="*", type=int)
    ap.add_argument("--work-w", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--tile-scale", type=float, default=0.6)
    ap.add_argument("--depth-scale", type=float, default=0.5)
    ap.add_argument("--lf-sigma", type=float, default=0.0, help="anchor low-freq colour to the original (px @full-res)")
    ap.add_argument("--no-delta", action="store_true")
    ap.add_argument("--no-grain", action="store_true")
    ap.add_argument("--no-offload", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    sids = a.scans or sorted(p.name for p in (src / "scans").iterdir())
    prepare_root(src, dst, sids)
    pipe = load_pipe(not a.no_offload)
    t0, n = time.time(), 0
    for sid in sids:
        spec = json.load(open(src / "specs" / sid / "spec.json"))
        prompt, seed = build_prompt(spec), seed_of(sid)
        jpgs = sorted((src / "scans" / sid / "rgb").glob("*.jpg"))
        for j in jpgs:
            k = int(j.stem)
            if a.frames and k not in a.frames:
                continue
            outp = dst / "scans" / sid / "rgb" / j.name
            if outp.exists():
                continue
            orig = np.asarray(Image.open(j).convert("RGB"))
            raw = np.load(src / "raw" / sid / f"frame_{k:06d}.npz")
            res = refine_frame(pipe, orig, raw["depth"], raw["seg"], prompt, seed, a.strength, a.work_w,
                               a.steps, a.cfg, a.tile_scale, a.depth_scale, not a.no_delta, not a.no_grain)
            if a.lf_sigma > 0:
                res = anchor_lowfreq(orig, res, a.lf_sigma)
            Image.fromarray(res).save(outp, quality=90)
            n += 1
            print(f"{sid} {k} done  {(time.time() - t0) / n:.1f}s/frame", flush=True)


if __name__ == "__main__":
    main()
