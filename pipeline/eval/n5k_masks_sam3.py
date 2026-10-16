"""Food masks for Nutrition5k overhead RGB with SAM 3 (image model, text prompts).

    python -m eval.n5k_masks_sam3 ../data/datasets/Nutrition5k --out ../results/n5k/masks_sam3

Replaces eval.n5k_masks (Grounding DINO + SAM 2.1). Prompts are generic only
("food", plus "plate" and "bowl"), never the dish's ingredient names, so the
masks stay usable for fair (non-oracle) numbers. A food instance that overlaps a
plate or bowl instance with IoU > --dish-iou is dropped (SAM 3 sometimes calls a
whole plate of food "food"), as in eval.n5k_masks.

6 GB Turing card: the image and text encoders run in fp16 (backbone.half()),
the rest in fp32 under fp16 autocast; see the local-GPU notes. Writes one uint8
PNG per dish (255 = food); resumable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.nutrition5k import read_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/n5k/masks_sam3"))
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--dish-iou", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    img_dir = args.root / "imagery/realsense_overhead"
    ids = []
    for s in ("depth_train_ids", "depth_test_ids"):
        ids += read_split(args.root, s)
    todo = [d for d in dict.fromkeys(ids)
            if (img_dir / d / "rgb.png").exists() and not (args.out / f"{d}.png").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} dishes to mask", flush=True)

    model = build_sam3_image_model(device="cpu")
    if args.device == "cuda":
        model.backbone.half()
    model = model.to(args.device)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.thresh)

    def masks(state, prompt: str, shape: tuple[int, int]) -> np.ndarray:
        o = proc.set_text_prompt(state=state, prompt=prompt)
        m = o["masks"]
        return m.reshape(-1, *m.shape[-2:]).bool().cpu().numpy() if len(o["scores"]) else np.zeros((0, *shape), bool)

    for i, d in enumerate(todo):
        img = Image.open(img_dir / d / "rgb.png").convert("RGB")
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"):
            st = proc.set_image(img)
            hw = (img.height, img.width)
            food = masks(st, "food", hw)
            dish = np.concatenate([masks(st, "plate", hw), masks(st, "bowl", hw)])
        out = np.zeros((img.height, img.width), bool)
        for f in food:
            iou = [(f & q).sum() / max((f | q).sum(), 1) for q in dish]
            if not iou or max(iou) < args.dish_iou:
                out |= f
        Image.fromarray((out * 255).astype(np.uint8)).save(args.out / f"{d}.png")
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} {d}: {len(food)} food / {len(dish)} dish masks, {out.mean():.3f} of image",
                  flush=True)


if __name__ == "__main__":
    main()
