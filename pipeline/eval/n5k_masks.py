"""Food masks for Nutrition5k overhead RGB: Grounding DINO ("food") boxes -> SAM 2.1 masks.

    python -m eval.n5k_masks data/datasets/Nutrition5k --out ../results/n5k/masks

Stand-in for foodscan.segment (SAM 3 text prompts) while the SAM 3 weights
are gated: both are open-vocabulary text -> mask, ungated, and fit a 6 GB GPU.
Writes one uint8 PNG per dish (255 = food). Grounding DINO also calls whole
plates "food", and full meals cover more of the frame than an empty plate,
so an area cutoff cannot separate them. Instead plates and bowls are detected
as their own classes and any food mask with IoU > --plate-iou against one is
dropped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                          Sam2Processor)

GDINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../results/n5k/masks"))
    ap.add_argument("--prompt", default="plate. bowl. food.")
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--max-box-frac", type=float, default=0.9)    # tray / whole rig
    ap.add_argument("--plate-iou", type=float, default=0.6)
    ap.add_argument("--splits", default="depth_train_ids,depth_test_ids")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gp = AutoProcessor.from_pretrained(GDINO)
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(dev).eval()
    sp = Sam2Processor.from_pretrained(SAM2)
    sm = Sam2Model.from_pretrained(SAM2).to(dev).eval()

    img_dir = args.root / "imagery/realsense_overhead"
    ids = []
    for s in args.splits.split(","):
        ids += [l.strip() for l in open(args.root / "dish_ids/splits" / f"{s}.txt") if l.strip()]
    todo = [d for d in ids if (img_dir / d / "rgb.png").exists() and not (args.out / f"{d}.png").exists()]
    print(f"{len(todo)} dishes to mask")

    for i, d in enumerate(todo):
        img = Image.open(img_dir / d / "rgb.png").convert("RGB")
        W, H = img.size
        with torch.no_grad():
            inp = gp(images=img, text=args.prompt, return_tensors="pt").to(dev)
            det = gp.post_process_grounded_object_detection(
                gm(**inp), inp.input_ids, threshold=args.box_thresh, text_threshold=0.25,
                target_sizes=[(H, W)])[0]
            boxes = det["boxes"].cpu().numpy()
            labels = det.get("text_labels", det.get("labels"))
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (W * H)
            keep = area < args.max_box_frac
            boxes, labels = boxes[keep], [l for l, k in zip(labels, keep) if k]
            is_food = np.array(["food" in str(l) for l in labels], bool)
            mask = np.zeros((H, W), bool)
            if is_food.any():
                si = sp(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(dev)
                so = sm(**si, multimask_output=False)
                m = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"])[0]
                inst = m.squeeze(1).numpy().astype(bool)          # (n_boxes, H, W)
                food, dish = inst[is_food], inst[~is_food]
                for f in food:
                    iou = [(f & d).sum() / max((f | d).sum(), 1) for d in dish]
                    if not iou or max(iou) < args.plate_iou:
                        mask |= f
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.out / f"{d}.png")
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} {d}: {int(is_food.sum())} food / {len(boxes)} boxes, {mask.mean():.3f} of image", flush=True)


if __name__ == "__main__":
    main()
