"""Frozen DINOv2 embeddings of Nutrition5k overhead RGB, for eval.n5k_regress.

    python -m eval.n5k_features ../data/datasets/Nutrition5k --masks ../results/n5k/masks_v2 \
        --out ../results/n5k/features_dinov2l.npz

Per dish, three 1024-d vectors from DINOv2-L at 448x336 (whole frame, no crop):
    cls   the CLS token
    mean  mean of all patch tokens
    food  mean of patch tokens weighted by the food mask (eval.n5k_masks) at
          patch resolution; zeros when the mask is empty
Each is averaged over the 4 flips (overhead views have no canonical orientation).
fp16 on GPU; ~5 min on an RTX 2060.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

MODEL = "facebook/dinov2-large"
W, H, P = 448, 336, 14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("../results/n5k/features_dinov2l.npz"))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    img_dir = args.root / "imagery/realsense_overhead"
    ids = []
    for s in ("depth_train_ids", "depth_test_ids"):
        ids += [l.strip() for l in open(args.root / "dish_ids/splits" / f"{s}.txt") if l.strip()]
    ids = [d for d in dict.fromkeys(ids) if (img_dir / d / "rgb.png").exists()]
    print(f"{len(ids)} dishes")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype).to(dev).eval()

    feats = {k: np.zeros((len(ids), 1024), np.float32) for k in ("cls", "mean", "food")}
    for b0 in range(0, len(ids), args.batch):
        batch = ids[b0:b0 + args.batch]
        imgs = torch.stack([torch.from_numpy(np.asarray(
            Image.open(img_dir / d / "rgb.png").convert("RGB").resize((W, H), Image.BICUBIC))).permute(2, 0, 1)
            for d in batch]).float() / 255
        imgs = (imgs - MEAN) / STD
        masks = []
        for d in batch:
            mp = args.masks / f"{d}.png"
            m = np.asarray(Image.open(mp).convert("L")) > 0 if mp.exists() else np.zeros((480, 640), bool)
            masks.append(torch.from_numpy(m.astype(np.float32))[None])
        masks = torch.stack(masks)                                          # (B,1,480,640)
        acc = {k: 0.0 for k in feats}
        for flip in ([], [3], [2], [2, 3]):
            x = imgs.flip(flip) if flip else imgs
            m = masks.flip(flip) if flip else masks
            with torch.no_grad():
                tok = model(pixel_values=x.to(dev, dtype)).last_hidden_state.float().cpu()
            cls, patches = tok[:, 0], tok[:, 1:]                            # (B,N,1024), row-major grid
            w = F.adaptive_avg_pool2d(m, (H // P, W // P)).flatten(1)       # food fraction per patch
            food = (patches * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)
            acc["cls"] += cls / 4
            acc["mean"] += patches.mean(1) / 4
            acc["food"] += food / 4
        for k in feats:
            feats[k][b0:b0 + len(batch)] = acc[k].numpy()
        if b0 % 400 == 0:
            print(f"  {b0}/{len(ids)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids), **{k: v.astype(np.float16) for k, v in feats.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
