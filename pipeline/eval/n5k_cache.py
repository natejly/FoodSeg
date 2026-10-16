"""Precompute fixed-size training inputs for eval.n5k_train.

    python -m eval.n5k_cache ../data/datasets/Nutrition5k --masks ../results/n5k/masks_v2 \
        --out ../data/cache/n5k_448

Per dish at 448x336 by default, --size 640 480 for full resolution (memmapped .npy, one row per dish, order in ids.txt):
    rgb.npy        uint8  (N,336,448,3)
    h_counter.npy  fp16   (N,336,448)   height above the counter plane, m (0 = no depth)
    thick.npy      fp16   (N,336,448)   food height above its container, m: the fitted
                                        dish surface for bowls, the ring plane for plates
                                        (same rule as vol_container); 0 off the food
    mask.npy       uint8  (N,336,448)   food mask (eval.n5k_masks)
Depth-derived maps are computed at 640x480 and resized nearest-neighbour.
~5 GB for the 3,262 depth-split dishes; ~5 min on 16 cores.
"""
from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from eval.nutrition5k import (BOWL_RIM, MAX_FOOD_H, ON_SCALE_RIM, UP, backproject, dish_volume,
                              load_depth_m, load_food, read_split, ring_plane)

W, H = 448, 336


def thickness(depth: np.ndarray, food: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(height above counter, food height above container), both (480,640) float32, 0 = unknown."""
    maps: dict = {}
    info = dish_volume(depth, food, maps=maps)
    rim = info["dish_rim_mm"]
    bowl = info["n_dishes"] > 0 and rim >= 1000 * ON_SCALE_RIM and rim > 1000 * BOWL_RIM
    if bowl:
        thick = maps["dish_thick"]
    else:
        plane = ring_plane(depth, food)
        hgt = plane.height(backproject(depth))
        thick = np.zeros(depth.shape, np.float32)
        thick[depth > 0] = np.clip(hgt, 0.0, MAX_FOOD_H)
        thick[~food] = 0.0
    return np.nan_to_num(maps["h_counter"], nan=0.0), np.nan_to_num(thick, nan=0.0)


def one(d: str, img_dir: Path, mask_dir: Path, size: tuple[int, int] = (W, H)) -> tuple:
    dd = img_dir / d
    rgb = cv2.cvtColor(cv2.imread(str(dd / "rgb.png")), cv2.COLOR_BGR2RGB)
    try:
        depth = load_depth_m(dd / "depth_raw.png")
    except AttributeError:            # one N5k depth PNG is unreadable: keep RGB, zero depth
        depth = np.zeros(rgb.shape[:2])
    food = load_food(dd, mask_dir)
    if food is None:
        food = np.zeros(depth.shape, bool)
    if food.sum() >= 50 and (depth > 0).any():
        hc, th = thickness(depth, food)
    else:
        hc, th = np.zeros(depth.shape, np.float32), np.zeros(depth.shape, np.float32)
    rs = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)  # noqa: E731
    return (cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), rs(hc).astype(np.float16),
            rs(th).astype(np.float16), rs(food.astype(np.uint8)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("../data/cache/n5k_448"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--size", type=int, nargs=2, default=(W, H), metavar=("W", "H"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img_dir = args.root / "imagery/realsense_overhead"
    ids = []
    for s in ("depth_train_ids", "depth_test_ids"):
        ids += read_split(args.root, s)
    ids = [d for d in dict.fromkeys(ids) if (img_dir / d / "depth_raw.png").exists()]
    n = len(ids)
    print(f"{n} dishes")
    w, h = args.size
    mm = {k: np.lib.format.open_memmap(args.out / f"{k}.npy", "w+", dt, shape)
          for k, dt, shape in (("rgb", np.uint8, (n, h, w, 3)), ("h_counter", np.float16, (n, h, w)),
                               ("thick", np.float16, (n, h, w)), ("mask", np.uint8, (n, h, w)))}
    with Pool(args.workers) as pool:
        for i, out in enumerate(pool.imap(partial(one, img_dir=img_dir, mask_dir=args.masks, size=(w, h)), ids, chunksize=4)):
            for k, a in zip(("rgb", "h_counter", "thick", "mask"), out):
                mm[k][i] = a
            if i % 400 == 0:
                print(f"  {i}/{n}", flush=True)
    for a in mm.values():
        a.flush()
    (args.out / "ids.txt").write_text("\n".join(ids) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
