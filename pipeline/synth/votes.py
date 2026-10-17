"""Per-frame instance masks for lifting: oracle, pseudo-SAM (corrupted oracle) or real SAM 3.

All maps are uint8 at the LiDAR depth resolution (256x192): 0 background, 1..K food items,
253 clutter, 254 container. The same label ids are used in every frame of a scene.

pseudo_sam() imitates what a video segmenter gets wrong, seeded from the scene seed so a scan's
corruption is reproducible; its parameters are returned for gt.json["votes"]:

    boundary jitter   per frame, every instance is eroded or dilated by 1-3 px
    merge             p=0.2 per scene: two touching items share one id in a random subset of frames
    drop              p=0.05 per item: the item vanishes in a random subset of frames
    container FP      p=0.3 per scene: the container is labelled as an extra 'food' id (K+1)
                      in a random subset of frames (it keeps 254 in the others)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CONTAINER_ID, CLUTTER_ID = 254, 253
D_W, D_H = 256, 192


def _to_depth_res(m: np.ndarray) -> np.ndarray:
    if m.shape == (D_H, D_W):
        return np.array(m)
    return np.array(Image.fromarray(m).resize((D_W, D_H), Image.NEAREST))


def load_oracle_masks(scan: Path) -> dict[int, np.ndarray]:
    """Full oracle id maps at depth resolution. New scans: gt/seg_ids (food + 253 + 254). Older
    scans only have food-only gt/masks plus gt/container, so clutter (253) is unrecoverable there."""
    scan = Path(scan)
    out: dict[int, np.ndarray] = {}
    full = scan / "gt" / "seg_ids"
    if full.exists() and any(full.glob("*.png")):
        for p in sorted(full.glob("*.png")):
            out[int(p.stem)] = _to_depth_res(np.asarray(Image.open(p)))
        return out
    for p in sorted((scan / "gt" / "masks").glob("*.png")):
        m = _to_depth_res(np.asarray(Image.open(p)).copy())
        c = scan / "gt" / "container" / p.name
        if c.exists():
            cm = _to_depth_res(np.asarray(Image.open(c))) > 0
            m[(m == 0) & cm] = CONTAINER_ID
        out[int(p.stem)] = m
    return out


def from_sam3_masks(frame_mask_dir: Path) -> dict[int, np.ndarray]:
    """HOOK for real SAM 3 labels: `<scan>/masks_sam3/NNNNNN.png` as written by
    foodscan/segment.py (uint8 label map at video resolution, 0 = background, 1..K instance ids
    consistent across frames). Returned at depth resolution (nearest). SAM ids are *not* the GT
    item ids; to compare with GT, match them first (Hungarian on mask IoU), which is not done here."""
    d = Path(frame_mask_dir)
    return {int(p.stem): _to_depth_res(np.asarray(Image.open(p)).copy()) for p in sorted(d.glob("*.png"))}


def _touching_pairs(masks: dict[int, np.ndarray], ids: list[int]) -> list[tuple[int, int]]:
    pairs = set()
    for m in masks.values():
        for i in ids:
            if not (m == i).any():
                continue
            ring = ndi.binary_dilation(m == i, iterations=2) & ~(m == i)
            for j in set(np.unique(m[ring]).tolist()) & set(ids):
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _jitter(m: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    out = np.zeros_like(m)
    ids = [i for i in np.unique(m) if i]
    ops = []
    for i in rng.permutation(ids):
        k = int(rng.integers(1, 4)) * (1 if rng.random() < 0.5 else -1)
        b = m == i
        b2 = ndi.binary_dilation(b, iterations=k) if k > 0 else ndi.binary_erosion(b, iterations=-k)
        out[b2 & (out == 0)] = i
        ops.append(k)
    return out, ops


def pseudo_sam(masks: dict[int, np.ndarray], seed: int, item_ids: list[int]) -> tuple[dict[int, np.ndarray], dict]:
    rng = np.random.default_rng([int(seed), 7919])
    frames = sorted(masks)
    present = {i for m in masks.values() for i in np.unique(m).tolist()}
    items = [i for i in item_ids if i in present]
    params: dict = {"mode": "pseudo", "seed": int(seed), "jitter_px": [1, 3], "merge": None,
                    "dropped": {}, "container_fp": None}
    if len(items) >= 2 and rng.random() < 0.2:
        pairs = _touching_pairs(masks, items)
        if pairs:
            a, b = pairs[int(rng.integers(len(pairs)))]
            sub = sorted(rng.choice(frames, size=max(1, len(frames) // 2), replace=False).tolist())
            params["merge"] = {"ids": [a, b], "into": a, "frames": sub}
    for i in items:
        if rng.random() < 0.05:
            n = max(1, int(len(frames) * rng.uniform(0.2, 0.6)))
            params["dropped"][str(i)] = sorted(rng.choice(frames, size=n, replace=False).tolist())
    fp_id = (max(items) if items else 0) + 1
    if CONTAINER_ID in present and rng.random() < 0.3:
        n = max(1, int(len(frames) * rng.uniform(0.2, 0.7)))
        params["container_fp"] = {"as_id": fp_id, "frames": sorted(rng.choice(frames, size=n, replace=False).tolist())}
    out = {}
    for fi in frames:
        m = masks[fi].copy()
        if params["merge"] and fi in params["merge"]["frames"]:
            m[m == params["merge"]["ids"][1]] = params["merge"]["into"]
        for i, fr in params["dropped"].items():
            if fi in fr:
                m[m == int(i)] = 0
        if params["container_fp"] and fi in params["container_fp"]["frames"]:
            m[m == CONTAINER_ID] = fp_id
        out[fi], _ = _jitter(m, rng)
    return out, params


def get_masks(scan: Path, mode: str, seed: int, item_ids: list[int]) -> tuple[dict[int, np.ndarray], dict]:
    if mode == "sam3":
        return from_sam3_masks(Path(scan) / "masks_sam3"), {"mode": "sam3"}
    oracle = load_oracle_masks(scan)
    if mode == "oracle":
        return oracle, {"mode": "oracle"}
    return pseudo_sam(oracle, seed, item_ids)


def lift(scan, points: np.ndarray, normals: np.ndarray, masks: dict[int, np.ndarray],
         depth_tol: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visibility-gated, cosine-weighted voting; same gating as foodscan.labels.lift_masks
    (|z - lidar depth| < depth_tol, grazing > 75 deg dropped), but with a compact column per
    distinct label so ids up to 254 do not need an (N, 255) array. Background (0) votes count.
    Returns lift_label int32, lift_share float32, lift_nvotes int16."""
    from foodscan.labels import visible_indices
    cols = sorted({0} | {int(i) for m in masks.values() for i in np.unique(m)})
    col_of = np.zeros(256, np.int64)
    for c, i in enumerate(cols):
        col_of[i] = c
    votes = np.zeros((len(points), len(cols)), np.float32)
    nv = np.zeros(len(points), np.int32)
    by_index = {f.index: f for f in scan.frames}
    for fi, m in masks.items():
        fr = by_index.get(fi)
        if fr is None:
            continue
        idx, mu, mv, w = visible_indices(fr, points, m.shape, depth_tol=depth_tol, normals=normals)
        if len(idx) == 0:
            continue
        np.add.at(votes, (idx, col_of[m[mv, mu]]), w)
        nv[idx] += 1
    total = votes.sum(1)
    win = votes.argmax(1)
    label = np.where(total > 0, np.asarray(cols)[win], 0).astype(np.int32)
    share = np.where(total > 0, votes.max(1) / np.maximum(total, 1e-9), 0).astype(np.float32)
    return label, share, np.minimum(nv, 32767).astype(np.int16)
