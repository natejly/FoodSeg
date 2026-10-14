"""Dataset index: difficulty tags and QC flags for every synthetic scan (docs/dataset_curation.md).

    python -m synth.manifest ../data/synth/hard/scans [--out ../data/synth/hard/manifest.jsonl]

Reads gt.json, gt/points.npz and (if synth.seg_export has run) gt/seg_points.npz. One JSON
line per scan with tags (n_items, touching / same-category / similar-colour pairs,
stacked, container, clutter, min_item_points) and `qc` flags; `ok` is False when any
dropping flag is set. Prints a summary of tag and flag counts. Runs in `synth` or `foodseg`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

TOUCH_M = 0.003          # GT surfaces closer than this touch (~1 mm sample spacing + collision margins)
SIMILAR_RGB = 0.1        # mean colour difference (0-1 sRGB) below which neighbours look alike
DROP = {"off_container", "leaves_off_dish", "barely_visible", "volume_mismatch", "fusion_misaligned"}


def _item_colours(scan: Path, n_items: int) -> dict[int, np.ndarray]:
    """Mean rendered colour per item from the first frame's mask (0-1 sRGB)."""
    from PIL import Image
    rgb = np.asarray(Image.open(scan / "rgb" / "000000.jpg"), np.float32) / 255
    m = np.asarray(Image.open(scan / "gt" / "masks" / "000000.png"))
    if m.shape != rgb.shape[:2]:
        m = np.asarray(Image.fromarray(m).resize(rgb.shape[1::-1], Image.NEAREST))
    return {i: rgb[m == i].mean(0) for i in range(1, n_items + 1) if (m == i).sum() > 50}


def describe(scan: Path) -> dict:
    gt = json.loads((scan / "gt.json").read_text())
    items = gt["items"]
    by_id = {it["id"]: it for it in items}
    z = np.load(scan / "gt" / "points.npz")
    pts = {i: z["xyz"][z["label"] == i] for i in by_id}
    trees = {i: cKDTree(p) for i, p in pts.items() if len(p)}
    cols = _item_colours(scan, len(items))

    touching, same_cat, similar = [], [], []
    ids = sorted(trees)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if trees[i].query(pts[j], distance_upper_bound=TOUCH_M)[0].min() <= TOUCH_M:
                touching.append((i, j))
                if by_id[i]["category"] == by_id[j]["category"]:
                    same_cat.append((i, j))
                if i in cols and j in cols and np.abs(cols[i] - cols[j]).mean() < SIMILAR_RGB:
                    similar.append((i, j))

    qc = []
    if any(it["off_container"] for it in items):
        qc.append("off_container")
    # a salad's bbox centre stays on the dish even when leaves slid onto the table: check its surface
    W2A = np.array(gt["world_to_arkit"])
    r_dish = gt["container"]["R"] + gt["container"].get("rim_w", 0.0)
    for i in by_id:
        if (by_id[i].get("shape") or {}).get("kind") == "leaves" and len(pts[i]):
            loc = (pts[i] - W2A[:3, 3]) @ W2A[:3, :3]
            if (np.hypot(loc[:, 0], loc[:, 1]) > r_dish).mean() > 0.10:
                qc.append("leaves_off_dish")
    if any(it["frames_visible"] < 3 for it in items):
        qc.append("barely_visible")
    if any(abs(it["scene_volume_ml"] - it["volume_ml"]) > 0.01 * it["volume_ml"] for it in items):
        qc.append("volume_mismatch")
    min_pts = None
    sp = scan / "gt" / "seg_points.npz"
    if sp.exists():
        s = np.load(sp)
        lab = s["label"]
        counts = [int((lab == i).sum()) for i in by_id]
        min_pts = min(counts) if counts else 0
        if min_pts < 50:
            qc.append("tiny_item")
        # alignment on labelled (non-table) points only; table points have no GT samples
        on = lab > 0
        if on.any() and np.median(s["dist"][on]) > 0.002:
            qc.append("fusion_misaligned")
    else:
        qc.append("not_exported")

    kinds = Counter((it.get("shape") or {}).get("kind", "v0") for it in items)
    kind = {i: (by_id[i].get("shape") or {}).get("kind") for i in by_id}
    on_food = any({kind[i], kind[j]} & {"piece"} and {kind[i], kind[j]} & {"single", "pile"}
                  for i, j in touching)
    return {"scan": scan.name, "preset": gt.get("preset", "v0"), "container": gt["container"]["kind"],
            "n_items": len(items), "touching_pairs": len(touching), "same_category_pairs": len(same_cat),
            "similar_colour_pairs": len(similar),
            "stacked": kinds.get("topping", 0) > 0 or on_food,     # topping, or an item resting on food
            "clutter": len(gt.get("distractors", [])), "shape_kinds": dict(kinds),
            "categories": sorted({it["category"] for it in items}), "min_item_points": min_pts,
            "total_volume_ml": gt["total_volume_ml"], "qc": qc, "ok": not (set(qc) & DROP)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--out", default=None, help="default: <scans>/../manifest.jsonl")
    args = ap.parse_args()
    root = Path(args.scans)
    out = Path(args.out) if args.out else root.parent / "manifest.jsonl"
    rows = [describe(s) for s in sorted(p for p in root.iterdir() if (p / "gt.json").exists())]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} scans -> {out}; ok {sum(r['ok'] for r in rows)}")
    for r in rows:
        print(f"  {r['scan']} {r['container']:<5} items {r['n_items']} touch {r['touching_pairs']} "
              f"same-cat {r['same_category_pairs']} similar {r['similar_colour_pairs']} "
              f"stacked {int(r['stacked'])} clutter {r['clutter']} min_pts {r['min_item_points']} qc {r['qc']}")
    flags = Counter(q for r in rows for q in r["qc"])
    print("qc flags:", dict(flags))
    print("scenes with touching pairs:", sum(r["touching_pairs"] > 0 for r in rows),
          " same-category:", sum(r["same_category_pairs"] > 0 for r in rows),
          " similar colour:", sum(r["similar_colour_pairs"] > 0 for r in rows))


if __name__ == "__main__":
    main()
