"""Dataset index: difficulty tags and QC flags for every synthetic scan (docs/dataset_curation.md).

    python -m synth.manifest ../data/synth/hard/scans [--out ../data/synth/hard/manifest.jsonl]

Reads gt.json, gt/points.npz and (if synth.seg_export has run) gt/seg_points.npz. One JSON
line per scan with tags (n_items, touching / same-category / similar-colour pairs,
stacked, container, clutter, min_item_points) and `qc` flags; `ok` is False when any
dropping flag is set. `--report` adds per-batch ICP/drop stats. Prints a summary of tag and flag counts. Runs in `synth` or `foodseg`.
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
DROP = {"off_container", "leaves_off_dish", "barely_visible", "volume_mismatch", "fusion_misaligned",
        "not_exported", "tiny_item", "invisible_item"}      # pose_drift_large is informational, not dropping
VOLUME_TOL = 0.10        # |scene - spec| volume mismatch above this fraction of the item volume
MIN_FUSED_POINTS = 50    # at REF_VIEWS views; scaled by n_views / REF_VIEWS (capped at 1)
REF_VIEWS = 24
POST_ICP_MAX_M = 0.002   # median fused-to-GT distance after the rigid ICP correction
DRIFT_BASE_M = 0.002     # slack added to the pose-drift bound


def min_points_threshold(n_views: int | None) -> float:
    return MIN_FUSED_POINTS * min(1.0, (n_views or REF_VIEWS) / REF_VIEWS)


def drift_bound(pn: dict, n_frames: int) -> float:
    """Expected bound (m) on the pre-ICP median fused-to-GT distance from PoseNoiseParams.
    Per-frame velocity is AR(1) with momentum m and per-step rms `trans_step_m`, so the accumulated
    translation after n frames has rms  step * sqrt(n (1+m)/(1-m))  (random walk inflated by the
    momentum); tracking jumps (prob jump_p, mean magnitude jump_m) add  sqrt(n jump_p) * mean jump.
    Bound = 3 * rms_walk + jump term + 2 mm (rotation is ~0.1 deg/frame, mm-level at 30 cm, folded in
    by the 3x). The fused cloud sits at the average drift, so the median distance is below the end drift."""
    n = max(int(n_frames), 1)
    m = float(pn.get("momentum", 0.0))
    walk = float(pn["trans_step_m"]) * np.sqrt(n * (1 + m) / (1 - m))
    jm = pn.get("jump_m") or (0.0, 0.0)
    jump = np.sqrt(n * float(pn.get("jump_p", 0.0))) * 0.5 * (jm[0] + jm[1])
    return float(3 * walk + jump + DRIFT_BASE_M)


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
    if any(it["frames_visible"] == 0 for it in items):
        qc.append("invisible_item")
    elif any(it["frames_visible"] < 3 for it in items):
        qc.append("barely_visible")
    bad = [it for it in items if abs(it["scene_volume_ml"] - it["volume_ml"]) > VOLUME_TOL * it["volume_ml"]]
    causes = {}
    if bad:
        qc.append("volume_mismatch")
        for it in bad:
            sh = (it.get("shape") or {}).get("kind")
            causes[it["id"]] = ("off_dish" if it["off_container"] else
                                "cloth_sim" if sh == "leaves" else
                                "scaled_rescue" if it["scene_volume_ml"] < it["volume_ml"] and sh == "single" else
                                "unknown")
    min_pts = None
    min_thr = pre = post = bound = None
    prov = gt.get("provenance") or {}
    n_views = prov.get("n_views")
    if n_views is None and (scan / "frames.jsonl").exists():
        n_views = sum(1 for _ in open(scan / "frames.jsonl"))
    pn = gt.get("pose_noise")
    if pn and n_views:
        bound = drift_bound(pn, n_views)
    sigma0 = (gt.get("lidar_params") or {}).get("sigma0_m")
    dropped: list[int] = [i for i in by_id if by_id[i]["frames_visible"] == 0]
    sp = scan / "gt" / "seg_points.npz"
    if sp.exists():
        s = np.load(sp)
        lab = s["label"]
        counts = [int((lab == i).sum()) for i in by_id]
        min_pts = min(counts) if counts else 0
        min_thr = min_points_threshold(n_views)
        if min_pts < min_thr:
            qc.append("tiny_item")
            dropped += [i for i, c in zip(by_id, counts) if c < min_thr and i not in dropped]
        # alignment on labelled (non-table) points only; table points have no GT samples.
        # Poses carry drift, so judge the residual after seg_export's rigid ICP correction.
        se = gt.get("seg_export") or {}
        on = lab > 0
        post = se.get("median_post_mm")
        pre = se.get("median_pre_mm")
        if post is None and on.any():
            post = pre = float(np.median(s["dist"][on]) * 1000)      # legacy export: no ICP stats
        if post is not None and post > POST_ICP_MAX_M * 1000:
            qc.append("fusion_misaligned")
        if pre is not None and bound is not None and pre > bound * 1000:
            qc.append("pose_drift_large")
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
            "min_item_points_threshold": min_thr, "n_views": n_views,
            "lidar_sigma0_mm": None if sigma0 is None else sigma0 * 1000,
            "icp_pre_mm": pre, "icp_post_mm": post, "drift_bound_mm": None if bound is None else bound * 1000,
            "tol_mm": ((gt.get("seg_export") or {}).get("tol") or 0) * 1000 or None,
            "total_volume_ml": gt["total_volume_ml"], "qc": qc, "volume_mismatch_cause": causes,
            "dropped_items": dropped, "ok": not (set(qc) & DROP)}


def report(rows: list[dict]) -> None:
    ok = [r for r in rows if r["ok"]]
    n_items = sum(r["n_items"] for r in rows)
    n_drop = sum(len(r["dropped_items"]) for r in rows)
    med = lambda k: (float(np.median([r[k] for r in rows if r.get(k) is not None]))
                     if any(r.get(k) is not None for r in rows) else float("nan"))
    tol = [r["tol_mm"] for r in rows if r.get("tol_mm")]
    print(f"report: scans ok {len(ok)}/{len(rows)}, items ok {n_items - n_drop}/{n_items}")
    drops = Counter(q for r in rows for q in set(r["qc"]) & DROP)
    print("  scans dropped by reason:", dict(drops) or "none")
    print(f"  median ICP distance pre {med('icp_pre_mm'):.2f} mm -> post {med('icp_post_mm'):.2f} mm, "
          f"mean tol {np.mean(tol) if tol else float('nan'):.2f} mm, "
          f"pose_drift_large {sum('pose_drift_large' in r['qc'] for r in rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--out", default=None, help="default: <scans>/../manifest.jsonl")
    ap.add_argument("--report", action="store_true", help="print the per-batch ICP / drop-reason report")
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
    if args.report:
        report(rows)
    flags = Counter(q for r in rows for q in r["qc"])
    print("qc flags:", dict(flags))
    print("scenes with touching pairs:", sum(r["touching_pairs"] > 0 for r in rows),
          " same-category:", sum(r["same_category_pairs"] > 0 for r in rows),
          " similar colour:", sum(r["similar_colour_pairs"] > 0 for r in rows))


if __name__ == "__main__":
    main()
