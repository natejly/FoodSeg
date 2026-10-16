"""Error analysis of the remaining lifted-label errors after the co-vote linker (CPU only, no SAM).

    python -m eval.lift_errors [--procs 4] [--only s000000 ...]    # -> results/lift_errors/{items.csv,results.json,summary.md}

Inputs: results/lift_replay/<sid>/lift.npz (replayed co-vote + cos+qual lift), the raw per-frame SAM 3 maps
(data/synth/hard_v12/scans/<sid>/masks_sam3_v2i/raw, per-frame-unique ids) and the GT cloud. Metrics are on the
2 mm canonical voxel scene, manifest-dropped items ignored (same 52 items as results/lift_sweep).
An optional results/lift_errors/reading.md is pasted at the top of summary.md.

Error type per GT item (priority order): ok (IoU >= .75) > missed (< 5 % of its points carry a food id) >
fragmented (>= 2 lift ids each > 20 % of the item) > merged (an id holding > 20 % of the item also holds > 20 % of
another item) > boundary (IoU .5-.75, single id) > under-covered (everything else: low IoU, no competing id).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline"))

OUT = REPO / "results" / "lift_errors"
REPLAY = REPO / "results" / "lift_replay"
FRAC = 0.20          # an id "counts" for an item when it holds more than this share of the item's points
TYPES = ["ok", "boundary", "fragmented", "merged", "under-covered", "missed"]
MIN_VIS = 30         # visible points for an item (or fragment) to count as seen in a frame
CONTACT = 0.004      # m, 3D contact / seam radius


REPLAY40 = REPO / "results" / "lift_replay40"   # `python -m foodscan.boundary_refine replay40` (any-batch co-vote replay)
_use40 = False


def replay_dir():
    return REPLAY40 if _use40 else REPLAY


def batch_of(sid: str) -> str:
    return sid.rpartition("/")[0] or "v12"


# ------------------------------------------------------------------------------------------------ pure helpers
def classify(iou: float, food_cov: float, id_fracs: dict, other_cover: dict) -> tuple[str, dict]:
    """id_fracs {lift id: share of this item's points}; other_cover {lift id: max share of any OTHER item's points}.
    Returns (primary error type, flags)."""
    big = [i for i, f in id_fracs.items() if f > FRAC]
    frag = len(big) >= 2
    merged = any(other_cover.get(i, 0.0) > FRAC for i in big)
    flags = dict(fragmented=frag, merged=merged, missed=food_cov < 0.05)
    if iou >= 0.75:
        return "ok", flags
    if flags["missed"]:
        return "missed", flags
    if frag:
        return "fragmented", flags
    if merged:
        return "merged", flags
    if iou >= 0.5:
        return "boundary", flags
    return "under-covered", flags


def snap_iou(i: int, g: int, p: int, fn_d: np.ndarray, fp_d: np.ndarray, thr: float) -> float:
    """IoU after snapping every misassigned point within `thr` of the true boundary to GT.
    i = |g & p|, g = |g|, p = |pred|; fn_d / fp_d = distances to the true boundary of the missed / false points."""
    fn_n, fp_n = int((fn_d <= thr).sum()), int((fp_d <= thr).sum())
    inter = i + fn_n
    union = g + (len(fp_d) - fp_n)           # pred' = pred + fn_near - fp_near; union with g is g + remaining false points
    return inter / max(union, 1)


def item_kind(it: dict) -> str:
    sh = (it.get("shape") or {}).get("kind", "")
    src = it.get("source", "")
    if sh == "topping" or "topping" in src:
        return "topping"
    if sh == "leaves" or "leaves" in src:
        return "leaf"
    if sh == "pile":
        return "pile"
    if src.startswith("procedural") or src in ("primitive", "objaverse"):
        return "procedural"
    return "scan"


def size_bin(v: float) -> str:
    return "<15 ml" if v < 15 else "15-50 ml" if v < 50 else "50-150 ml" if v < 150 else ">=150 ml"


# ------------------------------------------------------------------------------------------------ per-scan analysis
def analyse(sid: str) -> dict:
    from scipy.spatial import cKDTree
    from eval.lift_sweep import AUX, BIG, ScanData, roots
    from seg3d import metrics as M

    sd = ScanData(sid)
    scan_dir = roots(sid)[0]
    gt = json.loads((scan_dir / "gt.json").read_text())
    items = {int(it["id"]): it for it in gt["items"]}
    zfull = np.load(scan_dir / "gt" / "seg_points.npz")
    glab_full = zfull["label"].astype(np.int32)
    rgb = zfull["rgb"].astype(np.float32)
    rgb = rgb * 255 if rgb.max() <= 1.5 else rgb
    lift_full = np.load(replay_dir() / sid.replace("/", "__") / "lift.npz")["lift_label"].astype(np.int64)
    lift_full = np.where((lift_full > 0) & (lift_full < 252), lift_full, 0)       # food ids only

    sc, vi = sd.sc, sd.vi
    sel = ~sc["ignore"]
    vfull = vi[sel]                                    # full-cloud index of each evaluated voxel
    xyz, gi, lab = sc["xyz"][sel], sc["inst"][sel], sc["label"][sel]
    nrm = sd.nrm[vfull]
    pi = lift_full[vfull]
    g_ids, _, m = M.match(gi, pi)
    n_items = len(g_ids)
    foodm = (gi > 0)
    tree_all = cKDTree(xyz)

    # container rim: top 15 % of the container's height range
    cm = lab == 254
    if cm.sum() > 50:
        zc = xyz[cm, 2]
        rim_xyz = xyz[cm][zc >= zc.min() + 0.85 * (zc.max() - zc.min())]
        rim_tree = cKDTree(rim_xyz)
    else:
        rim_tree = None

    # ---- per-frame raw-map view of every full-cloud point
    frame_ids = sorted(sd.work)
    fr = []                                            # (frame index, point idx, raw id, score, edge)
    for fv in sd.fvs:
        fr.append((frame_ids[fv.pos], fv.idx, fv.lab, fv.score, fv.edge))

    rows, frag_pairs, merge_pairs = [], [], []
    boundary = []
    for g in g_ids:
        it = items[g]
        gm = gi == g
        n = int(gm.sum())
        p_match, iou = (m[g] if g in m else (0, 0.0))
        ids, cnt = np.unique(pi[gm][pi[gm] > 0], return_counts=True)
        id_fracs = {int(i): c / n for i, c in zip(ids, cnt)}
        food_cov = float((pi[gm] > 0).mean())
        other_cover = {}
        for h in g_ids:
            if h == g:
                continue
            hm = gi == h
            hi, hc = np.unique(pi[hm][pi[hm] > 0], return_counts=True)
            for i, c in zip(hi, hc):
                other_cover[int(i)] = max(other_cover.get(int(i), 0.0), c / hm.sum())
        etype, flags = classify(iou, food_cov, id_fracs, other_cover)

        # 3D neighbourhood: contact / stacked / rim
        pts = xyz[gm]
        near = tree_all.query_ball_point(pts, CONTACT)
        touch_n = 0
        above = 0
        under = 0
        touch_with = set()
        for k, nb in enumerate(near):
            oth = [j for j in nb if foodm[j] and gi[j] != g]
            if oth:
                touch_n += 1
                touch_with.update(int(gi[j]) for j in oth)
                dz = pts[k, 2] - np.mean([xyz[j, 2] for j in oth])
                above += dz > 0.001
                under += dz < -0.001
        touch_frac = touch_n / n
        touching = touch_frac >= 0.05
        stacked = (touching and above / max(touch_n, 1) > 0.5) or (it.get('shape') or {}).get('on_item') is not None
        supports = touching and under / max(touch_n, 1) > 0.5
        rim_near = False
        if rim_tree is not None:
            rim_near = bool((rim_tree.query(pts, distance_upper_bound=0.008)[0] < 0.008).mean() >= 0.02)

        # 2D: per-frame SAM presence for this item (full cloud)
        gfull = np.nonzero(glab_full == g)[0]
        vis = pres = split2d = 0
        sc_list = []
        for fi, idx, rid, scr, edge in fr:
            mk = glab_full[idx] == g
            nv = int(mk.sum())
            if nv < MIN_VIS:
                continue
            vis += 1
            r = rid[mk]
            food = (r > 0) & (r < BIG)
            if food.mean() >= 0.5:
                pres += 1
                sc_list.append(float(scr[mk][food].mean()))
            ur, uc = np.unique(r[food], return_counts=True)
            split2d += int((uc / nv > FRAC).sum() >= 2)
        rows.append(dict(
            sid=sid, item=g, name=it["name"], category=it["category"], kind=item_kind(it), source=it["source"],
            volume_ml=float(it["volume_ml"]), size_bin=size_bin(it["volume_ml"]), n_points=n,
            n_points_full=int(len(gfull)), occluded_fraction=it.get("occluded_fraction"),
            iou=float(iou), error_type=etype, frag=flags["fragmented"], merged=flags["merged"],
            n_ids_gt20=int(sum(f > FRAC for f in id_fracs.values())), food_cov=food_cov,
            matched_id=int(p_match), matched_share=float(id_fracs.get(int(p_match), 0.0)),
            frames_visible=vis, frac_frames_sam_mask=pres / max(vis, 1), frac_frames_split2d=split2d / max(vis, 1),
            mean_sam_score=float(np.mean(sc_list)) if sc_list else float("nan"),
            touches=bool(touching), touch_frac=float(touch_frac), stacked=bool(stacked), supports=bool(supports),
            rim_near=rim_near, touches_items=sorted(touch_with)))

        # boundary-distance analysis (matched id only; unmatched stays IoU 0)
        if g in m:
            pm = pi == p_match
            fn, fp = gm & ~pm, ~gm & pm
            d_fn = cKDTree(xyz[~gm]).query(xyz[fn])[0] if fn.any() and (~gm).any() else np.zeros(0)
            d_fp = cKDTree(xyz[gm]).query(xyz[fp])[0] if fp.any() else np.zeros(0)
            boundary.append(dict(item=g, etype=etype, i=int((gm & pm).sum()), g=n, p=int(pm.sum()),
                                 d_fn=d_fn.tolist(), d_fp=d_fp.tolist()))
        else:
            boundary.append(dict(item=g, etype=etype, i=0, g=n, p=0, d_fn=[], d_fp=[]))
    for r, b in zip(rows, boundary):
        r["_b"] = b

    # ---- fragments (items flagged fragmented): per-frame 2D origin + seam properties
    by_item = {r["item"]: r for r in rows}
    for g in g_ids:
        r = by_item[g]
        if not r["frag"]:
            continue
        gm = gi == g
        ids = [i for i in np.unique(pi[gm]) if i > 0 and (pi[gm] == i).mean() > FRAC]
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = int(ids[a_i]), int(ids[b_i])
                A = (glab_full == g) & (lift_full == a)
                B = (glab_full == g) & (lift_full == b)
                co = split = same = miss = 0
                for fi, idx, rid, scr, edge in fr:
                    ma, mb = A[idx], B[idx]
                    if ma.sum() < 20 or mb.sum() < 20:
                        continue
                    co += 1
                    ra, rb = rid[ma], rid[mb]

                    def dom(x):
                        f = x[(x > 0) & (x < BIG)]
                        if len(f) < 0.5 * len(x):
                            return -1
                        u, c = np.unique(f, return_counts=True)
                        return int(u[c.argmax()])
                    da, db = dom(ra), dom(rb)
                    if da < 0 or db < 0:
                        miss += 1
                    elif da == db:
                        same += 1
                    else:
                        split += 1
                # seam in the voxel scene
                sa, sb = np.nonzero(gm & (pi == a))[0], np.nonzero(gm & (pi == b))[0]
                ta, tb = cKDTree(xyz[sa]), cKDTree(xyz[sb])
                pr = ta.query_ball_tree(tb, CONTACT)
                pa = np.array([sa[i] for i, l in enumerate(pr) for _ in l], int)
                pb = np.array([sb[j] for l in pr for j in l], int)
                seam = dict(n_pairs=int(len(pa)))
                if len(pa) >= 10:
                    rng = np.random.default_rng(0)
                    k = rng.permutation(len(pa))[:4000]
                    pa, pb = pa[k], pb[k]
                    # control: neighbour pairs inside the same fragment
                    ctl = []
                    for s_, t_ in ((sa, ta), (sb, tb)):
                        pr2 = t_.query_pairs(CONTACT, output_type="ndarray")
                        if len(pr2):
                            pr2 = pr2[rng.permutation(len(pr2))[:2000]]
                            ctl.append(np.stack([s_[pr2[:, 0]], s_[pr2[:, 1]]], 1))
                    ctl = np.concatenate(ctl) if ctl else np.zeros((0, 2), int)

                    def props(u, v):
                        dz = np.abs(xyz[u, 2] - xyz[v, 2])
                        ang = np.degrees(np.arccos(np.clip(np.abs((nrm[u] * nrm[v]).sum(1)), 0, 1)))
                        col = np.abs(rgb[vfull[u]] - rgb[vfull[v]]).mean(1)
                        return dz, ang, col
                    sdz, sang, scol = props(pa, pb)
                    cdz, cang, ccol = props(ctl[:, 0], ctl[:, 1]) if len(ctl) else (np.zeros(1),) * 3
                    thr_c = float(np.percentile(ccol, 90))

                    def raw_split(u, v):
                        fu, fv_ = vfull[u], vfull[v]
                        tot = np.zeros(len(u))
                        dif = np.zeros(len(u))
                        edge_hit = np.zeros(len(u))
                        for fi, idx, rid, scr, edge in fr:
                            ar = np.full(len(glab_full), -1, np.int64)
                            ae = np.full(len(glab_full), 1e3, np.float32)
                            ar[idx], ae[idx] = rid, edge
                            ok = (ar[fu] >= 0) & (ar[fv_] >= 0)
                            tot += ok
                            dif += ok & (ar[fu] != ar[fv_])
                            edge_hit += ok & (np.minimum(ae[fu], ae[fv_]) <= 3)
                        t = np.maximum(tot, 1)
                        return float((dif / t).mean()), float((edge_hit / t).mean())
                    s_split, s_edge = raw_split(pa, pb)
                    c_split, c_edge = raw_split(ctl[:, 0], ctl[:, 1]) if len(ctl) else (float("nan"),) * 2
                    seam.update(dz_mm_seam=float(sdz.mean() * 1e3), dz_mm_ctl=float(cdz.mean() * 1e3),
                                dz_gt3_seam=float((sdz > 0.003).mean()), dz_gt3_ctl=float((cdz > 0.003).mean()),
                                ang_seam=float(sang.mean()), ang_ctl=float(cang.mean()),
                                ang_gt30_seam=float((sang > 30).mean()), ang_gt30_ctl=float((cang > 30).mean()),
                                col_seam=float(scol.mean()), col_ctl=float(ccol.mean()),
                                col_gt_p90_seam=float((scol > thr_c).mean()),
                                rawsplit_seam=s_split, rawsplit_ctl=c_split, edge3px_seam=s_edge, edge3px_ctl=c_edge)
                frag_pairs.append(dict(sid=sid, item=g, a=a, b=b, share_a=float((lift_full[glab_full == g] == a).mean()),
                                       share_b=float((lift_full[glab_full == g] == b).mean()), co_visible=co,
                                       split_frac=split / max(co, 1), same_frac=same / max(co, 1),
                                       miss_frac=miss / max(co, 1), **seam))

    # ---- merges: an id > 20 % of item g that also holds > 20 % of another item h
    done = set()
    for g in g_ids:
        gm = gi == g
        for i in [i for i in np.unique(pi[gm]) if i > 0 and (pi[gm] == i).mean() > FRAC]:
            for h in g_ids:
                hm = gi == h
                if h <= g or (pi[hm] == i).mean() <= FRAC or (sid, g, h, int(i)) in done:
                    continue
                done.add((sid, g, h, int(i)))
                co = shared = 0
                co_i = shared_i = 0
                Gf = glab_full == g
                Hf = glab_full == h
                Gi, Hi = Gf & (lift_full == i), Hf & (lift_full == i)
                for fi, idx, rid, scr, edge in fr:
                    for (X, Y, c_, s_) in ((Gf, Hf, 0, 0), (Gi, Hi, 1, 1)):
                        mx, my = X[idx], Y[idx]
                        if mx.sum() < 20 or my.sum() < 20:
                            continue

                        def dom(x):
                            f = x[(x > 0) & (x < BIG)]
                            if len(f) < 0.5 * len(x):
                                return -1
                            u, c = np.unique(f, return_counts=True)
                            return int(u[c.argmax()])
                        dx, dy = dom(rid[mx]), dom(rid[my])
                        if dx < 0 or dy < 0:
                            continue
                        if c_ == 0:
                            co += 1
                            shared += dx == dy
                        else:
                            co_i += 1
                            shared_i += dx == dy
                merge_pairs.append(dict(
                    sid=sid, g=g, h=h, lift_id=int(i), cat_g=items[g]["category"], cat_h=items[h]["category"],
                    same_category=items[g]["category"] == items[h]["category"],
                    share_g=float((pi[gm] == i).mean()), share_h=float((pi[hm] == i).mean()),
                    touching=bool(h in by_item[g]["touches_items"]),
                    co_visible=co, one_mask_frac=shared / max(co, 1),
                    co_visible_idpts=co_i, one_mask_frac_idpts=shared_i / max(co_i, 1)))
    return dict(sid=sid, rows=rows, frag_pairs=frag_pairs, merge_pairs=merge_pairs)


# ------------------------------------------------------------------------------------------------ pooling / report
def boundary_summary(rows: list) -> dict:
    """Pooled per-point error distance to the true boundary + IoU bound if boundary-near errors were fixed."""
    out = {}
    d_all = np.concatenate([np.array(r["_b"]["d_fn"] + r["_b"]["d_fp"]) for r in rows] or [np.zeros(0)])
    out["n_error_points"] = int(len(d_all))
    for t in (0.003, 0.006, 0.012):
        out[f"frac_err_within_{int(t*1e3)}mm"] = float((d_all <= t).mean()) if len(d_all) else float("nan")
    base = np.array([r["iou"] for r in rows])
    for t in (0.003, 0.006, 0.012):
        new = []
        for r in rows:
            b = r["_b"]
            new.append(snap_iou(b["i"], b["g"], b["p"], np.array(b["d_fn"]), np.array(b["d_fp"]), t) if b["p"] > 0 else 0.0)
        new = np.array(new)
        out[f"miou_snap_{int(t*1e3)}mm"] = float(new.mean())
        out[f"gain_snap_{int(t*1e3)}mm"] = float(new.mean() - base.mean())
        out[f"by_type_gain_{int(t*1e3)}mm"] = {
            ty: float((new - base)[[r["error_type"] == ty for r in rows]].sum() / len(rows)) for ty in TYPES}
    out["miou"] = float(base.mean())
    out["by_type_within"] = {}
    for ty in TYPES:
        d = np.concatenate([np.array(r["_b"]["d_fn"] + r["_b"]["d_fp"]) for r in rows if r["error_type"] == ty]
                           or [np.zeros(0)])
        out["by_type_within"][ty] = dict(n=int(len(d)), w3=float((d <= 0.003).mean()) if len(d) else float("nan"),
                                         w6=float((d <= 0.006).mean()) if len(d) else float("nan"))
    return out


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def group_table(rows, key, title, order=None):
    keys = order or sorted({key(r) for r in rows})
    n_tot = len(rows)
    loss_tot = sum(1 - r["iou"] for r in rows)
    lines = [f"| {title} | n items | mIoU | " + " | ".join(TYPES) + " | share of mIoU loss |",
             "|---|---|---|" + "---|" * len(TYPES) + "---|"]
    for k in keys:
        rr = [r for r in rows if key(r) == k]
        if not rr:
            continue
        cnt = [sum(r["error_type"] == t for r in rr) for t in TYPES]
        lines.append(f"| {k} | {len(rr)} | {np.mean([r['iou'] for r in rr]):.3f} | " + " | ".join(map(str, cnt)) +
                     f" | {sum(1 - r['iou'] for r in rr) / loss_tot:.2f} |")
    return "\n".join(lines)


def build_report(res: list, bsum: dict, rows: list, fp: list, mp: list) -> str:
    L = []
    N = len(rows)
    miou = np.mean([r["iou"] for r in rows])
    loss_tot = sum(1 - r["iou"] for r in rows)
    reading = (OUT / "reading.md")
    if reading.exists():
        L.append(reading.read_text().rstrip() + "\n")
    nsc = len({r["sid"] for r in rows})
    L.append(f"# Lift error analysis ({N} GT items, {nsc} real-vote scans, co-vote + cos+qual replay)\n")
    L.append(f"Pooled instance mIoU {miou:.3f} (loss 1 - mIoU = {1 - miou:.3f}). Code `pipeline/eval/lift_errors.py`; "
             "per-item table `items.csv`, everything raw `results.json`. Error type rules in the module docstring "
             "(ok = IoU >= .75; fragmented = >= 2 lift ids each > 20 % of the item; merged = the id holding > 20 % of "
             "the item also holds > 20 % of another item). Touch = >= 5 % of the item's voxels have another item's "
             "voxel within 4 mm; stacked = touching and > 50 % of contact voxels sit > 1 mm above the neighbour (or a topping placed on an item); rim = >= 2 % "
             "of voxels within 8 mm of the container's top 15 % height band.\n")
    # 1 tables
    L.append("## 1. Error types\n")
    L.append("| error type | n items | share | mean IoU | mIoU loss contributed (sum(1-IoU)/N) | share of total loss |")
    L.append("|---|---|---|---|---|---|")
    for t in TYPES:
        rr = [r for r in rows if r["error_type"] == t]
        if not rr:
            L.append(f"| {t} | 0 | 0 | - | 0 | 0 |")
            continue
        ls = sum(1 - r["iou"] for r in rr)
        L.append(f"| {t} | {len(rr)} | {len(rr) / N:.2f} | {np.mean([r['iou'] for r in rr]):.3f} | {ls / N:.3f} | "
                 f"{ls / loss_tot:.2f} |")
    L.append("")
    batches = sorted({batch_of(r["sid"]) for r in rows})
    if len(batches) > 1:
        L.append("### Error types per batch\n")
        L.append("| batch | scans | items | inst mIoU | " + " | ".join(f"{t} n (loss share)" for t in TYPES) + " |")
        L.append("|---|---|---|---|" + "---|" * len(TYPES))
        for b in batches:
            rb = [r for r in rows if batch_of(r["sid"]) == b]
            lt = sum(1 - r["iou"] for r in rb)
            cells = []
            for t in TYPES:
                rr = [r for r in rb if r["error_type"] == t]
                cells.append(f"{len(rr)} ({sum(1 - r['iou'] for r in rr) / lt:.2f})")
            L.append(f"| {b} | {len({r['sid'] for r in rb})} | {len(rb)} | {np.mean([r['iou'] for r in rb]):.3f} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("Secondary flags (not exclusive): fragmented "
             f"{sum(r['frag'] for r in rows)}, merged {sum(r['merged'] for r in rows)}, both "
             f"{sum(r['frag'] and r['merged'] for r in rows)}; items with IoU >= .75 that are still fragmented "
             f"{sum(r['frag'] and r['error_type'] == 'ok' for r in rows)}.\n")
    L.append("### By kind\n")
    L.append(group_table(rows, lambda r: r["kind"], "kind"))
    L.append("\n### By size (item volume)\n")
    L.append(group_table(rows, lambda r: r["size_bin"], "size", ["<15 ml", "15-50 ml", "50-150 ml", ">=150 ml"]))
    L.append("\n### By touching / stacked / rim\n")
    L.append(group_table(rows, lambda r: ("stacked" if r["stacked"] else "touching" if r["touches"] else "isolated"),
                         "contact", ["isolated", "touching", "stacked"]))
    L.append("")
    L.append(group_table(rows, lambda r: "rim nearby" if r["rim_near"] else "no rim", "container rim"))
    L.append("")
    L.append(group_table(rows, lambda r: "same category as a neighbour" if any(
        by_cat(rows, r)) else "no same-category neighbour", "same-category neighbour"))
    # sam stats
    L.append("\n### SAM presence per error type\n")
    L.append("| error type | frames with a SAM food mask over >= 50 % of the item | frames where SAM splits the item "
             "(>= 2 raw masks each > 20 %) | mean SAM score |")
    L.append("|---|---|---|---|")
    for t in TYPES:
        rr = [r for r in rows if r["error_type"] == t]
        if rr:
            L.append(f"| {t} | {np.mean([r['frac_frames_sam_mask'] for r in rr]):.2f} | "
                     f"{np.mean([r['frac_frames_split2d'] for r in rr]):.2f} | "
                     f"{np.nanmean([r['mean_sam_score'] for r in rr]):.2f} |")
    # 2 fragments
    L.append("\n## 2. Fragmented items: where do the fragments come from\n")
    L.append(f"{len(fp)} fragment pairs in {len({(p['sid'], p['item']) for p in fp})} items. Per pair, over frames where "
             "both fragments (> 20 % of the item each) have >= 20 visible points: SAM `split` = the dominant raw mask "
             "under each fragment differs (2D split), `same` = one raw mask covers both, `miss` = one fragment has "
             "no food mask.\n")
    L.append("| sid | item | category | share a/b | co-visible frames | split | same mask | missing | seam pairs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    cat = {(r["sid"], r["item"]): r["category"] for r in rows}
    for p in fp:
        L.append(f"| {p['sid']} | {p['item']} | {cat[(p['sid'], p['item'])]} | {p['share_a']:.2f}/{p['share_b']:.2f} | "
                 f"{p['co_visible']} | {p['split_frac']:.2f} | {p['same_frac']:.2f} | {p['miss_frac']:.2f} | "
                 f"{p['n_pairs']} |")
    if fp:
        two_d = sum(p["split_frac"] > 0.5 for p in fp)
        one_m = sum(p["same_frac"] > 0.5 for p in fp)
        mixed = len(fp) - two_d - one_m
        L.append(f"\nPairs split as separate masks in > 50 % of co-visible frames: **{two_d} / {len(fp)}**; one SAM mask "
                 f"over both in > 50 %: **{one_m} / {len(fp)}**; neither (split in some views only, i.e. cross-view "
                 f"inconsistency / linking, or a fragment lacks a mask): **{mixed} / {len(fp)}**. Mean over pairs: "
                 f"split {np.mean([p['split_frac'] for p in fp]):.2f}, same {np.mean([p['same_frac'] for p in fp]):.2f}, "
                 f"missing {np.mean([p['miss_frac'] for p in fp]):.2f}.\n")
        sp = [p for p in fp if "dz_mm_seam" in p]
        L.append(f"### Seam properties ({len(sp)} pairs with >= 10 seam voxel pairs within 4 mm), seam vs control "
                 "(neighbour pairs inside one fragment)\n")
        L.append("| quantity | seam | control |")
        L.append("|---|---|---|")
        for lbl, k1, k2 in (("raw-mask id differs between the two sides (mean fraction of frames)", "rawsplit_seam", "rawsplit_ctl"),
                            ("seam within 3 px of a raw-mask edge", "edge3px_seam", "edge3px_ctl"),
                            ("height step |dz| (mm)", "dz_mm_seam", "dz_mm_ctl"),
                            ("fraction with |dz| > 3 mm", "dz_gt3_seam", "dz_gt3_ctl"),
                            ("normal angle (deg)", "ang_seam", "ang_ctl"),
                            ("fraction with normal angle > 30 deg", "ang_gt30_seam", "ang_gt30_ctl"),
                            ("colour difference (mean |dRGB|, 0-255)", "col_seam", "col_ctl")):
            L.append(f"| {lbl} | {np.nanmean([p[k1] for p in sp]):.3f} | {np.nanmean([p[k2] for p in sp]):.3f} |")
        L.append(f"| fraction of seam pairs with colour difference above the control's 90th percentile | "
                 f"{np.nanmean([p['col_gt_p90_seam'] for p in sp]):.2f} | 0.10 |")
    # 3 merges
    L.append("\n## 3. Merged items: 2D or 3D?\n")
    L.append("For each (lift id holding > 20 % of both items g and h): over frames where both items have >= 20 visible "
             "points and a food mask, the fraction in which ONE raw SAM mask is dominant under both (2D merge). "
             "`id pts` restricts to the points that carry the merged lift id.\n")
    L.append("| sid | g (category) | h (category) | share of g / h | touching | co-visible | one SAM mask | co-vis (id pts) | one mask (id pts) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in mp:
        L.append(f"| {p['sid']} | {p['g']} ({p['cat_g']}) | {p['h']} ({p['cat_h']}) | {p['share_g']:.2f}/{p['share_h']:.2f} | "
                 f"{'yes' if p['touching'] else 'no'} | {p['co_visible']} | {p['one_mask_frac']:.2f} | "
                 f"{p['co_visible_idpts']} | {p['one_mask_frac_idpts']:.2f} |")
    if mp:
        d2 = sum(p["one_mask_frac"] > 0.5 for p in mp)
        d2i = sum(p["one_mask_frac_idpts"] > 0.5 for p in mp)
        L.append(f"\nMerge pairs where one SAM mask spans both items in > 50 % of co-visible frames (2D), whole items: **{d2} / {len(mp)}**; "
                 f"restricted to the points carrying the merged id: **{d2i} / {len(mp)}** (mean {np.mean([p['one_mask_frac_idpts'] for p in mp]):.2f}); "
                 f"mean one-mask fraction {np.mean([p['one_mask_frac'] for p in mp]):.2f}; same category "
                 f"{sum(p['same_category'] for p in mp)} / {len(mp)}; touching {sum(p['touching'] for p in mp)} / {len(mp)}.\n")
    # 4 boundary
    b = bsum
    L.append("\n## 4. Boundary errors\n")
    L.append(f"Per-point errors of the matched lift id (false negatives + false positives, {b['n_error_points']} voxels, "
             "2 mm grid; distance to the item's true boundary):\n")
    L.append(f"- within 3 mm: **{b['frac_err_within_3mm']:.2f}**, within 6 mm: **{b['frac_err_within_6mm']:.2f}**, within 12 mm: "
             f"{b['frac_err_within_12mm']:.2f} of all error voxels.")
    L.append("\n| error type | error voxels | within 3 mm | within 6 mm |")
    L.append("|---|---|---|---|")
    for t in TYPES:
        w = b["by_type_within"][t]
        L.append(f"| {t} | {w['n']} | {fmt(w['w3'], 2)} | {fmt(w['w6'], 2)} |")
    L.append(f"\nIf every misassigned voxel within the band were snapped to GT (matching and ids unchanged): instance mIoU "
             f"{b['miou']:.3f} -> **{b['miou_snap_3mm']:.3f}** (3 mm, +{b['gain_snap_3mm']:.3f}), "
             f"**{b['miou_snap_6mm']:.3f}** (6 mm, +{b['gain_snap_6mm']:.3f}), {b['miou_snap_12mm']:.3f} (12 mm, "
             f"+{b['gain_snap_12mm']:.3f}). Gain by the item's primary error type (6 mm band, mIoU points):")
    L.append("\n| error type | gain at 3 mm | gain at 6 mm | gain at 12 mm |")
    L.append("|---|---|---|---|")
    for t in TYPES:
        L.append(f"| {t} | {b['by_type_gain_3mm'][t]:.3f} | {b['by_type_gain_6mm'][t]:.3f} | {b['by_type_gain_12mm'][t]:.3f} |")
    L.append("")
    return "\n".join(L)


def by_cat(rows, r):
    return [o["category"] == r["category"] for o in rows
            if o["sid"] == r["sid"] and o["item"] != r["item"] and o["item"] in r["touches_items"]]


def write_outputs(res: list, out: Path | None = None) -> None:
    global OUT
    OUT = out or OUT
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for s in res for r in s["rows"]]
    fp = [p for s in res for p in s["frag_pairs"]]
    mp = [p for s in res for p in s["merge_pairs"]]
    bsum = boundary_summary(rows)
    cols = ["sid", "item", "name", "category", "kind", "source", "volume_ml", "size_bin", "n_points", "n_points_full",
            "iou", "error_type", "frag", "merged", "n_ids_gt20", "food_cov", "matched_id", "matched_share",
            "frames_visible", "frac_frames_sam_mask", "frac_frames_split2d", "mean_sam_score", "touches", "touch_frac",
            "stacked", "supports", "rim_near", "occluded_fraction"]
    with open(OUT / "items.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([round(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols])
    clean = [{k: v for k, v in r.items() if k != "_b"} for r in rows]
    (OUT / "results.json").write_text(json.dumps(dict(items=clean, frag_pairs=fp, merge_pairs=mp, boundary=bsum),
                                                 indent=1, default=str))
    (OUT / "summary.md").write_text(build_report(res, bsum, rows, fp, mp))


def _run(sid):
    return analyse(sid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--all40", action="store_true", help="the 40 real-vote scans (hard_v12 + hard_v13) from results/lift_replay40 "
                    "-> results/lift_errors/scans40")
    a = ap.parse_args()
    global _use40
    if a.all40:
        from foodscan.boundary_refine import sids40
        _use40, sids, out = True, a.only or sids40(), OUT / "scans40"
    else:
        from foodscan.replay_lift import SIDS
        sids, out = a.only or SIDS, None
    with Pool(min(a.procs, len(sids))) as p:
        res = p.map(_run, sids)
    write_outputs(res, out)
    print("wrote", out or OUT)


if __name__ == "__main__":
    main()
