"""Lift per-frame 2D instance masks onto a fused 3D point cloud by voting.

The tracker (SAM 3 video mode) is expected to give temporally consistent
instance IDs, so a label map per frame is enough: for every frame, project all
3D points into the image, keep the ones that are actually visible (their depth
agrees with the frame's depth map), and add a vote for the label under each
projected pixel. The final label is the vote winner; the winner's share of the
votes is a per-point confidence you can threshold or report.

    votes = lift_masks(scan, masks, points)         # (N, L) float
    labels, margin = decide(votes)
    per_object_points = split_by_label(points, labels)

`masks[frame_index]` is an (H, W) integer array at either depth or RGB
resolution; 0 = unlabelled/background, 1..L = instance IDs. Missing frames are
skipped, so you can run the segmenter on a stride.

Lift channel (shared with the 3D refiner, pipeline/seg3d). `lift_scan` returns a
`LiftResult` and `LiftResult.save()` writes `<scan>/lift.npz` with, per fused point:

    lift_label   int32   scene-local instance id; 0 = background or no votes;
                         253 = clutter, 254 = container (kept, never merged)
    lift_share   float32 fraction of the point's (cos-weighted) votes won by lift_label
    lift_nvotes  int16   number of frames that voted (visible + unoccluded)
    lift_agree   uint8   1 if two disjoint frame halves (even/odd mask frames) both
                         voted and chose the same label, else 0

Cross-view ID repair follows the view-consensus idea of MaskClustering (Yan et al.,
CVPR 2024, arXiv 2401.07745): per-frame mask ids are nodes, nodes from different frames
are linked when they cover the same 3D points, and linked ids are merged so a tracker
ID break does not split one object into two instances.
"""
from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .loader import Frame, Scan


def project(points: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> pixel (u, v) and camera depth z, OpenCV convention."""
    pc = (T_wc[:3, :3] @ points.T).T + T_wc[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return u, v, z


# Depth-gate tolerance multiplier by ARKit LiDAR confidence (0 low, 1 medium, 2 high).
CONF_TOL_SCALE = np.array([2.0, 1.5, 1.0])
CLUTTER, CONTAINER = 253, 254


def visible_indices(frame: Frame, points: np.ndarray, mask_shape: tuple[int, int],
                    depth_tol: float = 0.01, min_conf: int = 2,
                    normals: np.ndarray | None = None, max_grazing_deg: float = 75.0,
                    conf_scale: bool = True):
    """Points that project inside the image and are not occluded in this frame.

    Returns (indices, u_int, v_int, weight). Occlusion test compares the
    point's camera depth with the LiDAR depth at that pixel; the depth map is
    always 256x192, so pixel coords are computed at that resolution and then
    rescaled to the mask resolution. With `conf_scale` and a confidence map the
    tolerance is depth_tol * CONF_TOL_SCALE[conf] (only matters when min_conf < 2).
    """
    dw, dh = frame.depth_size
    u, v, z = project(points, frame.K_depth, frame.T_wc)
    ok = np.isfinite(u) & np.isfinite(v)
    ui = np.floor(np.where(ok, u, -1)).astype(int)
    vi = np.floor(np.where(ok, v, -1)).astype(int)
    inb = ok & (z > 0.05) & (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
    idx = np.nonzero(inb)[0]
    uf, vf = u[idx], v[idx]            # sub-pixel depth coords, kept for the mask lookup
    ui, vi, z = ui[idx], vi[idx], z[idx]

    depth = frame.masked_depth(min_conf=min_conf)
    d = depth[vi, ui]
    tol = depth_tol
    if conf_scale:
        c = frame.confidence()
        if c is not None:
            tol = depth_tol * CONF_TOL_SCALE[np.clip(c[vi, ui], 0, 2)]
    vis = (d > 0) & (np.abs(d - z) < tol)
    idx, ui, vi, z, uf, vf = idx[vis], ui[vis], vi[vis], z[vis], uf[vis], vf[vis]

    # weight: cosine of viewing angle if normals are given, else 1
    weight = np.ones(len(idx))
    if normals is not None and len(idx):
        cam_pos = frame.T_cw[:3, 3]
        to_cam = cam_pos - points[idx]
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
        cos = np.abs(np.sum(normals[idx] * to_cam, axis=1))
        keep = cos > np.cos(np.deg2rad(max_grazing_deg))
        idx, uf, vf, cos = idx[keep], uf[keep], vf[keep], cos[keep]
        weight = cos

    # Mask lookup uses the sub-pixel position (K_depth has a -0.5 px shift: pixel centres
    # are integers), not the depth cell, so boundaries are not quantised to 7.5 px.
    mh, mw = mask_shape
    mu = np.clip(np.floor((uf + 0.5) * mw / dw).astype(int), 0, mw - 1)
    mv = np.clip(np.floor((vf + 0.5) * mh / dh).astype(int), 0, mh - 1)
    return idx, mu, mv, weight


def lift_masks(scan: Scan, masks: Mapping[int, np.ndarray], points: np.ndarray,
               n_labels: int | None = None, normals: np.ndarray | None = None,
               depth_tol: float = 0.01, count_background: bool = True,
               full: bool = False, **lift_kwargs):
    """Accumulate per-point label votes over all frames that have a mask.

    With `full=True` this is `lift_scan` and returns a `LiftResult` (lift_label,
    lift_share, lift_nvotes, lift_agree + the legacy hard decision); extra kwargs go
    to `lift_scan`. Default keeps the old (N, L+1) vote-matrix return.

    Returns votes of shape (N, L+1); column 0 is background. With
    `count_background` a visible point under an unlabelled pixel votes for
    background, which lets the plate/table win where the segmenter said nothing.
    """
    if full:
        return lift_scan(scan, masks, points, normals=normals, depth_tol=depth_tol, **lift_kwargs)
    by_index = {f.index: f for f in scan.frames}
    if n_labels is None:
        n_labels = max(int(m.max()) for m in masks.values())
    votes = np.zeros((len(points), n_labels + 1), dtype=np.float32)

    for fi, mask in masks.items():
        frame = by_index.get(fi)
        if frame is None:
            continue
        idx, mu, mv, w = visible_indices(frame, points, mask.shape, depth_tol=depth_tol, normals=normals)
        if len(idx) == 0:
            continue
        lab = mask[mv, mu].astype(int)
        if not count_background:
            keep = lab > 0
            idx, lab, w = idx[keep], lab[keep], w[keep]
        np.add.at(votes, (idx, lab), w)
    return votes


# ---------------------------------------------------------------------------
# Lift channel
# ---------------------------------------------------------------------------

@dataclass
class _FrameVotes:
    pos: int                 # position in sorted mask-frame order (for halves)
    idx: np.ndarray          # int32 point indices that are visible in this frame
    lab: np.ndarray          # raw mask value under each (int32)
    w: np.ndarray            # float32 vote weight (cosine of the viewing angle)
    score: np.ndarray | None = None   # SAM score of the mask under the pixel (1 for unscored / background)
    edge: np.ndarray | None = None    # distance (mask px) from the pixel to the nearest label change in the frame
    qual: np.ndarray | None = None    # quality of the mask under the pixel (solidity x saturating area; 1 for background)


def frame_mask_features(mask: np.ndarray, frame_scores: Mapping[int, float] | None = None,
                        qual_area: float = 400.0):
    """Per-frame vote-weight ingredients: (edge distance map, score lookup, quality lookup).

    edge: (H, W) float32, pixels to the nearest 4-neighbour label change (image borders are not edges).
    score/qual: float32 lookup tables indexed by mask value; background (0) and unscored ids get 1.0.
    quality = solidity (area / convex-hull area of the mask's pixels) * min(1, area / qual_area).
    """
    from scipy import ndimage
    b = np.zeros(mask.shape, bool)
    b[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    b[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge = ndimage.distance_transform_edt(~b).astype(np.float32) if b.any() else np.full(mask.shape, 1e3, np.float32)
    top = int(mask.max()) + 1
    score = np.ones(top, np.float32)
    qual = np.ones(top, np.float32)
    for k, v in (frame_scores or {}).items():
        if 0 < int(k) < top:
            score[int(k)] = v
    try:
        import cv2
    except ImportError:                                    # pragma: no cover
        cv2 = None
    for lid in np.unique(mask):
        if lid <= 0:
            continue
        m = mask == lid
        area = float(m.sum())
        if cv2 is not None:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = np.concatenate(cnts, 0)
            hull = cv2.contourArea(cv2.convexHull(pts)) + 0.5 * len(cv2.convexHull(pts))   # pixel-area correction
        else:
            ys, xs = np.nonzero(m)
            hull = float((ys.ptp() + 1) * (xs.ptp() + 1))
        qual[int(lid)] = min(1.0, area / max(hull, 1.0)) * min(1.0, area / qual_area)
    return edge, score, qual


WEIGHT_TOKENS = ("cos", "score", "edge", "qual")


def vote_weights(fv: _FrameVotes, weighting: str = "cos", edge_px: float = 6.0, edge_floor: float = 0.1,
                 score_power: float = 1.0) -> np.ndarray:
    """Vote weight of one frame under a scheme such as "cos", "cos+score", "cos+score+edge", "cos+score+edge+qual".

    cos   cosine of the viewing angle (always applied, it is the base weight)
    score x SAM detection score ** score_power
    edge  x edge_floor + (1 - edge_floor) * min(1, d / edge_px), d = pixels to the nearest 2D mask edge
    qual  x per-mask quality (solidity x saturating area)
    """
    toks = set(weighting.split("+"))
    bad = toks - set(WEIGHT_TOKENS)
    if bad:
        raise ValueError(f"unknown weighting tokens {sorted(bad)}; use {WEIGHT_TOKENS} joined with '+'")
    w = fv.w.astype(np.float32)
    if "score" in toks:
        w = w * fv.score ** score_power
    if "edge" in toks:
        w = w * (edge_floor + (1.0 - edge_floor) * np.minimum(1.0, fv.edge / max(edge_px, 1e-6)))
    if "qual" in toks:
        w = w * fv.qual
    return w.astype(np.float32)


def collect_votes(scan: Scan, masks: Mapping[int, np.ndarray], points: np.ndarray,
                  normals: np.ndarray | None = None, depth_tol: float = 0.01,
                  min_conf: int = 1, max_grazing_deg: float = 75.0,
                  scores: Mapping[int, Mapping[int, float]] | None = None, features: bool = False,
                  qual_area: float = 400.0) -> list[_FrameVotes]:
    """One projection pass: per frame, which points are visible and which mask id they fall on.

    With `features=True` each vote also carries score / edge distance / mask quality (see
    `frame_mask_features`; `scores[frame][mask id]` = SAM score) so `vote_weights` can re-weight it."""
    by_index = {f.index: f for f in scan.frames}
    out = []
    for pos, fi in enumerate(sorted(masks)):
        frame = by_index.get(fi)
        if frame is None:
            continue
        mask = masks[fi]
        idx, mu, mv, w = visible_indices(frame, points, mask.shape, depth_tol=depth_tol,
                                         min_conf=min_conf, normals=normals,
                                         max_grazing_deg=max_grazing_deg)
        if len(idx):
            fv = _FrameVotes(pos, idx.astype(np.int32), mask[mv, mu].astype(np.int32), np.asarray(w, np.float32))
            if features:
                edge, slut, qlut = frame_mask_features(mask, (scores or {}).get(fi), qual_area)
                fv.edge, fv.score, fv.qual = edge[mv, mu], slut[mask[mv, mu]], qlut[mask[mv, mu]]
            out.append(fv)
    return out


def repair_ids(fvs: list[_FrameVotes], n_points: int, groups: Mapping[int, object] | None = None,
               overlap_thresh: float = 0.5, min_points: int = 20, min_support: int = 2,
               cooccur_max: float = 0.2, max_points: int = 40000, seed: int = 0,
               aux_ids: Collection[int] | None = None, mutual: bool = False,
               ) -> tuple[dict[int, int], dict]:
    """View-consensus merge of per-frame mask ids (MaskClustering-style, arXiv 2401.07745).

    Nodes are (frame, id) masks, represented by the set of 3D points that voted for them.
    Two nodes from different frames and different ids are linked when their point overlap
    exceeds `overlap_thresh` of the SMALLER node. Ids with >= `min_support` linked node
    pairs are merged (union-find, strongest links first), except:
      * only ids 1..252 take part (253 clutter / 254 container are never merged),
      * ids from different concepts (`groups[id]`) are never merged,
      * ids that co-occur in the same frame (cannot-link, as two real objects do) in more
        than `cooccur_max` of the frames of the rarer one are never merged.
    Overlap is measured on a random subset of <= max_points points for speed.
    `aux_ids` (default: every id >= CLUTTER) names the ids that never take part, for maps whose
    food ids exceed 253 (per-frame-unique raw ids). `mutual=True` additionally requires the two
    nodes to be each other's best-overlap partner in the other's frame (mutual-best-overlap rule).
    Returns ({id: representative_id} for merged ids only, stats).
    """
    from scipy import sparse
    is_aux = (lambda l: l in aux_ids) if aux_ids is not None else (lambda l: l >= CLUTTER)

    rng = np.random.default_rng(seed)
    sub = np.full(n_points, -1, np.int64)
    if n_points > max_points:
        pick = rng.choice(n_points, max_points, replace=False)
    else:
        pick = np.arange(n_points)
    sub[pick] = np.arange(len(pick))

    node_frame, node_id, rows, cols = [], [], [], []
    for fv in fvs:
        s = sub[fv.idx]
        for lid in np.unique(fv.lab):
            if lid <= 0 or is_aux(lid):
                continue
            sel = (fv.lab == lid) & (s >= 0)
            if sel.sum() < 1:
                continue
            k = len(node_id)
            node_frame.append(fv.pos)
            node_id.append(int(lid))
            rows.append(np.full(int(sel.sum()), k))
            cols.append(s[sel])
    stats = {"ids": len({*node_id}), "merges": 0, "merged": {}}
    if len(node_id) < 2:
        return {}, stats
    M = sparse.csr_matrix((np.ones(sum(len(r) for r in rows), np.float32),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(len(node_id), len(pick)))
    size = np.asarray(M.sum(1)).ravel()
    O = (M @ M.T).tocoo()
    nf, ni = np.array(node_frame), np.array(node_id)

    support: dict[tuple[int, int], list[float]] = {}
    cand = []
    for a, b, o in zip(O.row, O.col, O.data):
        if a >= b or nf[a] == nf[b] or ni[a] == ni[b]:
            continue
        sm = min(size[a], size[b])
        if sm < min_points or o / sm <= overlap_thresh:
            continue
        cand.append((int(a), int(b), float(o / sm)))
    if mutual:                                   # keep a pair only if each is the other's best partner in its frame
        best: dict[tuple[int, int], tuple[float, int]] = {}
        for a, b, f in cand:
            for x, y in ((a, b), (b, a)):
                k = (x, int(nf[y]))
                if k not in best or f > best[k][0]:
                    best[k] = (f, y)
        cand = [(a, b, f) for a, b, f in cand
                if best[(a, int(nf[b]))][1] == b and best[(b, int(nf[a]))][1] == a]
    for a, b, f in cand:
        key = (min(ni[a], ni[b]), max(ni[a], ni[b]))
        support.setdefault(key, []).append(f)

    # cannot-link from co-occurrence (full-resolution counts, not the subset)
    frames_of: dict[int, set[int]] = {}
    for fv in fvs:
        for lid in np.unique(fv.lab):
            if lid > 0 and not is_aux(lid):
                frames_of.setdefault(int(lid), set()).add(fv.pos)

    def cannot(a: int, b: int) -> bool:
        if groups is not None and groups.get(a) != groups.get(b):
            return True
        fa, fb = frames_of.get(a, set()), frames_of.get(b, set())
        return len(fa & fb) > cooccur_max * max(1, min(len(fa), len(fb)))

    parent = {i: i for i in frames_of}
    members = {i: {i} for i in frames_of}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), ovs in sorted(support.items(), key=lambda kv: -np.mean(kv[1])):
        if len(ovs) < min_support:
            continue
        ra, rb = find(a), find(b)
        if ra == rb or any(cannot(x, y) for x in members[ra] for y in members[rb]):
            continue
        keep, drop = min(ra, rb), max(ra, rb)
        parent[drop] = keep
        members[keep] |= members.pop(drop)
        stats["merges"] += 1
    remap = {i: find(i) for i in parent if find(i) != i}
    stats["merged"] = {int(k): int(v) for k, v in remap.items()}
    return remap, stats


def cluster_ids(fvs: list[_FrameVotes], n_points: int, thresh: float = 0.7, sim: str = "cosine",
                linkage: str = "average", min_points: int = 20, cooccur_max: float = 0.2,
                aux_ids: Collection[int] | None = None, max_points: int = 40000, seed: int = 0,
                groups: Mapping[int, object] | None = None, max_ids: int = 4000) -> tuple[dict[int, int], dict]:
    """Merge mask ids by agglomerative clustering of the instance x instance co-vote matrix.

    Each id is the set of 3D points that voted for it (any frame). Similarity of two ids:
    `cosine` = |A & B| / sqrt(|A||B|), `min` = |A & B| / min(|A|,|B|), `iou`. Distance = 1 - sim;
    ids that co-occur in the same frame (cannot-link, as in repair_ids) or belong to different
    concepts get distance 2. Clusters are cut at `thresh` (scipy `linkage`, default average), so a
    merge needs a high *mean* similarity to every member, not one strong pair. Returns the same
    ({id: representative}, stats) as repair_ids.
    """
    from scipy import sparse
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as hlink
    from scipy.spatial.distance import squareform

    rng = np.random.default_rng(seed)
    sub = np.full(n_points, -1, np.int64)
    pick = rng.choice(n_points, max_points, replace=False) if n_points > max_points else np.arange(n_points)
    sub[pick] = np.arange(len(pick))
    is_aux = (lambda l: l in aux_ids) if aux_ids is not None else (lambda l: l >= CLUTTER)
    frames_of: dict[int, set[int]] = {}
    rows, cols = [], []
    for fv in fvs:
        s = sub[fv.idx]
        for lid in np.unique(fv.lab):
            if lid <= 0 or is_aux(lid):
                continue
            frames_of.setdefault(int(lid), set()).add(fv.pos)
            sel = (fv.lab == lid) & (s >= 0)
            if sel.any():
                rows.append(np.full(int(sel.sum()), int(lid)))
                cols.append(s[sel])
    ids = sorted(frames_of)
    stats = {"ids": len(ids), "merges": 0, "merged": {}}
    if len(ids) < 2 or len(ids) > max_ids:
        return {}, stats
    col_of = {l: i for i, l in enumerate(ids)}
    r = np.array([col_of[x] for x in np.concatenate(rows)])
    M = sparse.csr_matrix((np.ones(len(r), np.float32), (r, np.concatenate(cols))), shape=(len(ids), len(pick)))
    M.data[:] = 1.0                                      # a point voting for an id in several frames counts once
    size = np.asarray(M.sum(1)).ravel()
    O = (M @ M.T).toarray()
    if sim == "cosine":
        S = O / np.sqrt(np.maximum(np.outer(size, size), 1e-9))
    elif sim == "min":
        S = O / np.maximum(np.minimum.outer(size, size), 1e-9)
    elif sim == "iou":
        S = O / np.maximum(np.add.outer(size, size) - O, 1e-9)
    else:
        raise ValueError(sim)
    D = 1.0 - S
    fsets = [frames_of[l] for l in ids]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            bad = size[i] < min_points or size[j] < min_points
            if not bad:
                bad = len(fsets[i] & fsets[j]) > cooccur_max * max(1, min(len(fsets[i]), len(fsets[j])))
            if not bad and groups is not None:
                bad = groups.get(ids[i]) != groups.get(ids[j])
            if bad:
                D[i, j] = D[j, i] = 2.0
    np.fill_diagonal(D, 0.0)
    lab = fcluster(hlink(squareform(np.clip(D, 0, 2), checks=False), method=linkage), t=thresh, criterion="distance")
    rep: dict[int, int] = {}
    for l, c in zip(ids, lab):
        rep.setdefault(int(c), l)                         # smallest id of the cluster (ids are sorted)
    remap = {l: rep[int(c)] for l, c in zip(ids, lab) if rep[int(c)] != l}
    stats["merges"] = len(remap)
    stats["merged"] = {int(k): int(v) for k, v in remap.items()}
    return remap, stats


def _accumulate(fvs, ids: np.ndarray, remap: dict[int, int], n_points: int, subset=None,
                weighting: str = "cos", weight_kwargs: dict | None = None):
    """Vote matrix (N, len(ids)) over compact columns `ids` (sorted raw values incl. 0) and frame counts."""
    top = int(max(ids.max(), max((fv.lab.max() for fv in fvs), default=0))) + 1
    lut = np.full(top, -1, np.int64)
    lut[ids] = np.arange(len(ids))
    for src, dst in remap.items():
        lut[src] = lut[dst]
    votes = np.zeros((n_points, len(ids)), np.float32)
    nv = np.zeros(n_points, np.int32)
    for fv in fvs:
        if subset is not None and not subset(fv.pos):
            continue
        w = fv.w if weighting == "cos" else vote_weights(fv, weighting, **(weight_kwargs or {}))
        votes[fv.idx, lut[fv.lab]] += w   # idx unique within a frame -> no add.at needed
        nv[fv.idx] += 1
    return votes, nv


def _decide_cols(votes: np.ndarray, nv: np.ndarray, ids: np.ndarray, min_nvotes: int,
                 margin: float = 0.0, bg_abstain: bool = False):
    """Argmax decision. `margin`: the winner must lead the runner-up by >= margin of the total vote
    (else label 0). `bg_abstain`: background (column 0) votes are dropped, so a point is background
    only when nothing but background voted for it."""
    if bg_abstain:
        votes = votes.copy()
        votes[:, 0] = 0.0
    tot = votes.sum(1)
    col = votes.argmax(1)
    top = votes.max(1)
    ok = (nv >= min_nvotes) & (tot > 0)
    if margin > 0 and votes.shape[1] > 1:
        second = np.partition(votes, -2, axis=1)[:, -2]
        ok &= (top - second) >= margin * np.maximum(tot, 1e-9)
    lab = np.where(ok, ids[col], 0).astype(np.int32)
    share = np.where(ok, top / np.maximum(tot, 1e-9), 0.0).astype(np.float32)
    return lab, share


def compute_superpoints(points: np.ndarray, normals: np.ndarray | None = None, rgb: np.ndarray | None = None,
                        n_target: int = 400, w_normal: float = 0.5, w_color: float = 0.0, color_scale: float = 20.0,
                        iters: int = 6, k_centres: int = 12, min_frac: float = 0.15, seed: int = 0) -> np.ndarray:
    """GT-free geometric over-segmentation (SLIC/VCCS-style, in 3D) -> (N,) int32 superpoint ids.

    Seeds by farthest-point sampling, then `iters` rounds of local k-means: every point joins the
    best of its `k_centres` nearest centres under d^2 = |dx|^2/S^2 + w_normal*|dn|^2
    + w_color*|dLab|^2/color_scale^2 (S = mean centre spacing, normals sign-aligned to the centre).
    Disconnected fragments (kNN graph, k=6) are split off and fragments smaller than
    `min_frac` * mean size are absorbed by the neighbouring superpoint they touch most.
    `rgb` (N,3, 0..1 or 0..255) is only used when w_color > 0.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64)
    n = len(pts)
    n_target = int(min(max(n_target, 1), n))
    rng = np.random.default_rng(seed)
    sel = np.empty(n_target, np.int64)                    # farthest-point sampling
    sel[0] = rng.integers(n)
    dmin = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for i in range(1, n_target):
        sel[i] = int(dmin.argmax())
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[sel[i]], axis=1))
    use_n = normals is not None and w_normal > 0
    nrm = np.asarray(normals, np.float64) if use_n else None
    use_c = rgb is not None and w_color > 0
    if use_c:
        import cv2
        c = np.asarray(rgb, np.float32)
        if c.max() > 1.5:
            c = c / 255.0
        lab = cv2.cvtColor(c.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(np.float64)
    cx = pts[sel].copy()
    cn = nrm[sel].copy() if use_n else None
    cc = lab[sel].copy() if use_c else None
    k = min(k_centres, n_target)
    S = max(float(cKDTree(cx).query(cx, k=min(2, n_target))[0][:, -1].mean()), 1e-6) if n_target > 1 else 1.0
    arange = np.arange(n)
    asg = np.zeros(n, np.int64)
    for _ in range(iters):
        d, ci = cKDTree(cx).query(pts, k=k)
        d, ci = (d[:, None], ci[:, None]) if k == 1 else (d, ci)
        cost = (d / S) ** 2
        if use_n:
            cnn = cn[ci]
            sg = np.sign(np.einsum("nd,nkd->nk", nrm, cnn))
            sg[sg == 0] = 1.0
            cost = cost + w_normal * ((nrm[:, None, :] * sg[..., None] - cnn) ** 2).sum(-1)
        if use_c:
            cost = cost + w_color * ((lab[:, None, :] - cc[ci]) ** 2).sum(-1) / color_scale ** 2
        asg = ci[arange, cost.argmin(1)]
        cnt = np.bincount(asg, minlength=n_target).astype(np.float64)
        has = cnt > 0
        for dim in range(3):
            cx[has, dim] = np.bincount(asg, pts[:, dim], n_target)[has] / cnt[has]
        if use_n:
            sgc = np.sign(np.einsum("nd,nd->n", nrm, cn[asg]))
            sgc[sgc == 0] = 1.0
            for dim in range(3):
                cn[has, dim] = np.bincount(asg, nrm[:, dim] * sgc, n_target)[has]
            cn[has] /= np.maximum(np.linalg.norm(cn[has], axis=1, keepdims=True), 1e-9)
        if use_c:
            for dim in range(3):
                cc[has, dim] = np.bincount(asg, lab[:, dim], n_target)[has] / cnt[has]
    # connectivity: split disconnected pieces, absorb tiny ones
    nn = cKDTree(pts).query(pts, k=min(7, n))[1][:, 1:]
    ii = np.repeat(arange, nn.shape[1])
    jj = nn.ravel()
    same = asg[ii] == asg[jj]
    G = sparse.coo_matrix((np.ones(int(same.sum())), (ii[same], jj[same])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    min_size = max(1, int(min_frac * n / n_target))
    for _ in range(4):
        size = np.bincount(comp)
        small = np.nonzero(size < min_size)[0]
        if len(small) == 0:
            break
        for cid in small:
            m = np.nonzero(comp == cid)[0]
            nb = comp[nn[m].ravel()]
            nb = nb[nb != cid]
            if len(nb):
                comp[m] = np.bincount(nb).argmax()
    _, out = np.unique(comp, return_inverse=True)
    return out.astype(np.int32)


def superpoint_majority(labels: np.ndarray, sp: np.ndarray, valid: np.ndarray | None = None,
                        weights: np.ndarray | None = None) -> np.ndarray:
    """Per-superpoint majority of the per-point `labels` (counted over `valid` points, optionally
    `weights`-weighted; label 0 competes like any class). Every point takes its superpoint's winner;
    a superpoint with no valid point keeps label 0."""
    lab = np.asarray(labels)
    uniq, col = np.unique(lab, return_inverse=True)
    n_sp = int(sp.max()) + 1
    w = np.ones(len(lab)) if weights is None else np.asarray(weights, np.float64)
    if valid is not None:
        w = np.where(valid, w, 0.0)
    tab = np.zeros((n_sp, len(uniq)))
    np.add.at(tab, (sp, col), w)
    win = uniq[tab.argmax(1)]
    return np.where(tab.sum(1) > 0, win, 0)[sp].astype(lab.dtype)


def superpoint_votes(votes: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """Soft variant: sum the (N, C) vote matrix over each superpoint and give it back to every member."""
    from scipy import sparse
    n_sp = int(sp.max()) + 1
    P = sparse.csr_matrix((np.ones(len(sp), np.float32), (sp, np.arange(len(sp)))), shape=(n_sp, len(sp)))
    return (P @ votes)[sp]


@dataclass
class LiftResult:
    lift_label: np.ndarray     # (N,) int32
    lift_share: np.ndarray     # (N,) float32
    lift_nvotes: np.ndarray    # (N,) int16
    lift_agree: np.ndarray     # (N,) uint8
    votes: np.ndarray          # (N, C) float32 compact vote matrix (columns = ids)
    ids: np.ndarray            # (C,) raw/merged mask ids per column; ids[0] == 0
    merge_map: dict = field(default_factory=dict)   # {merged_id: representative_id}
    stats: dict = field(default_factory=dict)

    @property
    def hard_label(self) -> np.ndarray:
        """Legacy `decide` convention: -1 unseen, else winner id (0 = background)."""
        return np.where(self.lift_nvotes > 0, self.lift_label, -1)

    def save(self, path, points: np.ndarray | None = None, normals: np.ndarray | None = None,
             meta: Mapping | None = None) -> Path:
        """Write lift.npz. Optional xyz/normals/meta are stored so the file is self-contained."""
        arrays = dict(lift_label=self.lift_label.astype(np.int32), lift_share=self.lift_share.astype(np.float32),
                      lift_nvotes=self.lift_nvotes.astype(np.int16), lift_agree=self.lift_agree.astype(np.uint8))
        if points is not None:
            arrays["xyz"] = np.asarray(points, np.float32)
        if normals is not None:
            arrays["normals"] = np.asarray(normals, np.float32)
        arrays["meta_json"] = np.array(json.dumps({
            "instances": {str(k): v for k, v in (meta or {}).items()},
            "clutter_label": CLUTTER, "container_label": CONTAINER,
            "merge_map": {str(k): v for k, v in self.merge_map.items()},
            "stats": self.stats}))
        path = Path(path)
        np.savez_compressed(path, **arrays)
        return path


def load_lift(path) -> dict:
    """Read lift.npz back: arrays by key, plus `meta` (parsed meta_json)."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return d


def estimate_normals(points: np.ndarray, radius: float = 0.01, max_nn: int = 30) -> np.ndarray:
    """PCA normals (sign arbitrary; the lift uses |cos|). Needs open3d."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals)


def lift_scan(scan: Scan, masks: Mapping[int, np.ndarray], points: np.ndarray,
              normals: np.ndarray | None = None, meta: Mapping[int, dict] | None = None,
              depth_tol: float = 0.01, min_conf: int = 1, max_grazing_deg: float = 75.0,
              min_nvotes: int = 2, repair: bool = True, repair_kwargs: dict | None = None,
              agree: bool = True, smooth_radius: float = 0.0,
              weighting: str = "cos", weight_kwargs: dict | None = None,
              scores: Mapping[int, Mapping[int, float]] | None = None,
              margin: float = 0.0, bg_abstain: bool = False,
              superpoints: np.ndarray | None = None, sp_mode: str = "hard") -> LiftResult:
    """Lift per-frame masks onto `points` with a visibility-gated, cos-weighted vote.

    Visibility: |z - lidar depth| < depth_tol * CONF_TOL_SCALE[conf] and grazing angle
    < max_grazing_deg (needs `normals`). `min_conf=1` admits medium-confidence LiDAR with a
    1.5x tolerance (the old pipeline dropped them). `repair` merges split ids across views
    (see repair_ids; same-concept only when `meta` has concept_index). `smooth_radius`
    (m, e.g. 0.004; default 0 = off) sums each point's votes with its neighbours' within
    that radius before the decision. `lift_agree` compares even vs odd mask frames.

    Optional quality flags (defaults reproduce the plain cos-weighted vote exactly; see
    results/lift_sweep/summary.md for which are worth switching on):
      weighting      "cos" | "cos+score" | "cos+score+edge" | "cos+score+edge+qual" (vote_weights;
                     score/edge/qual need `scores[frame][id]` = SAM score and cost one distance
                     transform per frame; `weight_kwargs`: edge_px, edge_floor, score_power)
      margin         winner must lead the runner-up by this fraction of the total vote
      bg_abstain     background votes are abstentions instead of a class
      superpoints    (N,) ids from `compute_superpoints`; every superpoint takes one label
                     (`sp_mode` "hard" = majority of the point labels, "soft" = argmax of the summed votes)
    """
    fvs = collect_votes(scan, masks, points, normals, depth_tol, min_conf, max_grazing_deg,
                        scores=scores, features=weighting != "cos")
    return lift_from_votes(fvs, len(points), points, meta=meta, min_nvotes=min_nvotes, repair=repair,
                           repair_kwargs=repair_kwargs, agree=agree, smooth_radius=smooth_radius,
                           weighting=weighting, weight_kwargs=weight_kwargs, margin=margin,
                           bg_abstain=bg_abstain, superpoints=superpoints, sp_mode=sp_mode)


DEFAULT_LINKED_WEIGHTING = "cos+qual"     # results/lift_sweep: +0.03 inst mIoU on top of co-vote linking


def lift_for_meal(scan: Scan, masks: Mapping[int, np.ndarray], points: np.ndarray,
                  normals: np.ndarray | None = None, meta: Mapping[int, dict] | None = None,
                  weighting: str = "cos", repair: bool = True, smooth_radius: float = 0.0,
                  scores: Mapping[int, Mapping[int, float]] | None = None, save_to=None, **kw) -> LiftResult:
    """The one lift call used by run_meal and by segment's replay/CLI: lift_scan (repair pass, min_nvotes 2)
    on already-linked masks, optionally written to `save_to` (lift.npz with xyz/normals/instances)."""
    lift = lift_scan(scan, masks, points, normals=normals, meta=meta, repair=repair, smooth_radius=smooth_radius,
                     weighting=weighting, scores=scores, **kw)
    if save_to is not None:
        lift.save(save_to, points=points, normals=normals, meta=meta)
    return lift


def lift_from_votes(fvs: list[_FrameVotes], n: int, points: np.ndarray | None = None,
                    meta: Mapping[int, dict] | None = None, min_nvotes: int = 2, repair: bool = True,
                    repair_kwargs: dict | None = None, agree: bool = True, smooth_radius: float = 0.0,
                    weighting: str = "cos", weight_kwargs: dict | None = None, margin: float = 0.0,
                    bg_abstain: bool = False, superpoints: np.ndarray | None = None,
                    sp_mode: str = "hard", ids_extra: Collection[int] = (),
                    remap: dict[int, int] | None = None) -> LiftResult:
    """Everything in `lift_scan` after the projection pass, so callers can re-decide cheaply
    from one `collect_votes` result (e.g. relabelled by a different instance-linking rule)."""
    raw_ids = {0, *map(int, ids_extra)}
    for fv in fvs:
        raw_ids.update(np.unique(fv.lab).tolist())
    rstats = {"ids": 0, "merges": 0, "merged": {}}
    if remap is not None:                       # precomputed merge map (sweeps reuse one repair_ids call)
        repair = False
    else:
        remap = {}
    if repair and fvs:
        groups = {int(k): v.get("concept_index") for k, v in meta.items()} if meta else None
        remap, rstats = repair_ids(fvs, n, groups=groups, **(repair_kwargs or {}))
    ids = np.array(sorted(raw_ids - set(remap)), np.int64)

    votes, nv = _accumulate(fvs, ids, remap, n, weighting=weighting, weight_kwargs=weight_kwargs)
    nv16 = np.minimum(nv, np.iinfo(np.int16).max).astype(np.int16)
    dv = votes
    if smooth_radius > 0:
        from scipy import sparse
        from scipy.spatial import cKDTree
        pairs = cKDTree(points).query_pairs(smooth_radius, output_type="ndarray")
        A = sparse.coo_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        A = (A + A.T + sparse.identity(n, dtype=np.float32)).tocsr()
        dv = A @ votes
    if superpoints is not None and sp_mode == "soft":
        dv = superpoint_votes(dv, superpoints)
        sp_nv = np.zeros(int(superpoints.max()) + 1, np.int32)
        np.maximum.at(sp_nv, superpoints, nv)          # a superpoint counts as seen if any member was
        nv_dec = sp_nv[superpoints]
        lab, share = _decide_cols(dv, nv_dec, ids, min_nvotes, margin, bg_abstain)
    else:
        lab, share = _decide_cols(dv, nv, ids, min_nvotes, margin, bg_abstain)
        if superpoints is not None:
            lab = superpoint_majority(lab, superpoints, valid=nv >= min_nvotes)
    lab = np.where((nv > 0) | (superpoints is not None), lab, 0).astype(np.int32)

    agr = np.zeros(n, np.uint8)
    if agree and len(fvs) >= 2:
        va, na = _accumulate(fvs, ids, remap, n, subset=lambda p: p % 2 == 0)
        vb, nb = _accumulate(fvs, ids, remap, n, subset=lambda p: p % 2 == 1)
        la, _ = _decide_cols(va, na, ids, 1)
        lb, _ = _decide_cols(vb, nb, ids, 1)
        agr = ((na > 0) & (nb > 0) & (la == lb)).astype(np.uint8)
    return LiftResult(lab, share, nv16, agr, votes, ids, remap, {"repair": rstats, "n_frames": len(fvs)})


def decide(votes: np.ndarray, min_votes: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Winner label per point (-1 if unseen) and winner's vote share in [0, 1]."""
    total = votes.sum(1)
    labels = votes.argmax(1)
    margin = np.where(total > 0, votes.max(1) / np.maximum(total, 1e-9), 0.0)
    labels = np.where(total >= min_votes, labels, -1)
    return labels, margin


def split_by_label(points: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(k): points[labels == k] for k in np.unique(labels) if k > 0}


def label_colors(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random RGB per label for visualisation; background grey, unseen black."""
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1 if labels.size else 1
    palette = rng.uniform(0.2, 1.0, size=(max(n, 1), 3))
    palette[0] = 0.6
    colors = palette[np.clip(labels, 0, None)]
    colors[labels < 0] = 0.0
    return colors
