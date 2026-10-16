"""Instances from predicted offsets: DBSCAN / mean-shift on centre-shifted food points.
Same idea as eval/mf3d_composite.py (DBSCAN on points); here the points are first moved by the predicted
offset to their instance centre, which collapses touching foods that plain geometry can't separate."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, MeanShift


def cluster_instances(xyz, sem, offset, eps=0.006, min_samples=8, min_pts=20, method="dbscan",
                      bandwidth=0.012, food_class=3, assign_rest=True):
    """xyz/offset in metres (N,3), sem (N,) predicted class.  Returns inst (N,) int: 0 = not food, 1..K.
    Clusters smaller than min_pts are dissolved into the nearest kept cluster (assign_rest)."""
    inst = np.zeros(len(xyz), np.int64)
    fm = np.nonzero(sem == food_class)[0]
    if len(fm) < max(min_samples, 2):
        return inst
    shifted = xyz[fm] + offset[fm]
    if method == "meanshift":
        lab = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(shifted)
    else:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(shifted)
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    keep = ids[cnt >= min_pts]
    out = np.full(len(fm), -1, np.int64)
    for k, i in enumerate(keep):
        out[lab == i] = k + 1
    if assign_rest and len(keep) and (out < 0).any():
        good = out > 0
        _, nn = cKDTree(xyz[fm][good]).query(xyz[fm][~good], k=1)
        out[~good] = out[good][nn]
    out[out < 0] = 0
    inst[fm] = out
    return inst


def anchored_instances(xyz, sem, offset, lift, lift_share=None, boundary=None, share_thr=0.8, overlap_thr=0.3,
                       use_boundary=True, split_merged=False, split_frac=0.25, clusters=None, **cluster_kw):
    """"Refine, don't replace" decoding.  Start from the lift labels and only reassign the uncertain points
    U = (lift_share < share_thr) | (boundary == 1) | (predicted class != lift class).
    Predicted clusters (plain decoding: DBSCAN on xyz+offset) are mapped to lift ids by majority overlap (fraction
    of the cluster's points that carry that lift id); a cluster with < overlap_thr overlap with every lift id
    spawns a new id.  Points in U take the predicted class; predicted-food ones take their cluster's mapped id.
    Points outside U keep lift class and id.  split_merged (off by default; beyond the plain spec): when two or
    more clusters each cover >= split_frac of one lift instance, the largest keeps the id, the others get new ids
    and ALL points of that lift instance follow their cluster (lets the net split a merged lift instance).
    NB: `boundary` is a GT-derived flag in the synthetic data (label ambiguity band), so use_boundary leaks a
    little GT (about 0.3 % of points); use_boundary=False removes it.
    Returns (sem, inst): sem in {0 table,1 container,2 clutter,3 food}, inst 0 = not food."""
    n = len(xyz)
    lift = np.asarray(lift)
    lsem = np.zeros(n, np.int64)
    lsem[lift == 254] = 1
    lsem[lift == 253] = 2
    lfood = (lift > 0) & (lift < 252)
    lsem[lfood] = 3
    linst = np.where(lfood, lift, 0).astype(np.int64)
    unc = sem != lsem
    if lift_share is not None:
        unc |= (np.asarray(lift_share) < share_thr) & (lift > 0)
    if use_boundary and boundary is not None:
        unc |= np.asarray(boundary) == 1
    clus = cluster_instances(xyz, sem, offset, **cluster_kw) if clusters is None else clusters
    out_inst, out_sem = linst.copy(), lsem.copy()
    next_id = int(max(lift[lfood].max() if lfood.any() else 0, 0)) + 1000          # never collides with lift ids
    cmap = {}
    for c in np.unique(clus[clus > 0]):
        m = clus == c
        ids, cnt = np.unique(linst[m & lfood], return_counts=True)
        if len(ids) and cnt.max() / m.sum() >= overlap_thr:
            cmap[int(c)] = int(ids[cnt.argmax()])
        else:
            cmap[int(c)] = next_id
            next_id += 1
    if split_merged:
        for i in np.unique(linst[lfood]):
            lm = linst == i
            cs = [(int((lm & (clus == c)).sum()), c) for c in cmap]
            cs = sorted([(k, c) for k, c in cs if k >= split_frac * lm.sum()], reverse=True)
            if len(cs) >= 2:
                for k, (_, c) in enumerate(cs):
                    cmap[c] = int(i) if k == 0 else next_id
                    next_id += k > 0
                unc |= lm                                   # whole instance follows the clusters
    out_sem[unc] = sem[unc]
    out_inst[unc] = 0
    for c, new in cmap.items():
        m = unc & (clus == c) & (sem == 3)
        out_inst[m] = new
    # predicted food in U with no cluster (cannot happen with assign_rest unless no cluster at all): keep lift id
    miss = unc & (sem == 3) & (out_inst == 0) & lfood
    out_inst[miss] = linst[miss]
    out_inst[out_sem != 3] = 0
    return out_sem, out_inst
