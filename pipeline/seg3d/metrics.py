"""Metrics for one val set.  Everything is computed at voxel level (the points the net sees), metres.

semantic mIoU        mean IoU over the 4 classes present in GT.
instance mIoU        food only; Hungarian matching on IoU between GT items and predicted instances,
                     unmatched GT item = 0, averaged over GT items (pooled over scans).
PQ                   panoptic quality over food instances, match iff IoU > 0.5: sum IoU_tp / (TP + FP/2 + FN/2).
boundary IoU         GT items vs matched prediction, restricted to the band of points within 3 mm of a GT
                     label change (any class); mean over GT items having band points.
exact count rate     fraction of scans with #predicted instances == #GT items.
volume APE           per GT item, |V_pred - V_gt| / V_gt with V_gt = gt.json volume_ml and V_pred = convex hull of
                     the matched predicted instance's points together with their projection onto a flat base
                     plane at the item's own 1st-percentile height (mf3d_composite.hull_vol recipe, base moved because
                     food sits on the container floor, not on z=0); 0 for an unmatched item (APE = 1).
                     "vol_ape_gtpts" is the same hull computed from the GT item's own points = the floor
                     error of the volume estimator, independent of segmentation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def hull_vol_ml(p: np.ndarray) -> float:
    if len(p) < 10:
        return 0.0
    try:
        import open3d as o3d
        base = p.copy()
        base[:, 2] = np.percentile(p[:, 2], 1)       # flat base at the item's own lowest point (items sit on
                                                       # the container floor, not the table)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([p, base]).astype(np.float64)))
        return float(pc.compute_convex_hull()[0].get_volume() * 1e6)
    except Exception:                                       # noqa: BLE001
        return 0.0


def boundary_band(xyz, label, width=0.003):
    pairs = cKDTree(xyz).query_pairs(width, output_type="ndarray")
    band = np.zeros(len(xyz), bool)
    if len(pairs):
        diff = label[pairs[:, 0]] != label[pairs[:, 1]]
        band[pairs[diff].ravel()] = True
    return band


def match(gt, pred):
    g_ids = [i for i in np.unique(gt) if i > 0]
    p_ids = [i for i in np.unique(pred) if i > 0]
    iou = np.zeros((len(g_ids), max(len(p_ids), 1)))
    for a, g in enumerate(g_ids):
        gm = gt == g
        for b, p in enumerate(p_ids):
            pm = pred == p
            iou[a, b] = (gm & pm).sum() / max((gm | pm).sum(), 1)
    r, c = linear_sum_assignment(-iou) if len(g_ids) and p_ids else (np.array([], int), np.array([], int))
    m = {g_ids[a]: (p_ids[b], iou[a, b]) for a, b in zip(r, c)}
    return g_ids, p_ids, m


def point_correct(pred_sem, pred_inst, gt_sem, gt_inst):
    """Per-point correctness of a (sem, inst) prediction: non-food GT -> class matches; food GT -> the point lies in
    the predicted instance Hungarian-matched to its GT item (and is predicted food)."""
    ok = (pred_sem == gt_sem)
    _, _, m = match(gt_inst, pred_inst)
    food = gt_sem == 3
    ok_food = np.zeros(len(gt_sem), bool)
    for g, (p, _) in m.items():
        ok_food |= (gt_inst == g) & (pred_inst == p) & (pred_sem == 3)
    return np.where(food, ok_food, ok)


class Accumulator:
    def __init__(self, n_cls=4, volumes=True):
        self.conf = np.zeros((n_cls, n_cls), np.int64)
        self.inst_iou, self.bnd_iou, self.ape, self.ape_gt = [], [], [], []
        self.tp_iou, self.tp, self.fp, self.fn = 0.0, 0, 0, 0
        self.count_ok, self.n_scans = 0, 0
        self.volumes = volumes
        self.scan_n = []                      # GT items per scan (aligned with scan_miou), for item-level bootstraps
        self.scan_miou = []                   # per-scan mean instance IoU (for spread over scans)
        # refinement counts vs the lift prior: [wrong->right, wrong, right->wrong, right], over two point domains
        self.koc = np.zeros(5, np.int64)      # points, kept, unrecoverable, lift-wrong, lift-wrong & unrecoverable
        self.rf = {"all": np.zeros(4, np.int64), "food": np.zeros(4, np.int64)}

    def add(self, pred_sem, pred_inst, xyz, gt_sem, gt_inst, gt_label, items, lift=None):
        k = self.conf.shape[0]
        n_before = len(self.inst_iou)
        if lift is not None:
            ls, li = lift_baseline_pred(lift)
            rl = point_correct(ls, li, gt_sem, gt_inst)
            rm = point_correct(pred_sem, pred_inst, gt_sem, gt_inst)
            dom = {"all": np.ones(len(rl), bool), "food": (gt_sem == 3) | (ls == 3) | (pred_sem == 3)}
            for key, d in dom.items():
                self.rf[key] += [(~rl & rm & d).sum(), (~rl & d).sum(), (rl & ~rm & d).sum(), (rl & d).sum()]
        self.conf += np.bincount(gt_sem * k + pred_sem, minlength=k * k).reshape(k, k)
        g_ids, p_ids, m = match(gt_inst, pred_inst)
        self.n_scans += 1
        self.count_ok += int(len(g_ids) == len(p_ids))
        band = boundary_band(xyz, gt_label)
        for g in g_ids:
            iou = m[g][1] if g in m else 0.0
            self.inst_iou.append(iou)
            if iou > 0.5:
                self.tp += 1
                self.tp_iou += iou
            else:
                self.fn += 1
            gm = gt_inst == g
            pm = (pred_inst == m[g][0]) if g in m else np.zeros_like(gm)
            u = (gm | pm) & band
            if (gm & band).any():
                self.bnd_iou.append(((gm & pm) & band).sum() / max(u.sum(), 1))
            if self.volumes and items.get(int(g), {}).get("volume_ml"):
                vg = items[int(g)]["volume_ml"]
                vp = hull_vol_ml(xyz[pm]) if g in m else 0.0
                self.ape.append(abs(vp - vg) / vg)
                self.ape_gt.append(abs(hull_vol_ml(xyz[gm]) - vg) / vg)
        if len(self.inst_iou) > n_before:
            self.scan_miou.append(float(np.mean(self.inst_iou[n_before:])))
            self.scan_n.append(len(self.inst_iou) - n_before)
        self.fp += sum(1 for p in p_ids if p not in {v[0] for g, v in m.items() if v[1] > 0.5})

    def add_koc(self, pred_label, lift, gt_label, unrec):
        """Keep-or-change bookkeeping (label space, 252 == 0).  unrec (N,) bool: GT label is not among the point's
        candidates (cannot be fixed by any KOC decision)."""
        nl = lambda a: np.where(np.asarray(a) == 252, 0, a)          # noqa: E731
        pred, lift, gt = nl(pred_label), nl(lift), nl(gt_label)
        wrong = lift != gt
        self.koc += [len(pred), (pred == lift).sum(), unrec.sum(), wrong.sum(), (wrong & unrec).sum()]

    def result(self):
        c = self.conf
        iou = np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)
        present = c.sum(1) > 0
        pq = self.tp_iou / max(self.tp + 0.5 * self.fp + 0.5 * self.fn, 1e-9) if (self.tp + self.fp + self.fn) else float("nan")
        mean = lambda a: float(np.mean(a)) if len(a) else float("nan")      # noqa: E731
        return dict(sem_iou={n: float(v) for n, v in zip(["table", "container", "clutter", "food"], iou)},
                    sem_miou=float(iou[present].mean()), inst_miou=mean(self.inst_iou),
                    recall50=mean([x > 0.5 for x in self.inst_iou]), pq=float(pq), boundary_iou=mean(self.bnd_iou),
                    exact_count=self.count_ok / max(self.n_scans, 1), vol_ape=mean(self.ape),
                    vol_ape_median=float(np.median(self.ape)) if self.ape else float("nan"),
                    vol_ape_gtpts=mean(self.ape_gt), n_items=len(self.inst_iou), n_scans=self.n_scans,
                    scan_inst_miou=self.scan_miou, scan_n_items=self.scan_n, item_inst_iou=[float(x) for x in self.inst_iou],
                    **{f"{kind}_{key}": float(v[a] / max(v[b], 1)) for key, v in self.rf.items()
                       for kind, a, b in (("corrected", 0, 1), ("broken", 2, 3))},
                    **({"keep_rate": float(self.koc[1] / self.koc[0]), "unrecoverable": float(self.koc[2] / self.koc[0]),
                        "unrecoverable_of_wrong": float(self.koc[4] / max(self.koc[3], 1))} if self.koc[0] else {}),
                    **{f"n_wrong_{key}": int(v[1]) for key, v in self.rf.items()})


def lift_baseline_pred(lift):
    """Trivial baseline: lift_label as the prediction."""
    sem = np.zeros(len(lift), np.int64)
    sem[lift == 254] = 1
    sem[lift == 253] = 2
    food = (lift > 0) & (lift < 252)
    sem[food] = 3
    return sem, np.where(food, lift, 0).astype(np.int64)
