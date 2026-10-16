"""Keep-or-change (KOC) instance refinement head.

Per point: decide KEEP the lift label or CHANGE to one of the K nearest *other* lift labels within radius r
(background, container 254, clutter 253 and food ids are all ordinary candidates), or "none".

Candidate slots (C = K + 2):  0 = self (the point's own lift label), 1..K = nearest distinct other lift labels
(by distance to the nearest point carrying that label, <= r; unused slots are padding), C-1 = "none".
The target is the slot whose id equals the GT label (self wins ties); if the GT label is not a candidate the point is
"unrecoverable" and its target is the none slot.  Decoding = argmax, none -> keep (so an unrecoverable point
stays at its lift label).  Ids are compared literally (synthetic lift ids == GT ids; a lift merge therefore makes the
merged-away item unrecoverable).  Label 252 is folded into 0 (background).

Candidate embedding = mean of a projection of the point features over the M nearest points (within r) that carry
that label, [+ mean lift_share / lift_nvotes of those points], plus role / class one-hots, the point's own
lift_share / lift_nvotes, the distance to the candidate, the member fraction, and the (detached) log-prob that the
semantic head gives to the candidate's class.  Score = small MLP; softmax over candidates; a learnable bias favours keep.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

SHARE_COL, NV_COL = 28, 29                       # lift_share / lift_nvotes columns of the 30-ch feature vector
ROLE_SELF, ROLE_NB, ROLE_NONE = 0, 1, 2


def norm_label(l):
    l = np.asarray(l).astype(np.int64)
    return np.where(l == 252, 0, l)


def label_class(l):
    """0 background, 1 container, 2 clutter, 3 food (for normalised labels)."""
    l = np.asarray(l)
    return np.where(l == 254, 1, np.where(l == 253, 2, np.where((l > 0) & (l < 252), 3, 0)))


def build_candidates(xyz, lift, gt=None, K=4, r=0.025, M=16):
    """numpy.  xyz (N,3) metres, lift (N,) labels, gt (N,) GT labels or None.  Returns dict of arrays:
    cand_id (N,C) int64 (-1 none, -2 pad), cand_valid (N,C) bool, cand_role (N,C), cand_cls (N,C) (-1 none/pad),
    cand_dist (N,C) f32 (/r), cand_nb (N,C,M) int64 point indices (-1 pad), cand_tgt (N,) int64 (only with gt)."""
    n, C = len(xyz), K + 2
    lab = norm_label(lift)
    ids = np.unique(lab)
    trees, members = {}, {}
    for c in ids:
        members[int(c)] = np.nonzero(lab == c)[0]
        trees[int(c)] = cKDTree(xyz[members[int(c)]])
    D = np.full((n, len(ids)), np.inf)
    for j, c in enumerate(ids):
        d, _ = trees[int(c)].query(xyz, k=1, distance_upper_bound=r)
        D[:, j] = d
    self_j = np.searchsorted(ids, lab)
    D[np.arange(n), self_j] = np.inf                                   # others only
    kk = min(K, len(ids))
    order = np.argsort(D, axis=1)[:, :kk]
    dsel = np.take_along_axis(D, order, 1)
    cand_id = np.full((n, C), -2, np.int64)
    cand_dist = np.zeros((n, C), np.float32)
    cand_id[:, 0] = lab
    ok = np.isfinite(dsel)
    cand_id[:, 1:1 + kk] = np.where(ok, ids[order], -2)
    cand_dist[:, 1:1 + kk] = np.where(ok, dsel / r, 0)
    cand_id[:, C - 1] = -1
    valid = cand_id != -2
    role = np.zeros((n, C), np.int64)
    role[:, 1:] = ROLE_NB
    role[:, C - 1] = ROLE_NONE
    cls = np.where(cand_id >= 0, label_class(np.maximum(cand_id, 0)), -1)
    nb = np.full((n, C, M), -1, np.int64)
    for s in range(C - 1):
        for c in np.unique(cand_id[:, s][cand_id[:, s] >= 0]):
            rows = np.nonzero(cand_id[:, s] == c)[0]
            mem = members[int(c)]
            kq = min(M, len(mem))
            d, ix = trees[int(c)].query(xyz[rows], k=kq, distance_upper_bound=r)
            d, ix = d.reshape(len(rows), kq), ix.reshape(len(rows), kq)
            nb[rows, s, :kq] = np.where(np.isfinite(d), mem[np.minimum(ix, len(mem) - 1)], -1)
    out = dict(cand_id=cand_id, cand_valid=valid, cand_role=role, cand_cls=cls.astype(np.int64),
               cand_dist=cand_dist, cand_nb=nb)
    if gt is not None:
        out["cand_tgt"] = candidate_target(cand_id, gt)
    return out


def candidate_target(cand_id, gt):
    g = norm_label(gt)
    match = (cand_id == g[:, None]) & (cand_id >= 0)
    tgt = np.where(match.any(1), match.argmax(1), cand_id.shape[1] - 1)       # argmax -> first = self wins ties
    return tgt.astype(np.int64)


def decode_labels(logits, cand_id):
    """argmax over candidates; none (last slot) -> keep.  Returns (N,) labels, (N,) bool changed."""
    pick = logits.argmax(1)
    none = pick == cand_id.shape[1] - 1
    pick = torch.where(none, torch.zeros_like(pick), pick)
    lab = cand_id.gather(1, pick[:, None])[:, 0]
    return lab, pick != 0


class KOCHead(nn.Module):
    def __init__(self, d, dg=32, hid=64, keep_bias=2.0):
        super().__init__()
        self.pq, self.pg = nn.Linear(d, dg), nn.Linear(d, dg)
        n_in = 3 * dg + 2 + 3 + 4 + 2 + 3 + 1
        self.mlp = nn.Sequential(nn.Linear(n_in, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.keep_bias = nn.Parameter(torch.tensor(float(keep_bias)))

    def forward(self, f, sem_logits, feat_raw, b):
        """f (N,d) point features, sem_logits (N,4), feat_raw (N,30), b: dict with cand_* tensors.  -> (N,C) logits."""
        nb = b["cand_nb"]
        N, C, M = nb.shape
        own = feat_raw[:, SHARE_COL:NV_COL + 1].to(f.dtype)
        q = self.pq(f)
        g = self.pg(f)
        gx = torch.cat([g, own], 1)
        m = nb >= 0
        e = gx[nb.clamp(min=0)] * m.unsqueeze(-1).to(gx.dtype)
        cnt = m.sum(2)
        e = e.sum(2) / cnt.clamp(min=1).unsqueeze(-1).to(gx.dtype)             # (N,C,dg+2)
        eg, es = e[..., :-2], e[..., -2:]
        cls = b["cand_cls"]
        lp = F.log_softmax(sem_logits.float().detach(), 1)
        clp = lp.gather(1, cls.clamp(min=0)) * (cls >= 0)
        role = F.one_hot(b["cand_role"], 3).to(f.dtype)
        cls1 = F.one_hot(cls.clamp(min=0), 4).to(f.dtype) * (cls >= 0).unsqueeze(-1).to(f.dtype)
        qq = q.unsqueeze(1).expand(-1, C, -1)
        extra = torch.cat([es, role, cls1, own.unsqueeze(1).expand(-1, C, -1), b["cand_dist"].unsqueeze(-1).to(f.dtype),
                           (cnt.float() / M).unsqueeze(-1).to(f.dtype), (cnt == 0).unsqueeze(-1).to(f.dtype),
                           clp.unsqueeze(-1).to(f.dtype)], -1)
        x = torch.cat([qq, eg, qq * eg, extra], -1)
        logit = self.mlp(x).squeeze(-1).float()
        logit = torch.cat([logit[:, :1] + self.keep_bias, logit[:, 1:]], 1)
        return logit.masked_fill(~b["cand_valid"], -1e4)


def koc_loss(logits, b, keep_eps=0.05, change_w=2.0, gamma=0.0):
    """CE over candidates; target smoothed toward keep (eps), change points (target != self) weighted change_w,
    optional focal (1-p_t)^gamma; boundary-masked points (weight 0) ignored.  Returns (loss, parts)."""
    tgt = b["cand_tgt"].to(logits.device)
    w = b["weight"].to(logits.device)
    logp = F.log_softmax(logits.float(), 1)
    nll = -logp.gather(1, tgt[:, None])[:, 0]
    nll_keep = -logp[:, 0]
    ce = (1 - keep_eps) * nll + keep_eps * nll_keep
    if gamma > 0:
        ce = ce * (1 - torch.exp(-nll.detach())).pow(gamma)
    pw = torch.where(tgt != 0, torch.full_like(w, change_w), torch.ones_like(w)) * w
    loss = (ce * pw).sum() / pw.sum().clamp(min=1e-6)
    with torch.no_grad():
        pred = logits.argmax(1)
        pred = torch.where(pred == logits.shape[1] - 1, torch.zeros_like(pred), pred)
        ch = (tgt != 0) & (tgt != logits.shape[1] - 1) & (w > 0)
        parts = dict(koc=float(loss.detach()), koc_change_acc=float((pred[ch] == tgt[ch]).float().mean()) if ch.any() else float("nan"),
                     koc_keep_rate=float((pred == 0).float().mean()))
    return loss, parts
