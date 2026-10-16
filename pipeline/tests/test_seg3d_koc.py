import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seg3d import metrics as M                                      # noqa: E402
from seg3d.koc import KOCHead, build_candidates, decode_labels, koc_loss    # noqa: E402


def blobs(rng, n=600):
    """Two touching blobs (ids 1, 2, 2 cm apart) on a table; returns xyz, gt."""
    a = rng.normal(0, 0.006, (n, 3)) + [0, 0, 0.02]
    b = rng.normal(0, 0.006, (n, 3)) + [0.02, 0, 0.02]
    tab = np.c_[rng.uniform(-0.04, 0.06, (300, 2)), np.zeros(300)]
    return np.concatenate([a, b, tab]).astype(np.float32), np.r_[np.full(n, 1), np.full(n, 2), np.zeros(300)].astype(np.int64)


def test_candidates_self_k_none():
    rng = np.random.default_rng(0)
    xyz, gt = blobs(rng)
    lift = gt.copy()
    lift[:50] = 254                                         # a few container points
    lift[50:100] = 253                                      # and clutter
    kc = build_candidates(xyz, lift, gt, K=2, r=0.03, M=8)
    C = 4
    assert kc["cand_id"].shape == (len(xyz), C) and kc["cand_nb"].shape == (len(xyz), C, 8)
    assert (kc["cand_id"][:, 0] == lift).all()                                  # self always slot 0
    assert (kc["cand_id"][:, -1] == -1).all() and kc["cand_valid"][:, -1].all()  # none slot present
    others = kc["cand_id"][:, 1:-1]
    for i in range(len(xyz)):
        real = others[i][others[i] >= 0]
        assert len(real) <= 2 and lift[i] not in real and len(set(real.tolist())) == len(real)   # K respected, distinct, no self
    assert (kc["cand_id"][:, 1:-1] == 1).any() and (kc["cand_id"][:, 1:-1] == 254).any()
    assert ((kc["cand_nb"][:, 0] >= 0).sum(1) >= 1).all()                       # self neighbourhood non-empty
    ok = kc["cand_tgt"] != C - 1
    assert (kc["cand_tgt"][lift == gt] == 0).all()                              # correct points: keep
    # GT not among candidates -> none slot: dropped label 2 far from any point labelled 2
    lift2 = np.where(gt == 2, 0, gt)
    kc2 = build_candidates(xyz, lift2, gt, K=2, r=0.03, M=8)
    assert (kc2["cand_tgt"][gt == 2] == C - 1).all() and ok.mean() > 0.9


def test_head_fixes_strip_without_breaking():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    xyz, gt = blobs(rng, 700)
    lift = gt.copy()
    strip = (gt == 2) & (xyz[:, 0] < 0.0145)                  # strip of blob 2 nearest blob 1 mislabelled 1
    lift[strip] = 1
    assert strip.sum() > 60
    # point features: colour distinguishes the blobs (+ noise), plus lift_share / nvotes columns
    f = np.zeros((len(xyz), 30), np.float32)
    f[:, 3] = (gt == 1) * 1.0 + (gt == 2) * -1.0 + rng.normal(0, 0.3, len(xyz))
    f[:, 4] = rng.normal(0, 1, len(xyz))
    f[:, 28] = np.where(strip, 0.4, 0.9)
    f[:, 29] = 0.5
    kc = build_candidates(xyz, lift, gt, K=3, r=0.025, M=12)
    b = {k: torch.from_numpy(v) for k, v in kc.items()}
    b["weight"] = torch.ones(len(xyz))
    feat = torch.from_numpy(f)
    enc = torch.nn.Linear(30, 32)                             # stand-in for the PTv3 decoder features
    sem = torch.zeros(len(xyz), 4)
    head = KOCHead(32)
    opt = torch.optim.Adam(list(head.parameters()) + list(enc.parameters()), 3e-3)
    for _ in range(300):
        loss, _ = koc_loss(head(enc(feat), sem, feat, b), b, 0.05, 2.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lab, changed = decode_labels(head(enc(feat), sem, feat, b), b["cand_id"])
    lab = lab.numpy()
    right0 = lift == gt
    corrected = (lab[strip] == gt[strip]).mean()
    broken = (lab[right0] != gt[right0]).mean()
    assert corrected > 0.8, corrected
    assert broken < 0.02, broken
    ps, pi = M.lift_baseline_pred(lab)
    assert (pi[gt == 2] == 2).mean() > 0.97
