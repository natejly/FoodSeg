"""Learned 3D instance segmentation of touching foods (eval.mf3d_composite scenes).

    python -m eval.mf3d_seg_train ../data/metafood3d/composite --view full --out ../results/metafood3d/seg/dgcnn_full

A small DGCNN (3 EdgeConv layers, k = 16) predicts, per point, the offset to its
instance's centroid; shifted points are clustered with DBSCAN (eps picked on
training scenes) - the vote-then-cluster scheme of PointGroup-style methods,
sized for a 6 GB card (4096 points per scene). Geometry only: MetaFood3D clouds
carry no colour. Scored with eval.mf3d_composite.score_scene on the test scenes
(objects disjoint from training).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from eval.mf3d_composite import score_scene, summarize

N = 4096


def knn(x: torch.Tensor, k: int) -> torch.Tensor:          # x (B,C,N) -> (B,N,k)
    inner = -2 * x.transpose(2, 1) @ x
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k, dim=-1)[1]


def edge_feat(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, C, Np = x.shape
    k = idx.shape[-1]
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * Np
    xt = x.transpose(2, 1).reshape(B * Np, C)
    nb = xt[(idx + base).reshape(-1)].view(B, Np, k, C)
    ctr = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([nb - ctr, ctr], 3).permute(0, 3, 1, 2)   # (B,2C,N,k)


class DGCNN(nn.Module):
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.e1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.e2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.e3 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.glob = nn.Sequential(nn.Conv1d(192, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Conv1d(192 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                                  nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                                  nn.Conv1d(128, 3, 1))

    def forward(self, x):                                   # x (B,3,N), centred, metres x 10
        x1 = self.e1(edge_feat(x, knn(x, self.k))).max(-1)[0]
        x2 = self.e2(edge_feat(x1, knn(x1, self.k))).max(-1)[0]
        x3 = self.e3(edge_feat(x2, knn(x2, self.k))).max(-1)[0]
        f = torch.cat([x1, x2, x3], 1)
        g = self.glob(f).max(-1, keepdim=True)[0].expand(-1, -1, f.shape[-1])
        return self.head(torch.cat([f, g], 1))              # offsets (B,3,N), same units as x


def prep(P: np.ndarray, L: np.ndarray, rng):
    sel = rng.choice(P.shape[0], N, replace=False)
    p, l = P[sel].astype(np.float32), L[sel]
    c = p.mean(0)
    p = (p - c) * 10.0                                      # decimetres: unit-ish scale
    off = np.zeros_like(p)
    for i in np.unique(l):
        m = l == i
        off[m] = p[m].mean(0) - p[m]
    return p, off, l, sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--view", choices=["full", "top"], default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    tr, te = np.load(args.data / "train.npz"), np.load(args.data / "test.npz")
    Xtr, Ytr = tr[args.view], tr[f"{args.view}_lab"]
    Xte, Yte = te[args.view], te[f"{args.view}_lab"]
    dev = "cuda"
    model = DGCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(Xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler()
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(Xtr))
        for b in range(len(Xtr) // args.bs):
            batch = [prep(Xtr[i], Ytr[i], rng) for i in order[b * args.bs:(b + 1) * args.bs]]
            p = np.stack([x[0] for x in batch])
            th = rng.uniform(0, 2 * np.pi, len(batch))       # yaw augmentation
            R = np.stack([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in th]).astype(np.float32)
            p = np.einsum("bij,bnj->bni", R, p)
            off = np.einsum("bij,bnj->bni", R, np.stack([x[1] for x in batch]))
            x = torch.from_numpy(p).to(dev).transpose(1, 2)
            y = torch.from_numpy(off).to(dev).transpose(1, 2)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.smooth_l1_loss(model(x).float(), y, beta=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        print(json.dumps({"epoch": ep, "loss": tot / (len(Xtr) // args.bs), "sec": round(time.time() - t0, 1),
                          "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    @torch.no_grad()
    def shifted(P, L):
        model.eval()
        p, _, l, sel = prep(P, L, np.random.default_rng(1))
        o = model(torch.from_numpy(p)[None].to(dev).transpose(1, 2)).float()[0].T.cpu().numpy()
        return p, p + o, l

    def run(X, Y, idx, eps):
        sc = []
        for i in idx:
            p, s, l = shifted(X[i], Y[i])
            pred = DBSCAN(eps=eps, min_samples=8).fit_predict(s)
            sc.append(score_scene(p / 10.0, l, pred))
        return sc

    best = max(((eps, np.mean([s["miou"] for s in run(Xtr, Ytr, range(200), eps)])) for eps in (0.05, 0.1, 0.15, 0.2, 0.3)),
               key=lambda t: t[1])
    rep = {"view": args.view, "eps_dm": best[0], **summarize(run(Xte, Yte, range(len(Xte)), best[0]))}
    json.dump(rep, open(args.out / "summary.json", "w"), indent=2)
    print("TEST " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
