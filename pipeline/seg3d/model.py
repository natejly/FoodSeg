"""Models: PTv3 (Sonata init) + semantic/offset(/mass) heads, and a small DGCNN baseline.

The Sonata checkpoint is an encoder-only (enc_mode, traceable) PTv3.  We run its 5 encoder stages and build a
light decoder ourselves: every stage's features are projected to `dec_dim`, unpooled down the pooling chain
(`pooling_inverse`) and summed, so no pretrained decoder is needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import compat

compat.setup()
from sonata.structure import Point                          # noqa: E402
from .data import FEAT_DIM                                   # noqa: E402

N_CLASSES = 4


class Heads(nn.Module):
    def __init__(self, d, mass_head=False):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.sem = nn.Linear(d, N_CLASSES)
        self.off = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.mass = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, 1)) if mass_head else None

    def forward(self, f):
        f = self.trunk(f)
        return dict(sem=self.sem(f), offset=self.off(f), feat=f)

    def mass_pred(self, feat, inst, ids, log_vol):
        """Per-instance log-mass (g) from mean-pooled point features + log volume (ml). ids: list of item ids."""
        pooled = torch.stack([feat[inst == i].mean(0) for i in ids])
        return self.mass(torch.cat([pooled, log_vol.view(-1, 1).to(pooled.dtype)], 1)).squeeze(1)


def koc_forward(model, out, batch):
    """Keep-or-change logits (N,C) in out['koc'] when the model has a KOC head and the batch carries candidates."""
    if getattr(model, "koc", None) is not None and "cand_id" in batch:
        out["koc"] = model.koc(out["feat"], out["sem"], batch["feat"], batch)
    return out


class PTv3Seg(nn.Module):
    def __init__(self, pretrained=True, dec_dim=128, drop_path=0.1, patch_size=1024, mass_head=False,
                 ckpt="sonata", in_channels=FEAT_DIM):
        super().__init__()
        import sonata
        from sonata.model import PointTransformerV3
        cfg = None
        if pretrained:
            ck = sonata.load(ckpt, repo_id="facebook/sonata", ckpt_only=True)
            cfg = dict(ck["config"])
        else:
            cfg = dict(in_channels=9, order=("z", "z-trans", "hilbert", "hilbert-trans"), stride=(2, 2, 2, 2),
                       enc_depths=(3, 3, 3, 12, 3), enc_channels=(48, 96, 192, 384, 512),
                       enc_num_head=(3, 6, 12, 24, 32), enc_patch_size=(1024,) * 5, mlp_ratio=4, qkv_bias=True,
                       shuffle_orders=True, pre_norm=True, enc_mode=True, traceable=True)
        cfg.update(in_channels=in_channels, enable_flash=False, drop_path=drop_path, mask_token=False,
                   enc_patch_size=(patch_size,) * 5, enc_mode=True, traceable=True)
        self.backbone = PointTransformerV3(**cfg)
        self.report = "from scratch"
        if pretrained:
            sd = {k: v for k, v in ck["state_dict"].items() if "mask_token" not in k}
            own = self.backbone.state_dict()
            for k, v in list(sd.items()):                       # stem conv: widen input 9 -> in_channels
                if k in own and own[k].shape != v.shape:
                    w = own[k].clone()
                    nn.init.zeros_(w)               # new channels start silent: frozen == pretrained
                    w[..., : v.shape[-1]] = v
                    sd[k] = w
                    self.report = f"stem {k} widened {tuple(v.shape)}->{tuple(w.shape)}"
            miss = self.backbone.load_state_dict(sd, strict=False)
            self.report = f"sonata loaded; missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}; " + self.report
        ch = cfg["enc_channels"]
        self.proj = nn.ModuleList([nn.Linear(c, dec_dim) for c in ch])
        self.skip = nn.Linear(in_channels, dec_dim)          # raw features (incl. lift prior) -> finest level
        import spconv.pytorch as spconv
        self.local = spconv.SparseSequential(
            spconv.SubMConv3d(dec_dim, dec_dim, 3, padding=1, bias=False, indice_key="loc"),
            nn.BatchNorm1d(dec_dim), nn.ReLU(),
            spconv.SubMConv3d(dec_dim, dec_dim, 3, padding=1, bias=False, indice_key="loc"),
            nn.BatchNorm1d(dec_dim), nn.ReLU())
        self.heads = Heads(dec_dim, mass_head)
        self.mode = "none"

    # --- freezing --------------------------------------------------------------------------
    def set_backbone_trainable(self, mode: str):
        """'none': whole Sonata backbone frozen and run under no_grad (cheap: ~only decoder activations);
        'top': encoder stages 3,4 trainable, stem + stages 0-2 frozen (no_grad);
        'all': everything trainable (needs a small point budget on 6 GB)."""
        self.mode = mode
        for n, p in self.backbone.named_parameters():
            p.requires_grad = mode == "all" or (mode == "top" and (n.startswith("enc.enc3") or n.startswith("enc.enc4")))

    def _encode(self, batch):
        bb = self.backbone
        dev = batch["feat"].device
        point = Point(dict(coord=batch["coord"], grid_coord=batch["grid_coord"], feat=batch["feat"],
                           offset=batch["offset_ptr"].to(dev)))
        n_free = {"none": 0, "top": 3, "all": 5}[self.mode]
        with torch.set_grad_enabled(self.mode == "all"):
            point = bb.embedding(point)
            point.serialization(order=bb.order, shuffle_orders=bb.shuffle_orders)
            point.sparsify()
        for s in range(bb.num_stages):
            stage = getattr(bb.enc, f"enc{s}")
            with torch.set_grad_enabled(s >= n_free and self.mode != "none"):
                point = stage(point)
        return point

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "none":
            self.backbone.eval()
        return self

    def forward(self, batch):
        import spconv.pytorch as spconv
        point = self._encode(batch)
        pts, p = [], point
        while True:
            pts.append(p)
            if "pooling_parent" not in p.keys():
                break
            p = p["pooling_parent"]
        # pts[0] = coarsest ... pts[-1] = finest
        f = self.proj[len(pts) - 1](pts[0].feat)
        for lvl in range(len(pts) - 1):
            child, parent = pts[lvl], pts[lvl + 1]
            f = f[child["pooling_inverse"]] + self.proj[len(pts) - 2 - lvl](parent.feat)
        f = f + self.skip(batch["feat"].to(f.dtype))
        gc = batch["grid_coord"]
        idx = torch.cat([batch["batch"].view(-1, 1).int(), gc[:, [2, 1, 0]].int()], 1)
        st = spconv.SparseConvTensor(f, idx, [int(gc[:, 2].max()) + 1, int(gc[:, 1].max()) + 1, int(gc[:, 0].max()) + 1],
                                     int(batch["batch"].max()) + 1)
        f = f + self.local(st).features
        return koc_forward(self, self.heads(f), batch)


# ----------------------------------------------------------------------------- DGCNN baseline
def _knn(x, k):                                             # x (B,C,N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k, dim=-1)[1]


def _edge(x, idx):
    B, C, N = x.shape
    k = idx.shape[-1]
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    xt = x.transpose(2, 1).reshape(B * N, C)
    nb = xt[(idx + base).reshape(-1)].view(B, N, k, C)
    ctr = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([nb - ctr, ctr], 3).permute(0, 3, 1, 2)


class DGCNNSeg(nn.Module):
    """Same 3x EdgeConv(k) + global-feature layout as eval/mf3d_seg_train.py's DGCNN (that file's model only
    takes xyz and outputs offsets, so it is reimplemented here with the full feature vector and a semantic
    head).  Subsamples n_pts points per scene (kNN is O(N^2)), predicts there, copies to the rest by nearest
    sampled neighbour."""

    def __init__(self, in_channels=FEAT_DIM, k=16, n_pts=4096, dec_dim=128, mass_head=False):
        super().__init__()
        self.k, self.n_pts = k, n_pts

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.e1, self.e2, self.e3 = blk(2 * in_channels, 64), blk(128, 64), blk(128, 64)
        self.glob = nn.Sequential(nn.Conv1d(192, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.mix = nn.Sequential(nn.Conv1d(192 + 512, dec_dim, 1, bias=False), nn.BatchNorm1d(dec_dim), nn.LeakyReLU(0.2))
        self.heads = Heads(dec_dim, mass_head)

    def set_backbone_trainable(self, mode):
        pass

    def train(self, mode=True):
        """BatchNorm always uses per-scene batch statistics (batch = 1 scene).  With running statistics the net
        collapsed at eval (val acc 31 % vs 94 % with batch stats on the same weights: scene-level feature shifts are
        normalised away in training but averaged over scenes in the running stats), so eval == instance-norm style."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
                m.momentum = 0.0 if not mode else 0.1          # do not corrupt running stats while evaluating
        return self

    def forward(self, batch):
        outs, start = [], 0
        for n in batch["n"].tolist():
            x = batch["feat"][start:start + n]
            c = batch["coord"][start:start + n]
            start += n
            sel = torch.randperm(n, device=x.device)[: self.n_pts] if n > self.n_pts else torch.arange(n, device=x.device)
            xs = x[sel].t().unsqueeze(0)
            x1 = self.e1(_edge(xs, _knn(xs, self.k))).max(-1)[0]
            x2 = self.e2(_edge(x1, _knn(x1, self.k))).max(-1)[0]
            x3 = self.e3(_edge(x2, _knn(x2, self.k))).max(-1)[0]
            f = torch.cat([x1, x2, x3], 1)
            g = self.glob(f).max(-1, keepdim=True)[0].expand(-1, -1, f.shape[-1])
            fs = self.mix(torch.cat([f, g], 1))[0].t()                       # (n_s, D)
            if len(sel) < n:
                with torch.no_grad():
                    nn_idx = torch.cat([torch.cdist(c[i:i + 4096], c[sel]).argmin(1) for i in range(0, n, 4096)])
                fs = fs[nn_idx]
            outs.append(fs)
        return koc_forward(self, self.heads(torch.cat(outs)), batch)


def build(name="ptv3", **kw):
    if name == "dgcnn":
        m = DGCNNSeg(mass_head=kw.get("mass_head", False))
    else:
        m = PTv3Seg(pretrained=kw.get("pretrained", True), drop_path=kw.get("drop_path", 0.1),
                    patch_size=kw.get("patch_size", 1024), mass_head=kw.get("mass_head", False))
    m.koc = None
    if kw.get("koc", False):
        from .koc import KOCHead
        m.koc = KOCHead(m.heads.sem.in_features, keep_bias=kw.get("koc_bias", 2.0))
    return m
