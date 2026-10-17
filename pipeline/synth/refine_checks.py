"""Ground-truth preservation checks for refined frames (see synth.refine).

    python -m synth.refine_checks --orig <root> --ref <root_ref> [--scans ...] [--json out.json]

  boundary  seg-boundary F-score (tol 3 px @960) of a Canny edge map, orig vs refined, plus the
            symmetric edge drift (median distance from each edge px of orig to nearest refined edge).
  colour    per-label mean colour change (RGB, 0-255) over seg labels (food / clutter / dish / bg)
            and the area fraction where the blurred image changed a lot (hallucination / removal proxy).
  views     multi-view consistency: pixels of frame i are reprojected into frame j with depth + pose
            (occlusion-checked); mean |dRGB| of the matched pixels after Gaussian blur sigma 1.5/4/12 px
            @960, for orig and refined. Refinement should not add much beyond the renders' own
            view-dependence (the 'excess' = sqrt(ref^2 - orig^2)).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W0 = 960


def load(root: Path, sid: str, k: int):
    im = cv2.cvtColor(cv2.imread(str(root / "scans" / sid / "rgb" / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    return cv2.resize(im, (W0, W0 * h // w), interpolation=cv2.INTER_AREA)


def raw(root: Path, sid: str, k: int):
    r = np.load(root / "raw" / sid / f"frame_{k:06d}.npz")
    seg, dep = r["seg"], r["depth"]
    h, w = seg.shape
    sz = (W0, W0 * h // w)
    return (cv2.resize(seg, sz, interpolation=cv2.INTER_NEAREST),
            cv2.resize(dep, sz, interpolation=cv2.INTER_NEAREST))


def seg_boundary(seg):
    s = seg.astype(np.int32)
    b = np.zeros(seg.shape, bool)
    b[:, 1:] |= s[:, 1:] != s[:, :-1]
    b[1:, :] |= s[1:, :] != s[:-1, :]
    return b


def edges(im):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    return cv2.Canny(g, 25, 70) > 0


def boundary_metrics(a, b, seg):
    sb = seg_boundary(seg)
    # ignore the image border and seg boundaries between two background-ish regions
    out = {}
    dt_sb = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 3)
    for name, im in (("orig", a), ("ref", b)):
        e = edges(im)
        dt_e = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
        rec = float((dt_e[sb] <= 3).mean())          # seg boundary with an image edge nearby
        prec = float((dt_sb[e] <= 3).mean())         # image edge near a seg boundary
        out[name] = dict(recall=rec, precision=prec, f=2 * rec * prec / max(rec + prec, 1e-9),
                         med_dist_to_edge=float(np.median(dt_e[sb])))
    ea, eb = edges(a), edges(b)
    dtb = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    dta = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    out["edge_drift_med"] = float(np.median(dtb[ea]))
    out["edge_drift_p90"] = float(np.percentile(dtb[ea], 90))
    out["edge_within2_orig_to_ref"] = float((dtb[ea] <= 2).mean())
    out["edge_within2_ref_to_orig"] = float((dta[eb] <= 2).mean())
    return out


def label_metrics(a, b, seg):
    out = {}
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    groups = {"food": (seg >= 1) & (seg <= 252), "clutter": seg == 253, "dish": seg == 254, "bg": seg == 0}
    for n, m in groups.items():
        if m.sum() < 50:
            continue
        d = fb[m].mean(0) - fa[m].mean(0)
        out[n] = dict(dmean=[float(v) for v in d], dmean_abs=float(np.abs(d).mean()),
                      frac=float(m.mean()))
    # per food instance
    inst = []
    for l in np.unique(seg):
        if 1 <= l <= 252:
            m = seg == l
            if m.sum() > 400:
                inst.append(float(np.abs(fb[m].mean(0) - fa[m].mean(0)).mean()))
    out["food_instance_dmean_abs_mean"] = float(np.mean(inst)) if inst else None
    out["food_instance_dmean_abs_max"] = float(np.max(inst)) if inst else None
    # hallucination / removal proxy: where the low-pass image changes a lot
    la, lb = cv2.GaussianBlur(fa, (0, 0), 6), cv2.GaussianBlur(fb, (0, 0), 6)
    ch = np.abs(la - lb).mean(2)
    out["frac_lowpass_change_gt25"] = float((ch > 25).mean())
    out["frac_lowpass_change_gt15"] = float((ch > 15).mean())
    out["lowpass_change_mean"] = float(ch.mean())
    return out


def cam_matrices(root: Path, sid: str, pose_source: str = "frames"):
    """frame dicts with intrinsics + ARKit transform. pose_source "spec": the TRUE poses from the spec
    (world_to_arkit @ camera c2w; exported frames.jsonl poses may carry drift noise)."""
    fr = [json.loads(l) for l in open(root / "scans" / sid / "frames.jsonl")]
    if pose_source == "spec":
        sp = json.load(open(root / "specs" / sid / "spec.json"))
        W = np.array(sp["world_to_arkit"])
        for f, c in zip(fr, sp["camera"]["c2w"]):
            f["transform"] = (W @ np.array(c)).tolist()
    return fr


def reproject(fi, fj, di, dj, sc):
    """pixels of i -> pixel coords in j, predicted z in j, via ARKit poses (camera looks down -z, y up)."""
    K = np.array(fi["intrinsics"]) * np.array([[sc], [sc], [1]])
    Ti, Tj = np.array(fi["transform"]), np.array(fj["transform"])
    h, w = di.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = di
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, -y, -z, np.ones_like(z)], -1).reshape(-1, 4)   # ARKit camera space
    pw = pc @ Ti.T
    pj = pw @ np.linalg.inv(Tj).T
    zj = -pj[:, 2]
    uj = pj[:, 0] / np.maximum(zj, 1e-6) * K[0, 0] + K[0, 2]
    vj = -pj[:, 1] / np.maximum(zj, 1e-6) * K[1, 1] + K[1, 2]
    return uj.reshape(h, w), vj.reshape(h, w), zj.reshape(h, w)


def view_consistency(orig_root, ref_root, sid, ks, pairs=None, pose_source="frames", food_only=False):
    fr = cam_matrices(orig_root, sid, pose_source)
    ims = {}
    for k in ks:
        a, b = load(orig_root, sid, k), load(ref_root, sid, k)
        seg, dep = raw(orig_root, sid, k)
        ims[k] = (a, b, seg, dep)
    sc = W0 / fr[0]["imageWidth"]
    sig = (1.5, 4.0, 12.0)
    acc = {s: {"orig": [], "ref": []} for s in sig}
    valid = []
    pairs = pairs or [(i, j) for i in ks for j in ks if i != j and abs(i - j) <= 2]
    for i, j in pairs:
        ai, bi, si, di = ims[i]
        aj, bj, sj, dj = ims[j]
        uj, vj, zj = reproject(fr[i], fr[j], di, dj, sc)
        h, w = di.shape
        ok = (di > 0.05) & ((si > 0) & (si <= 252) if food_only else (si > 0)) & (uj >= 1) & (uj < w - 2) & (vj >= 1) & (vj < h - 2) & (zj > 0.05)
        m = np.zeros_like(ok)
        zs = cv2.remap(dj, uj.astype(np.float32), vj.astype(np.float32), cv2.INTER_NEAREST)
        ss = cv2.remap(sj, uj.astype(np.float32), vj.astype(np.float32), cv2.INTER_NEAREST)
        # visible in j (depth agreement), same label, and away from depth edges
        gz = cv2.Laplacian(di, cv2.CV_32F)
        m = ok & (np.abs(zs - zj) < 0.01 * zj) & (ss == si) & (np.abs(gz) < 0.004)
        valid.append(float(m.sum() / max((si > 0).sum(), 1)))
        for s in sig:
            for nm, (xi, xj) in (("orig", (ai, aj)), ("ref", (bi, bj))):
                fi_ = cv2.GaussianBlur(xi.astype(np.float32), (0, 0), s)
                fj_ = cv2.GaussianBlur(xj.astype(np.float32), (0, 0), s)
                samp = cv2.remap(fj_, uj.astype(np.float32), vj.astype(np.float32), cv2.INTER_LINEAR)
                acc[s][nm].append(np.abs(fi_ - samp).mean(2)[m])
    out = {"valid_frac": float(np.mean(valid))}
    for s in sig:
        o = np.concatenate(acc[s]["orig"])
        r = np.concatenate(acc[s]["ref"])
        out[f"sigma{s}"] = dict(orig=float(o.mean()), ref=float(r.mean()),
                                excess=float(np.sqrt(max((r ** 2).mean() - (o ** 2).mean(), 0))),
                                n=int(o.size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scans", nargs="*")
    ap.add_argument("--frames", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--json")
    ap.add_argument("--pose-source", default="spec", choices=["frames", "spec"])
    ap.add_argument("--food-only", action="store_true", help="view consistency over food pixels only")
    a = ap.parse_args()
    o, r = Path(a.orig), Path(a.ref)
    sids = a.scans or sorted(p.name for p in (r / "scans").iterdir())
    res = {}
    for sid in sids:
        ks = [k for k in a.frames if (r / "scans" / sid / "rgb" / f"{k:06d}.jpg").exists()]
        if not ks:
            continue
        bm, lm = [], []
        for k in ks:
            A, B = load(o, sid, k), load(r, sid, k)
            seg, _ = raw(o, sid, k)
            bm.append(boundary_metrics(A, B, seg))
            lm.append(label_metrics(A, B, seg))
        res[sid] = dict(boundary=bm, label=lm, views=view_consistency(o, r, sid, ks, pose_source=a.pose_source, food_only=a.food_only) if len(ks) > 1 else None)
        print(sid, "done", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"))
    _summary(res)


def _summary(res):
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")
    B = [b for s in res.values() for b in s["boundary"]]
    L = [b for s in res.values() for b in s["label"]]
    print("boundary F  orig %.3f  ref %.3f   recall %.3f -> %.3f   precision %.3f -> %.3f" % (
        mean([b["orig"]["f"] for b in B]), mean([b["ref"]["f"] for b in B]),
        mean([b["orig"]["recall"] for b in B]), mean([b["ref"]["recall"] for b in B]),
        mean([b["orig"]["precision"] for b in B]), mean([b["ref"]["precision"] for b in B])))
    print("edge drift median %.2f px  p90 %.2f px  (@960 wide); edges within 2px: o->r %.3f r->o %.3f" % (
        mean([b["edge_drift_med"] for b in B]), mean([b["edge_drift_p90"] for b in B]),
        mean([b["edge_within2_orig_to_ref"] for b in B]), mean([b["edge_within2_ref_to_orig"] for b in B])))
    print("median dist seg-boundary px -> nearest image edge: orig %.2f  ref %.2f px" % (
        mean([b["orig"]["med_dist_to_edge"] for b in B]), mean([b["ref"]["med_dist_to_edge"] for b in B])))
    for g in ("food", "clutter", "dish", "bg"):
        print("label %-8s mean|dRGB| %.2f" % (g, mean([l[g]["dmean_abs"] for l in L if g in l])))
    print("food instance mean|dRGB| avg %.2f  max-avg %.2f" % (
        mean([l["food_instance_dmean_abs_mean"] for l in L]), mean([l["food_instance_dmean_abs_max"] for l in L])))
    print("lowpass change >25: %.4f  >15: %.4f  mean %.2f" % (
        mean([l["frac_lowpass_change_gt25"] for l in L]), mean([l["frac_lowpass_change_gt15"] for l in L]),
        mean([l["lowpass_change_mean"] for l in L])))
    V = [s["views"] for s in res.values() if s["views"]]
    if V:
        print("views valid frac %.3f" % mean([v["valid_frac"] for v in V]))
        for s in ("sigma1.5", "sigma4.0", "sigma12.0"):
            print("  %-9s |dRGB| orig %.2f  ref %.2f  excess %.2f" % (
                s, mean([v[s]["orig"] for v in V]), mean([v[s]["ref"] for v in V]), mean([v[s]["excess"] for v in V])))


if __name__ == "__main__":
    main()
