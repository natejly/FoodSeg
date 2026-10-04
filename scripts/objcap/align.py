"""Align an Object Capture mesh into the scan's ARKit world frame (metric, +Y up).

Fits a similarity (Umeyama) from Object Capture camera centres to ARKit camera
centres, so the mesh lines up with frames.jsonl poses for mask lifting. The
fitted scale is an independent check of Object Capture's metric scale.

    python scripts/objcap/align.py SCAN_DIR MODEL.obj MODEL.poses.json OUT.ply
"""
import json
import sys

import numpy as np
import open3d as o3d


def umeyama(src, dst):
    """s, R, t minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def main(scan, model, poses, out):
    arkit = {r["index"]: np.array(r["transform"])[:3, 3]
             for r in map(json.loads, open(f"{scan}/frames.jsonl"))}
    oc = {int(k): np.array(v)[:3, 3] for k, v in json.load(open(poses)).items()}
    ids = sorted(arkit.keys() & oc.keys())
    src, dst = np.array([oc[i] for i in ids]), np.array([arkit[i] for i in ids])
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    print(f"{len(ids)} frames  scale {s:.4f}  camera residual median {np.median(res)*1000:.1f} mm, "
          f"max {res.max()*1000:.1f} mm")

    mesh = o3d.io.read_triangle_mesh(model, enable_post_processing=True)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    mesh.transform(T)
    o3d.io.write_triangle_mesh(out, mesh)
    print("aligned ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
