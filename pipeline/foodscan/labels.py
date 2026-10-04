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
"""
from __future__ import annotations

from collections.abc import Mapping

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


def visible_indices(frame: Frame, points: np.ndarray, mask_shape: tuple[int, int],
                    depth_tol: float = 0.01, min_conf: int = 2,
                    normals: np.ndarray | None = None, max_grazing_deg: float = 75.0):
    """Points that project inside the image and are not occluded in this frame.

    Returns (indices, u_int, v_int, weight). Occlusion test compares the
    point's camera depth with the LiDAR depth at that pixel; the depth map is
    always 256x192, so pixel coords are computed at that resolution and then
    rescaled to the mask resolution.
    """
    dw, dh = frame.depth_size
    u, v, z = project(points, frame.K_depth, frame.T_wc)
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    inb = (z > 0.05) & (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
    idx = np.nonzero(inb)[0]
    ui, vi, z = ui[idx], vi[idx], z[idx]

    depth = frame.masked_depth(min_conf=min_conf)
    d = depth[vi, ui]
    vis = (d > 0) & (np.abs(d - z) < depth_tol)
    idx, ui, vi, z = idx[vis], ui[vis], vi[vis], z[vis]

    # weight: cosine of viewing angle if normals are given, else 1
    weight = np.ones(len(idx))
    if normals is not None and len(idx):
        cam_pos = frame.T_cw[:3, 3]
        to_cam = cam_pos - points[idx]
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
        cos = np.abs(np.sum(normals[idx] * to_cam, axis=1))
        keep = cos > np.cos(np.deg2rad(max_grazing_deg))
        idx, ui, vi, cos = idx[keep], ui[keep], vi[keep], cos[keep]
        weight = cos

    mh, mw = mask_shape
    mu = np.clip((ui * mw) // dw, 0, mw - 1)
    mv = np.clip((vi * mh) // dh, 0, mh - 1)
    return idx, mu, mv, weight


def lift_masks(scan: Scan, masks: Mapping[int, np.ndarray], points: np.ndarray,
               n_labels: int | None = None, normals: np.ndarray | None = None,
               depth_tol: float = 0.01, count_background: bool = True) -> np.ndarray:
    """Accumulate per-point label votes over all frames that have a mask.

    Returns votes of shape (N, L+1); column 0 is background. With
    `count_background` a visible point under an unlabelled pixel votes for
    background, which lets the plate/table win where the segmenter said nothing.
    """
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
