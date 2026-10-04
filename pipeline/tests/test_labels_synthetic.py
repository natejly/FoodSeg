"""Lift synthetic 2D masks (box vs table, derived from the depth itself) onto the
fused cloud and check that the points above the plane come out labelled 'box'.

    python tests/test_labels_synthetic.py [SCAN_DIR] [--stride 3] [--flip 0.0]
--flip corrupts that fraction of mask pixels per frame to test robustness of the vote.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodscan.fuse import fuse  # noqa: E402
from foodscan.labels import decide, lift_masks  # noqa: E402
from foodscan.loader import Frame, Scan  # noqa: E402
from foodscan.volume import fit_support_plane  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])
BOX = 1  # label id for the box


def synthetic_mask(frame: Frame, height_thresh: float = 0.004) -> np.ndarray:
    """Per-pixel label from the frame's own depth: unproject, height above z=0."""
    d = frame.depth(smoothed=True)              # noise-free copy in the synthetic scan
    h, w = d.shape
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    K = frame.K_depth
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1).reshape(-1, 3)
    pw = (frame.T_cw[:3, :3] @ pc.T).T + frame.T_cw[:3, 3]
    label = (pw[:, 2] > height_thresh).astype(np.uint8).reshape(h, w) * BOX
    label[d <= 0] = 0
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", nargs="?")
    ap.add_argument("--stride", type=int, default=3, help="run 'segmenter' on every Nth frame")
    ap.add_argument("--flip", type=float, default=0.0, help="fraction of mask pixels to corrupt")
    ap.add_argument("--voxel", type=float, default=0.0015)
    args = ap.parse_args()

    if args.scan_dir:
        scan_dir = Path(args.scan_dir)
    else:
        scan_dir = Path(tempfile.mkdtemp()) / "synth"
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_synthetic_scan.py"),
                        str(scan_dir), "--frames", "24", "--noise", "0.0005"],
                       check=True, stdout=subprocess.DEVNULL)
    scan = Scan(scan_dir)

    mesh, pcd = fuse(scan, voxel=args.voxel, stride=1)
    pts = np.asarray(pcd.points)
    pts = pts[np.linalg.norm(pts[:, :2], axis=1) < 0.2]
    plane = fit_support_plane(pts, up_hint=UP)
    truth = (plane.height(pts) > 0.004).astype(int) * BOX

    rng = np.random.default_rng(0)
    masks = {}
    for f in scan.good_frames(args.stride):
        m = synthetic_mask(f)
        if args.flip > 0:
            flip = rng.random(m.shape) < args.flip
            m = np.where(flip, BOX - m, m).astype(np.uint8)   # swap box<->table
        masks[f.index] = m

    votes = lift_masks(scan, masks, pts, n_labels=1, depth_tol=0.01)
    labels, margin = decide(votes)

    seen = labels >= 0
    acc = np.mean(labels[seen] == truth[seen])
    box_recall = np.mean(labels[truth == BOX] == BOX)
    box_prec = np.mean(truth[labels == BOX] == BOX) if np.any(labels == BOX) else float("nan")
    print(f"frames used {len(masks)}, points {len(pts)}, seen {seen.mean()*100:.1f}%")
    print(f"accuracy {acc*100:.2f}%   box recall {box_recall*100:.2f}%   box precision {box_prec*100:.2f}%")
    print(f"mean vote margin: box {margin[truth == BOX].mean():.3f}, table {margin[truth == 0].mean():.3f}")
    low = margin[seen] < 0.7
    print(f"low-confidence points (<70% agreement): {low.mean()*100:.2f}%")


if __name__ == "__main__":
    main()
