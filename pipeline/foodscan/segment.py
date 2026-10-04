"""SAM 3 / SAM 3.1 text-prompted video segmentation of a FoodScan capture.

    python -m foodscan.segment SCAN_DIR --names "white rice,grilled chicken,broccoli" [--sam31]

Writes SCAN_DIR/masks/NNNNNN.png (uint8 label map at video resolution, 0 = background,
1..K = instance ids consistent across frames) and SCAN_DIR/masks/instances.json
({id: {"name", "concept_index", "mean_score"}}). `labels.py` consumes these directly.

Protocol (from facebookresearch/sam3 source, Sept 2026): a text prompt resets the
session state, so each concept is one propagation; earlier concepts win overlaps.
Frames are read from SCAN_DIR/rgb (the predictor sorts by int(stem), which works
with the app's %06d names). Needs a Linux GPU box with the sam3 repo installed and
gated weights approved (`hf auth login`); this file cannot run on the Mac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def segment_scan(scan_dir: Path, names: list[str], thresh: float = 0.5, sam31: bool = False,
                 out_dir: Path | None = None) -> dict:
    frames_dir = scan_dir / "rgb"
    out_dir = out_dir or scan_dir / "masks"
    frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    n = len(frame_files)
    index_of = {int(p.stem): i for i, p in enumerate(frame_files)}  # frame_index -> position
    stems = [p.stem for p in frame_files]

    if sam31:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        pred = build_sam3_multiplex_video_predictor()
    else:
        from sam3.model_builder import build_sam3_video_predictor
        pred = build_sam3_video_predictor(gpus_to_use=[0], async_loading_frames=True)

    sid = pred.handle_request(dict(type="start_session", resource_path=str(frames_dir),
                                   offload_video_to_cpu=True))["session_id"]
    labels: dict[int, np.ndarray] = {}
    meta: dict[int, dict] = {}
    offset = 0
    for ci, name in enumerate(names):
        pred.handle_request(dict(type="reset_session", session_id=sid))
        pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=name,
                                 output_prob_thresh=thresh))
        max_id, scores = -1, {}
        for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                                 propagation_direction="forward", start_frame_index=0,
                                                 evict_cached_frame_outputs=True)):
            o, fi = r["outputs"], r["frame_index"]
            for oid, p, m in zip(o["out_obj_ids"], o["out_probs"], o["out_binary_masks"]):
                gid = int(oid) + offset + 1
                lab = labels.get(fi)
                if lab is None:
                    lab = labels[fi] = np.zeros(m.shape, np.uint8)
                lab[m & (lab == 0)] = gid
                scores.setdefault(gid, []).append(float(p))
                max_id = max(max_id, int(oid))
        for gid, s in scores.items():
            meta[gid] = {"name": name, "concept_index": ci, "mean_score": float(np.mean(s))}
        offset += max_id + 1
        print(f"concept '{name}': {len(scores)} instances")

    pred.handle_request(dict(type="close_session", session_id=sid))
    if hasattr(pred, "shutdown"):
        pred.shutdown()

    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = None, None
    for fi in range(n):
        lab = labels.get(fi)
        if lab is None:
            if h is None:
                w, h = Image.open(frame_files[0]).size
            lab = np.zeros((h, w), np.uint8)
        Image.fromarray(lab).save(out_dir / f"{stems[fi]}.png")
    (out_dir / "instances.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_masks(scan_dir: Path) -> tuple[dict[int, np.ndarray], dict]:
    """Read masks written by segment_scan into the {frame_index: label_map} form labels.py wants."""
    mdir = Path(scan_dir) / "masks"
    masks = {int(p.stem): np.asarray(Image.open(p)) for p in mdir.glob("*.png")}
    meta = json.loads((mdir / "instances.json").read_text())
    return masks, {int(k): v for k, v in meta.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--names", required=True, help="comma-separated food names (from the VLM)")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--sam31", action="store_true", help="use SAM 3.1 Object Multiplex predictor")
    args = ap.parse_args()
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    meta = segment_scan(Path(args.scan_dir), names, thresh=args.thresh, sam31=args.sam31)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
