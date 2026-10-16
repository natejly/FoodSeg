"""Extract MetaFood3D video frames + food masks for the MFP3D split (single-image benchmark).

    python -m eval.mf3d_frames ~/Downloads/_MetaFood3D_new_RGBD_videos.tar.gz \
        ../data/metafood3d/mfp3d_h5 --out ../data/metafood3d/frames

The MFP3D release (github.com/jingema99/MFP3D, Google-Drive h5 files) fixes the split:
401 train / 95 test objects, one image each (`images/<cat>/<obj>/rgb/0.jpg`, i.e. frame 0
of the capture video). This streams the RGBD_videos archive once and keeps, for every
object in either split, frames 0, 10, ..., 190 of `original/` and `masks/`, resized to
336x448 (portrait, like the source 1440x1920). Test scoring uses frame 0 only, as MFP3D;
the extra frames of *training* objects are augmentation. Writes <out>/<cat>__<obj>/<k>.jpg
(RGB) and <k>_m.png (mask), plus split.json.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import h5py
import numpy as np

W, H = 336, 448
FRAMES = set(range(0, 200, 10))


def split_objects(h5dir: Path) -> dict[str, list[str]]:
    out = {}
    for s in ("train", "test"):
        with h5py.File(h5dir / f"gt_{s}.h5", "r") as f:
            paths = [p.decode() for p in f["image_path"][:]]
        out[s] = sorted({"/".join(p.split("/")[-4:-2]) for p in paths})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", type=Path, nargs="+", help="RGBD_videos archive(s); objects already extracted are skipped")
    ap.add_argument("h5dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("../data/metafood3d/frames"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split = split_objects(args.h5dir)
    # archive names differ from the split's in letter case for some objects (Almond_1 vs almond_1)
    want = {o.lower(): o for o in split["train"] + split["test"]
            if not (args.out / o.replace("/", "__") / "0.jpg").exists()}
    (args.out / "split.json").write_text(json.dumps(split, indent=1))
    print(f"{len(split['train'])} train / {len(split['test'])} test objects", flush=True)

    n = 0
    seen = set()
    for tar in args.tar:
      with tarfile.open(tar, "r|gz") as t:
        print(f"reading {tar.name}, {len(want) - len(seen)} objects to find", flush=True)
        for m in t:
            if not m.isfile():
                continue
            parts = m.name.split("/")                     # RGBD_videos/<cat>/<obj>/<kind>/<k>.jpg
            if len(parts) != 5 or parts[3] not in ("original", "masks"):
                continue
            obj = want.get(f"{parts[1]}/{parts[2]}".lower())
            k = Path(parts[4]).stem
            if obj is None or not k.isdigit() or int(k) not in FRAMES:
                continue
            buf = np.frombuffer(t.extractfile(m).read(), np.uint8)
            d = args.out / obj.replace("/", "__")
            d.mkdir(exist_ok=True)
            if parts[3] == "original":
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                cv2.imwrite(str(d / f"{k}.jpg"), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
            else:
                msk = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(d / f"{k}_m.png"), (cv2.resize(msk, (W, H), interpolation=cv2.INTER_AREA) > 127)
                            .astype(np.uint8) * 255)
            n += 1
            if obj not in seen:
                seen.add(obj)
                if len(seen) % 50 == 0:
                    print(f"  {len(seen)} objects, {n} files", flush=True)
    missing = sorted(set(want.values()) - seen)
    print(f"done: {len(seen)} objects, {n} files; missing {len(missing)}: {missing[:10]}")


if __name__ == "__main__":
    main()
