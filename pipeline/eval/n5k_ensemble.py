"""Average eval.n5k_train runs (geometric mean of predictions) and score them.

    python -m eval.n5k_ensemble ../data/datasets/Nutrition5k ../results/n5k/train/a ../results/n5k/train/b

Runs must share the val split (same --seed); val is reported when every run saved
val_predictions.csv, so ensembles can be chosen on val before looking at test.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval.n5k_train import TARGETS
from eval.nutrition5k import read_metadata


def load(path: Path) -> dict[str, np.ndarray]:
    return {r["dish_id"]: np.array([float(r[f"pred_{t}"]) for t in TARGETS]) for r in csv.DictReader(open(path))}


def score(preds: list[dict], meta: dict) -> str:
    ids = sorted(set.intersection(*[set(p) for p in preds]))
    p = np.expm1(np.mean([np.log1p(np.stack([q[d] for d in ids])) for q in preds], 0))
    true = np.array([[meta[d][t] for t in TARGETS] for d in ids])
    mae = np.abs(p - true).mean(0)
    return f"n={len(ids)}  " + "  ".join(f"{t} {m:.1f} ({100 * m / true[:, i].mean():.1f}%)"
                                         for i, (t, m) in enumerate(zip(TARGETS, mae)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("runs", type=Path, nargs="+")
    args = ap.parse_args()
    meta = read_metadata(args.root)
    if all((r / "val_predictions.csv").exists() for r in args.runs):
        print("VAL  " + score([load(r / "val_predictions.csv") for r in args.runs], meta))
    print("TEST " + score([load(r / "test_predictions.csv") for r in args.runs], meta))


if __name__ == "__main__":
    main()
