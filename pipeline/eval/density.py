"""Per-ingredient density for volume -> mass, fitted on Nutrition5k train dishes.

One global density cannot fit leafy greens and rice (implied density spans ~10x
on N5k). N5k gives every dish's ingredient list with grams, so fit a specific
volume s_i (cm^3 per g, the inverse density) per ingredient from

    V_d / M_d  ~=  sum_i f_di s_i          f_di = ingredient i's mass fraction in dish d

where V_d is whatever geometric measure is being scored (volume in cm^3, or
mask area in cm^2; s_i is then cm^2 per g). The fit is ridge regression toward
the global median s0, so ingredients seen in a few dishes stay near s0 and
unseen ones fall back to it, with Huber reweighting because single-frame
volumes have heavy-tailed errors.

Two ways to use it at test time, both oracles on the food identity (not on
total mass):
    "frac"   true ingredient names and true mass fractions
    "names"  true names only; fractions from each ingredient's typical grams
             on train (what a VLM naming step could supply)
Predicted mass = V / sum_i f_i s_i.
"""
from __future__ import annotations

import numpy as np


class IngredientDensity:
    def __init__(self, lam: float = 5.0, huber: float = 1.0, iters: int = 10):
        self.lam, self.huber, self.iters = lam, huber, iters

    def _design(self, ingrs: list[list[tuple[str, float]]], how: str) -> np.ndarray:
        X = np.zeros((len(ingrs), len(self.vocab) + 1))      # last column: unseen ingredients
        for r, items in enumerate(ingrs):
            if how == "frac":
                w = {n: g for n, g in items}
            else:
                w = {n: self.typical_g.get(n, self.global_g) for n, _ in items}
            tot = sum(w.values())
            if tot <= 0:
                X[r, -1] = 1.0
                continue
            for n, g in w.items():
                X[r, self.vocab.get(n, len(self.vocab))] += g / tot
        return X

    def fit(self, V: np.ndarray, M: np.ndarray, ingrs: list[list[tuple[str, float]]]) -> "IngredientDensity":
        ok = (V > 1) & (M > 0)
        V, M = V[ok], M[ok]
        ingrs = [x for x, k in zip(ingrs, ok) if k]
        names = sorted({n for items in ingrs for n, _ in items})
        self.vocab = {n: i for i, n in enumerate(names)}
        grams: dict[str, list[float]] = {}
        for items in ingrs:
            for n, g in items:
                grams.setdefault(n, []).append(g)
        self.typical_g = {n: float(np.median(g)) for n, g in grams.items()}
        self.global_g = float(np.median([g for gs in grams.values() for g in gs]))

        e = V / M
        self.s0 = float(np.median(e))
        X = self._design(ingrs, "frac")
        k = X.shape[1]
        # residuals are scaled by s0 so lam and huber are unitless
        w = np.ones(len(e))
        for _ in range(self.iters):
            Xw = X * w[:, None]
            A = Xw.T @ X + self.lam * np.eye(k)
            b = Xw.T @ e + self.lam * self.s0
            s = np.linalg.solve(A, b)
            r = np.abs(X @ s - e) / self.s0
            w = np.where(r <= self.huber, 1.0, self.huber / np.maximum(r, 1e-12))
        s[-1] = self.s0
        self.s = np.clip(s, 0.1 * self.s0, 10 * self.s0)
        return self

    def specific_volume(self, ingrs: list[list[tuple[str, float]]], how: str) -> np.ndarray:
        return self._design(ingrs, how) @ self.s

    def predict_mass(self, V: np.ndarray, ingrs: list[list[tuple[str, float]]], how: str) -> np.ndarray:
        return V / self.specific_volume(ingrs, how)


def choose_lambda(V: np.ndarray, M: np.ndarray, ingrs, how: str,
                  grid=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0), folds: int = 5, seed: int = 0) -> float:
    """lam with the lowest k-fold mass MAE on the training dishes."""
    idx = np.random.default_rng(seed).permutation(len(V))
    best, best_err = grid[0], np.inf
    for lam in grid:
        err = 0.0
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            m = IngredientDensity(lam).fit(V[tr], M[tr], [ingrs[i] for i in tr])
            err += np.abs(m.predict_mass(V[te], [ingrs[i] for i in te], how) - M[te]).sum()
        if err < best_err:
            best, best_err = lam, err
    return best
