"""Sanity check for extracted landmarks in data/landmarks/."""
from pathlib import Path

import numpy as np

LM_DIR = Path("data/landmarks")
# Must match the point order used in src/extract.py
PARTS = {"left_hand": slice(0, 21), "pose": slice(21, 54), "right_hand": slice(54, 75)}
HANDS = np.r_[0:21, 54:75]
SAMPLE = 2000
SEED = 0


def main() -> None:
    X = np.load(LM_DIR / "X.npy", mmap_mode="r")
    y = np.load(LM_DIR / "y.npy")
    p = np.load(LM_DIR / "participants.npy")

    print(f"X {X.shape} {X.dtype} | y {y.shape} {y.dtype} | participants {p.shape}")
    assert len(X) == len(y) == len(p), "Length mismatch between X, y, participants"

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(X), size=min(SAMPLE, len(X)), replace=False))
    xs = np.asarray(X[idx], dtype=np.float32)  # (N, T, 75, 3)

    print(f"\n[sample of {len(idx)} videos]")
    print(f"NaN: {np.isnan(xs).sum()}  Inf: {np.isinf(xs).sum()}")

    missing = np.all(xs == 0, axis=-1)  # (N, T, 75) point fully zero
    print(f"Missing-point fraction overall: {missing.mean():.3f}")
    for name, s in PARTS.items():
        print(f"  {name:<11}{missing[:, :, s].mean():.3f}")

    no_hands = missing[:, :, HANDS].all(axis=(1, 2))
    print(f"Videos with no hand detected in any frame: {no_hands.mean():.3%}")

    vals = xs[~np.all(xs == 0, axis=-1)]  # present points only, (M, 3)
    q = np.percentile(vals, [0.1, 50, 99.9], axis=0)
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: p0.1={q[0, i]:+.2f}  median={q[1, i]:+.2f}  p99.9={q[2, i]:+.2f}")

    counts = np.bincount(y, minlength=250)
    print(f"\nClasses: {len(np.unique(y))} | per-class min {counts.min()} max {counts.max()}")
    signers, s_counts = np.unique(p, return_counts=True)
    print(f"Signers: {len(signers)} | per-signer min {s_counts.min()} max {s_counts.max()}")


if __name__ == "__main__":
    main()