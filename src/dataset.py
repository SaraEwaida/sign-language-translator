"""Dataset loading, signer-independent split and keypoint augmentation.

Input layout (from extract.py): X (N, 64, 75, 3), points ordered
left_hand 0-20, pose 21-53, right_hand 54-74; 0 = missing landmark.
Each sample is returned as a (64, 225) float32 tensor and an int64 label.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path("data/landmarks")
N_POINTS = 75

# MediaPipe pose left/right pairs (indices within the 33 pose points)
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def _flip_index() -> np.ndarray:
    """new_point[i] = old_point[idx[i]] after a horizontal flip."""
    idx = np.arange(N_POINTS)
    idx[0:21], idx[54:75] = np.arange(54, 75), np.arange(0, 21)  # swap hands
    for a, b in _POSE_PAIRS:
        idx[21 + a], idx[21 + b] = 21 + b, 21 + a
    return idx


FLIP_IDX = _flip_index()


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x: (T, 75, 3) float32, zeros = missing. Returns an augmented array."""
    present = np.any(x != 0, axis=-1, keepdims=True)  # (T, 75, 1)

    if rng.random() < 0.5:  # mirror: simulates a left-handed signer
        x, present = x[:, FLIP_IDX].copy(), present[:, FLIP_IDX]
        x[..., 0] *= -1

    theta = np.deg2rad(rng.uniform(-15, 15))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    x[..., :2] = x[..., :2] @ rot.T
    x *= np.float32(rng.uniform(0.8, 1.2))
    x[..., :2] += rng.uniform(-0.1, 0.1, size=2).astype(np.float32)

    T = len(x)  # temporal crop, then resample back to T frames
    length = max(2, int(round(T * rng.uniform(0.8, 1.0))))
    start = rng.integers(0, T - length + 1)
    idx = np.linspace(start, start + length - 1, T).round().astype(int)
    x, present = x[idx], present[idx]

    return x * present  # keep missing points at exactly 0


class SignDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = False):
        self.X, self.y, self.train = X, y, train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X[i].astype(np.float32)  # float16 in RAM -> float32 copy
        if self.train:
            # seeded from torch so each DataLoader worker gets different randomness
            rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
            x = augment(x, rng)
        return torch.from_numpy(x.reshape(len(x), -1)), torch.tensor(self.y[i], dtype=torch.long)


def signer_split(participants: np.ndarray, n_val: int = 3, seed: int = 42):
    """Hold out n_val whole signers for validation."""
    signers = np.unique(participants)
    rng = np.random.default_rng(seed)
    val_signers = np.sort(rng.choice(signers, size=n_val, replace=False))
    val_mask = np.isin(participants, val_signers)
    return np.where(~val_mask)[0], np.where(val_mask)[0], val_signers


def split_indices(p: np.ndarray, n_val_signers: int = 3,
                  subset: int | None = None, seed: int = 42):
    """Return (train_idx, val_idx, val_signers). subset = total samples to keep."""
    tr, va, val_signers = signer_split(p, n_val_signers, seed)
    if subset:
        rng = np.random.default_rng(seed)
        frac = min(1.0, subset / len(p))
        tr = np.sort(rng.choice(tr, size=max(1, int(len(tr) * frac)), replace=False))
        va = np.sort(rng.choice(va, size=max(1, int(len(va) * frac)), replace=False))

    assert not set(p[tr]) & set(p[va]), "signer leakage between train and val"
    return tr, va, val_signers


def load_splits(data_dir: Path = DATA_DIR, n_val_signers: int = 3,
                subset: int | None = None, seed: int = 42):
    """Return (train_ds, val_ds, val_signers). subset = total samples to keep."""
    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy").astype(np.int64)
    p = np.load(data_dir / "participants.npy")

    tr, va, val_signers = split_indices(p, n_val_signers, subset, seed)

    train_ds = SignDataset(np.asarray(X[tr]), y[tr], train=True)  # loads into RAM
    val_ds = SignDataset(np.asarray(X[va]), y[va], train=False)
    return train_ds, val_ds, val_signers


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_ds, val_ds, val_signers = load_splits(subset=4000)
    print(f"train {len(train_ds)} | val {len(val_ds)} | val signers {val_signers.tolist()}")

    xb, yb = next(iter(DataLoader(train_ds, batch_size=32, shuffle=True)))
    print(f"batch x {tuple(xb.shape)} {xb.dtype} | y {tuple(yb.shape)} {yb.dtype}")

    raw = train_ds.X[0].astype(np.float32)
    aug = augment(raw.copy(), np.random.default_rng(0))
    print(f"missing before {np.mean(np.all(raw == 0, -1)):.3f} | "
          f"after aug {np.mean(np.all(aug == 0, -1)):.3f} | NaN {np.isnan(aug).sum()}")