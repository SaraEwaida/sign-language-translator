"""Convert Google ASL Signs parquet files into fixed-size arrays.

Output (data/landmarks/):
  X.npy            (N, T, 75, 3) float16  left hand 21 + pose 33 + right hand 21
  y.npy            (N,) int16             sign index
  participants.npy (N,) int32             signer id, for signer-independent splits
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/landmarks")
T = 64  # frames per sample

LH, POSE, RH = slice(468, 489), slice(489, 522), slice(522, 543)
L_SHOULDER, R_SHOULDER = 21 + 11, 21 + 12  # indices inside the 75-point layout


def load_frames(path):
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    return df.to_numpy(np.float32).reshape(-1, 543, 3)


def preprocess(frames):
    pts = np.concatenate([frames[:, LH], frames[:, POSE], frames[:, RH]], axis=1)

    shoulders = pts[:, [L_SHOULDER, R_SHOULDER], :2]
    valid = np.isfinite(shoulders).all(axis=(1, 2))
    if valid.any():
        sh = shoulders[valid]
        center = sh.mean(axis=(0, 1))
        scale = np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1).mean()
    else:
        center, scale = np.zeros(2, np.float32), 1.0
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    pts[..., :2] = (pts[..., :2] - center) / scale
    pts[..., 2] = pts[..., 2] / scale

    idx = np.linspace(0, len(pts) - 1, T).round().astype(int)  # resample to T frames
    return np.nan_to_num(pts[idx], nan=0.0)  # missing landmarks (e.g. no hand) -> 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / "train.csv")
    sign_map = json.load(open(RAW / "sign_to_prediction_index_map.json"))

    X = np.lib.format.open_memmap(
        OUT / "X.npy", mode="w+", dtype=np.float16, shape=(len(df), T, 75, 3)
    )
    for i, path in enumerate(tqdm(df["path"], total=len(df))):
        X[i] = preprocess(load_frames(RAW / path))
    X.flush()

    np.save(OUT / "y.npy", df["sign"].map(sign_map).to_numpy(np.int16))
    np.save(OUT / "participants.npy", df["participant_id"].to_numpy(np.int32))
    print("done:", X.shape)


if __name__ == "__main__":
    main()