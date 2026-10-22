"""Evaluate a trained checkpoint on the signer-independent validation split.

Reports top-1/top-5, per-class accuracy, confusion matrix and the most
confused sign pairs. The split is rebuilt from the checkpoint's own seed and
subset, so the numbers are comparable to the val metrics logged in training.

Examples:
  python -m src.evaluate                              # checkpoints/bilstm.pt
  python -m src.evaluate --full-val --save-dir results/bilstm
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import DATA_DIR, SignDataset, split_indices
from src.train import MODELS

SIGN_MAP = Path("data/raw/sign_to_prediction_index_map.json")


def load_sign_names(path: Path = SIGN_MAP) -> list[str]:
    sign_to_idx = json.load(open(path))
    return [s for s, _ in sorted(sign_to_idx.items(), key=lambda kv: kv[1])]


@torch.no_grad()
def predict_top5(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels (N,), top-5 predictions (N, 5))."""
    model.eval()
    ys, tops = [], []
    for x, y in loader:
        tops.append(model(x.to(device)).topk(5, dim=1).indices.cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(tops).numpy()


def most_confused(cm: np.ndarray, k: int) -> list[tuple[int, int, int]]:
    """Top-k off-diagonal (true, pred, count) entries."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    pairs = [(*np.unravel_index(i, off.shape), off.flat[i]) for i in flat]
    return [(int(t), int(p), int(c)) for t, p, c in pairs if c > 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/bilstm.pt"))
    parser.add_argument("--full-val", action="store_true",
                        help="all val-signer samples, ignoring the checkpoint's --subset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=15, help="rows for worst classes / pairs")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="write confusion_matrix.npy and per_class.csv here")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    subset = None if args.full_val else train_args["subset"]

    p = np.load(DATA_DIR / "participants.npy")
    _, va, val_signers = split_indices(p, subset=subset, seed=train_args["seed"])
    if "val_signers" in ckpt:  # _last.pt has none; split_indices() recomputes them from the seed
        assert val_signers.tolist() == ckpt["val_signers"], \
            f"val signers {val_signers.tolist()} != checkpoint {ckpt['val_signers']}"

    X = np.load(DATA_DIR / "X.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y.npy").astype(np.int64)
    val_ds = SignDataset(np.asarray(X[va]), y[va], train=False)  # only val rows into RAM
    loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[ckpt.get("model_name", train_args["model"])]().to(device)
    model.load_state_dict(ckpt["model"])

    names = load_sign_names()
    n_classes = len(names)
    labels, top5 = predict_top5(model, loader, device)
    preds = top5[:, 0]

    logged = (f" | logged val top1 {ckpt['val_top1']:.3f} top5 {ckpt['val_top5']:.3f}"
              if "val_top1" in ckpt else "")
    print(f"checkpoint {args.ckpt} | epoch {ckpt['epoch']}{logged}")
    print(f"val signers {val_signers.tolist()} | samples {len(labels)} | "
          f"{'full val' if subset is None else f'subset {subset}'}")
    print(f"top1 {np.mean(preds == labels):.3f} | "
          f"top5 {np.mean((top5 == labels[:, None]).any(1)):.3f}")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    support = cm.sum(1)
    per_class = pd.DataFrame({
        "sign": names, "support": support,
        "accuracy": np.divide(np.diag(cm), support,
                              out=np.full(n_classes, np.nan), where=support > 0),
    })
    present = per_class[per_class.support > 0]
    print(f"\nper-class accuracy over {len(present)} classes: "
          f"mean {present.accuracy.mean():.3f} | median {present.accuracy.median():.3f} | "
          f"zero-accuracy classes {(present.accuracy == 0).sum()}")

    print(f"\nworst {args.top_k} classes:")
    for _, r in present.sort_values(["accuracy", "support"]).head(args.top_k).iterrows():
        print(f"  {r.sign:<15} acc {r.accuracy:.3f}  (n={r.support})")

    print("\nmost confused pairs (true -> predicted):")
    for t, pr, c in most_confused(cm, args.top_k):
        print(f"  {names[t]:<15} -> {names[pr]:<15} {c:4d}  ({c / support[t]:.0%} of {names[t]})")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.save_dir / "confusion_matrix.npy", cm)
        per_class.to_csv(args.save_dir / "per_class.csv", index_label="class_idx")
        print(f"\nsaved confusion_matrix.npy, per_class.csv -> {args.save_dir}")


if __name__ == "__main__":
    main()
