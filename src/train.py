"""Train a sign classifier with a signer-independent validation split.

Examples:
  python -m src.train --subset 4000 --epochs 2                        # smoke test
  python -u -m src.train --epochs 30 --resume 2>&1 | tee -a train_bilstm.log
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import load_splits
from src.models.lstm import BiLSTMClassifier
from src.models.tcn import TCNClassifier

MODELS = {"bilstm": BiLSTMClassifier, "tcn": TCNClassifier}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()  # no smoothing: interpretable val loss
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        top = logits.topk(5, dim=1).indices
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y[:, None]).any(dim=1).sum().item()
        n += len(y)
    return loss_sum / n, top1 / n, top5 / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="bilstm")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=None, help="total samples, for quick tests")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", action="store_true", help="continue from <out-dir>/<model>_last.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, val_signers = load_splits(subset=args.subset, seed=args.seed)
    print(f"device {device} | train {len(train_ds)} | val {len(val_ds)} | "
          f"val signers {val_signers.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
                            num_workers=args.num_workers)

    model = MODELS[args.model]().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{args.model}.pt"
    last_path = args.out_dir / f"{args.model}_last.pt"
    best_top1, start_epoch = -1.0, 1

    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        for key in ("model", "epochs", "batch_size", "lr", "subset", "seed"):
            if state["args"][key] != getattr(args, key):
                raise SystemExit(f"--{key.replace('_', '-')} differs from the saved run: "
                                 f"{state['args'][key]} vs {getattr(args, key)}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        best_top1, start_epoch = state["best_top1"], state["epoch"] + 1
        print(f"resumed from {last_path}: starting at epoch {start_epoch}, best top1 {best_top1:.3f}")
    elif args.resume:
        print(f"no {last_path} found, starting from scratch")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler, device)
        va_loss, top1, top5 = evaluate(model, val_loader, device)

        improved = top1 > best_top1
        if improved:
            best_top1 = top1
            torch.save({
                "model": model.state_dict(), "model_name": args.model,
                "args": vars(args) | {"out_dir": str(args.out_dir)},
                "epoch": epoch, "val_top1": top1, "val_top5": top5,
                "val_signers": val_signers.tolist(),
            }, ckpt_path)

        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} top1 {top1:.3f} top5 {top5:.3f} | "
              f"{time.time() - t0:.0f}s{'  *saved' if improved else ''}")

        tmp_path = last_path.with_suffix(".tmp")  # write then rename: safe if interrupted
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_top1": best_top1,
            "args": vars(args) | {"out_dir": str(args.out_dir)},
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        }, tmp_path)
        tmp_path.replace(last_path)

    print(f"best val top1 {best_top1:.3f} -> {ckpt_path}")


if __name__ == "__main__":
    main()