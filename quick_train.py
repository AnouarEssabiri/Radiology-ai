"""
quick_train.py
==============
Fast 5-epoch training run using the synthetic dataset.
Called by windows_fix.bat when no checkpoint exists yet.
Also useful for quick testing:
    python quick_train.py
    python quick_train.py --epochs 20 --batch_size 16
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.scripts.dataloader import build_dataloaders
from ai_model.models.radiology_model import build_model
from ai_model.training.trainer import LabelSmoothingLoss


def run(epochs: int = 5, batch_size: int = 8, lr: float = 1e-4):
    # ── Config ────────────────────────────────────────────────────────────
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Always use random weights (no internet download needed)
    cfg["model"]["pretrained_encoder"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Epochs     : {epochs}")
    print(f"Batch size : {batch_size}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dl, val_dl, _ = build_dataloaders(
        annotations_path="datasets/processed/annotations.json",
        vocab_path        ="datasets/processed/vocab.pkl",
        batch_size        =batch_size,
        num_workers       =0,          # Windows: keep 0 to avoid multiprocessing issues
        pin_memory        =False,
    )

    with open("datasets/processed/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    print(f"Vocab size : {len(vocab)}")
    print(f"Train      : {len(train_dl)} batches")
    print()

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_model(cfg["model"], vocab_size=len(vocab)).to(device)
    criterion = LabelSmoothingLoss(len(vocab), smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    print(f"Parameters : {model.count_parameters():,}\n")

    best_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tokens[:, :-1])
            loss   = criterion(logits, tokens[:, 1:])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_dl)

        # ── Quick val loss ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["image"].to(device)
                toks = batch["tokens"].to(device)
                val_loss += criterion(model(imgs, toks[:, :-1]), toks[:, 1:]).item()
        avg_val = val_loss / max(len(val_dl), 1)

        print(f"Epoch {epoch:>2}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        # ── Save best ──────────────────────────────────────────────────────
        if avg_val < best_loss:
            best_loss = avg_val
            os.makedirs("ai_model/checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "vocab_size":  len(vocab),
                    "val_loss":    avg_val,
                },
                "ai_model/checkpoints/best_model.pth",
            )
            print(f"           [OK] Best checkpoint saved (val={avg_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print("Checkpoint -> ai_model/checkpoints/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
