"""
datasets/scripts/dataloader.py
================================
PyTorch Dataset and DataLoader for chest X-ray → report generation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── Image transforms ────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return torchvision transform pipeline for the given split."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ─── Dataset ─────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Chest X-ray image + tokenised report dataset.

    Expects:
        annotations_path : path to annotations.json produced by prepare_dataset.py
        vocab_path       : path to vocab.pkl
        split            : "train" | "val" | "test"
    """

    def __init__(
        self,
        annotations_path: str,
        vocab_path:        str,
        split:             str  = "train",
        image_size:        int  = 224,
        max_seq_len:       int  = 150,
        transform:         Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.split       = split
        self.max_seq_len = max_seq_len

        # ── Load annotations ─────────────────────────────────────────────
        with open(annotations_path) as f:
            self.annotations = json.load(f)

        # ── Load vocabulary ───────────────────────────────────────────────
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        # ── Subset for this split ─────────────────────────────────────────
        all_records  = self.annotations["records"]
        split_indices = self.annotations["splits"][split]
        self.records  = [all_records[i] for i in split_indices]

        # ── Transforms ───────────────────────────────────────────────────
        self.transform = transform or get_transforms(split, image_size)

    # ── Dataset protocol ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # ── Image ─────────────────────────────────────────────────────────
        image = Image.open(rec["image_path"]).convert("RGB")
        image = self.transform(image)

        # ── Tokens ────────────────────────────────────────────────────────
        tokens = torch.tensor(rec["tokens"], dtype=torch.long)

        return {
            "image":   image,     # (C, H, W)
            "tokens":  tokens,    # (max_seq_len,)
            "report":  rec["report"],
            "uid":     rec["uid"],
        }


# ─── DataLoader factory ──────────────────────────────────────────────────────
def build_dataloaders(
    annotations_path: str,
    vocab_path:        str,
    batch_size:        int  = 32,
    image_size:        int  = 224,
    max_seq_len:       int  = 150,
    num_workers:       int  = 4,
    pin_memory:        bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = ChestXRayDataset(
            annotations_path=annotations_path,
            vocab_path=vocab_path,
            split=split,
            image_size=image_size,
            max_seq_len=max_seq_len,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else batch_size * 2,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders["train"], loaders["val"], loaders["test"]


# ─── Quick sanity-check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="datasets/processed/annotations.json")
    parser.add_argument("--vocab",       default="datasets/processed/vocab.pkl")
    args = parser.parse_args()

    train_dl, val_dl, test_dl = build_dataloaders(
        args.annotations, args.vocab, batch_size=4, num_workers=0
    )
    batch = next(iter(train_dl))
    print("Image  shape:", batch["image"].shape)
    print("Tokens shape:", batch["tokens"].shape)
    print("Sample report:", batch["report"][0][:80])
    print("Dataloader sizes — train:", len(train_dl),
          "val:", len(val_dl), "test:", len(test_dl))
