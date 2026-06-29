"""
datasets/scripts/generate_synthetic_dataset.py
================================================
Generates a synthetic chest X-ray dataset for IMMEDIATE testing —
no Kaggle account, no download, no internet needed.

Creates:
  - 500 synthetic grayscale chest X-ray-like images (224x224 PNG)
  - Realistic radiology reports for each image
  - Vocabulary, annotations.json, train/val/test splits

Run:
    python datasets/scripts/generate_synthetic_dataset.py

After this, you can immediately run training and the API.
"""

import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.scripts.prepare_dataset import Vocabulary

# ── Report templates ──────────────────────────────────────────────────────────
NORMAL_REPORTS = [
    "the lungs are clear bilaterally no focal consolidation pleural effusion or pneumothorax is seen the cardiomediastinal silhouette is within normal limits no acute cardiopulmonary abnormality",
    "lungs are clear without evidence of focal airspace disease no pleural effusion or pneumothorax cardiac silhouette is normal in size and contour",
    "no acute cardiopulmonary process the lungs are well expanded and clear the cardiac silhouette is normal the osseous structures are intact",
    "clear lungs bilaterally no pneumonia consolidation or atelectasis cardiac size normal no pleural effusion",
    "the lungs are hyperinflated but clear no focal consolidation or pleural effusion cardiac silhouette within normal limits bony thorax is intact",
    "lungs are clear no airspace disease identified heart size is normal mediastinum is within normal limits no pleural effusion or pneumothorax",
    "no acute pulmonary disease lungs clear bilateral no pleural effusion cardiac silhouette and mediastinum within normal limits",
    "clear lung fields bilaterally no infiltrates no effusions no pneumothorax cardiac silhouette normal in size",
]

ABNORMAL_REPORTS = [
    "there is a left lower lobe opacity consistent with pneumonia or atelectasis no pleural effusion the cardiac silhouette is mildly enlarged",
    "bilateral interstitial opacities are present consistent with pulmonary edema cardiac silhouette is enlarged consistent with cardiomegaly",
    "small right pleural effusion is noted with associated basilar atelectasis the lungs are otherwise clear",
    "there is mild cardiomegaly with pulmonary vascular congestion consistent with congestive heart failure no focal airspace disease",
    "right lower lobe consolidation is present consistent with pneumonia no pleural effusion cardiac silhouette is normal",
    "patchy bilateral airspace opacities are present consistent with multifocal pneumonia or aspiration",
    "there is evidence of left pneumothorax with partial collapse of the left lung mediastinum is shifted to the right",
    "bilateral pleural effusions are noted larger on the right side with associated compressive atelectasis",
    "there is a right upper lobe opacity with associated volume loss consistent with atelectasis or mass",
    "diffuse bilateral interstitial markings are increased consistent with pulmonary fibrosis or chronic lung disease",
    "there is mild hyperinflation of the lungs consistent with emphysema or chronic obstructive pulmonary disease flattened diaphragms",
    "a left lower lobe atelectasis is present the right lung is clear no pleural effusion cardiac silhouette normal",
]

ALL_REPORTS = NORMAL_REPORTS + ABNORMAL_REPORTS


# ── Synthetic X-ray image generator ──────────────────────────────────────────
def make_synthetic_xray(size: int = 224, seed: int = 0, pathology: str = None) -> Image.Image:
    """
    Generate a grayscale image that loosely resembles a chest X-ray:
    - Dark background (air)
    - Bright oval rib cage region
    - Darker lung fields (left/right)
    - Bright spine/mediastinum
    - Subtle rib-like structures
    - Optional opacity blob (abnormality simulation) correlated with pathology
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    # ── Chest wall (bright ellipse) ──────────────────────────────────────
    for y in range(size):
        for x in range(size):
            dx = (x - cx) / (size * 0.46)
            dy = (y - cy) / (size * 0.54)
            if dx*dx + dy*dy < 1.0:
                img[y, x] = 0.25 + 0.15 * rng.random()

    # ── Lung fields (darker ovals) ────────────────────────────────────────
    for side in [-1, 1]:
        lx = cx + side * size * 0.18
        for y in range(size):
            for x in range(size):
                dx = (x - lx) / (size * 0.20)
                dy = (y - cy) / (size * 0.34)
                if dx*dx + dy*dy < 1.0:
                    img[y, x] = 0.06 + 0.05 * rng.random()

    # ── Spine / mediastinum (bright vertical band) ────────────────────────
    for y in range(int(cy * 0.4), int(cy * 1.7)):
        width = int(size * 0.07)
        for x in range(cx - width, cx + width):
            if 0 <= x < size:
                img[y, x] = max(img[y, x], 0.5 + 0.15 * rng.random())

    # ── Diaphragm arches ─────────────────────────────────────────────────
    for side in [-1, 1]:
        ax = cx + side * size * 0.15
        for x in range(max(0, int(ax - size*0.22)), min(size, int(ax + size*0.22))):
            arch_y = int(cy + size*0.28 + side*0.0*(x - ax) + 0.06*((x - ax)**2) / size)
            for dy in range(-4, 5):
                if 0 <= arch_y + dy < size:
                    img[arch_y + dy, x] = max(img[arch_y + dy, x], 0.45 + 0.1*rng.random())

    # ── Ribs (faint arcs) ─────────────────────────────────────────────────
    for i in range(7):
        rib_y = cy - size*0.30 + i * size*0.10
        radius = size * (0.32 + i * 0.02)
        for angle in np.linspace(-1.2, 1.2, 120):
            rx = int(cx + radius * np.sin(angle))
            ry = int(rib_y + radius * 0.3 * np.cos(angle))
            for side in [-1, 1]:
                xi = int(cx + side * (rx - cx))
                if 0 <= xi < size and 0 <= ry < size:
                    img[ry, xi] = min(1.0, img[ry, xi] + 0.12 + 0.05*rng.random())

    # ── Pathology simulation (opacity blob or hyperlucency) ───────────────
    if pathology is not None:
        intensity = 0.25 + 0.20 * rng.random()
        # In conventional radiography view, patient's left is on the right side of the image (x > cx)
        if "left lower lobe" in pathology or "left basilar" in pathology or "compressive atelectasis" in pathology:
            blob_cx = cx + int(size * 0.18)
            blob_cy = cy + int(size * 0.15)
            blob_r  = rng.integers(size//12, size//8)
        elif "right lower lobe" in pathology or "right pleural" in pathology or "right side" in pathology:
            blob_cx = cx - int(size * 0.18)
            blob_cy = cy + int(size * 0.15)
            blob_r  = rng.integers(size//12, size//8)
        elif "right upper lobe" in pathology:
            blob_cx = cx - int(size * 0.18)
            blob_cy = cy - int(size * 0.15)
            blob_r  = rng.integers(size//12, size//8)
        elif "left pneumothorax" in pathology:
            # Hyperlucency (darkness) on left side
            blob_cx = cx + int(size * 0.18)
            blob_cy = cy
            blob_r  = rng.integers(size//8, size//6)
            intensity = -0.15
        elif "cardiomegaly" in pathology or "edema" in pathology or "heart size" in pathology or "congestion" in pathology:
            # Cardiomegaly: enlarge cardiac silhouette
            blob_cx = cx
            blob_cy = cy + int(size * 0.05)
            blob_r  = rng.integers(size//7, size//5)
        else:
            # General baseline abnormal blob
            blob_cx = cx + rng.integers(-size//5, size//5)
            blob_cy = cy + rng.integers(-size//6, size//5)
            blob_r  = rng.integers(size//12, size//6)

        for y in range(max(0, blob_cy - blob_r), min(size, blob_cy + blob_r)):
            for x in range(max(0, blob_cx - blob_r), min(size, blob_cx + blob_r)):
                dist = np.sqrt((x - blob_cx)**2 + (y - blob_cy)**2)
                if dist < blob_r:
                    fade = 1.0 - dist / blob_r
                    img[y, x] = np.clip(img[y, x] + intensity * fade, 0.0, 1.0)

    # ── Gaussian blur for realism ─────────────────────────────────────────
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=2.5)

    # ── Normalise and add grain ───────────────────────────────────────────
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)
    img = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode='L').convert('RGB')
    return pil


# ── Main generation ───────────────────────────────────────────────────────────
def generate(
    n_samples:   int = 500,
    image_size:  int = 224,
    max_seq_len: int = 150,
    min_freq:    int = 1,
    out_dir:     str = "datasets/processed",
    train_split: float = 0.80,
    val_split:   float = 0.10,
):
    out_dir   = Path(out_dir)
    img_dir   = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_samples} synthetic chest X-rays ({image_size}x{image_size})...")

    records = []
    for i in tqdm(range(n_samples), desc="Generating images"):
        # 50% chance of a normal report
        is_normal = random.random() > 0.50
        if is_normal:
            report_text = random.choice(NORMAL_REPORTS)
            pathology_name = None
        else:
            report_text = random.choice(ABNORMAL_REPORTS)
            pathology_name = report_text

        # Generate synthetic X-ray image correlated with pathology
        img  = make_synthetic_xray(size=image_size, seed=i, pathology=pathology_name)
        uid  = f"synth_{i:05d}"
        path = str(img_dir / f"{uid}.png")
        img.save(path)

        records.append({"uid": uid, "image_path": path, "report": report_text})

    # ── Build vocabulary ──────────────────────────────────────────────────
    print("\nBuilding vocabulary...")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build([r["report"] for r in records])
    vocab_path = str(out_dir / "vocab.pkl")
    vocab.save(vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens saved -> {vocab_path}")

    # ── Encode and split ──────────────────────────────────────────────────
    print("Encoding tokens and splitting...")
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(records))
    n_train = int(len(records) * train_split)
    n_val   = int(len(records) * val_split)

    splits = {
        "train": indices[:n_train].tolist(),
        "val":   indices[n_train:n_train + n_val].tolist(),
        "test":  indices[n_train + n_val:].tolist(),
    }

    ann_records = []
    for rec in records:
        ann_records.append({
            "uid":        rec["uid"],
            "image_path": rec["image_path"],
            "report":     rec["report"],
            "tokens":     vocab.encode(rec["report"], max_seq_len),
        })

    annotations = {
        "vocab_size":  len(vocab),
        "max_seq_len": max_seq_len,
        "splits":      splits,
        "records":     ann_records,
    }

    ann_path = str(out_dir / "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n[OK] Synthetic dataset ready!")
    print(f"  Images      : {n_samples}  ->  {img_dir}")
    print(f"  Vocabulary  : {len(vocab)} tokens  ->  {vocab_path}")
    print(f"  Annotations : {ann_path}")
    print(f"  Train / Val / Test : {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"\nYou can now run:")
    print(f"  python ai_model/training/trainer.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=500)
    parser.add_argument("--image_size", type=int,   default=224)
    parser.add_argument("--out_dir",    type=str,   default="datasets/processed")
    args = parser.parse_args()

    generate(n_samples=args.n, image_size=args.image_size, out_dir=args.out_dir)
