"""
datasets/scripts/prepare_chest_xray_dataset.py
================================================
Converts the Kaggle Chest X-ray Pneumonia dataset into the annotations.json
format used by this project's training pipeline.

Dataset structure expected:
    datasets/chest_xray/train/NORMAL/*.jpeg
    datasets/chest_xray/train/PNEUMONIA/*.jpeg
    datasets/chest_xray/val/NORMAL/*.jpeg
    datasets/chest_xray/val/PNEUMONIA/*.jpeg
    datasets/chest_xray/test/NORMAL/*.jpeg
    datasets/chest_xray/test/PNEUMONIA/*.jpeg

Usage:
    python datasets/scripts/prepare_chest_xray_dataset.py
    python datasets/scripts/prepare_chest_xray_dataset.py --data_dir datasets/chest_xray --out_dir datasets/processed
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.scripts.prepare_dataset import Vocabulary

random.seed(42)

# ---------------------------------------------------------------------------
# Radiology report templates per class
# ---------------------------------------------------------------------------

NORMAL_REPORTS = [
    "the lungs are clear bilaterally no focal consolidation pleural effusion or pneumothorax is seen the cardiomediastinal silhouette is within normal limits no acute cardiopulmonary abnormality",
    "lungs are clear without evidence of focal airspace disease no pleural effusion or pneumothorax cardiac silhouette is normal in size and contour",
    "no acute cardiopulmonary process the lungs are well expanded and clear the cardiac silhouette is normal the osseous structures are intact",
    "clear lungs bilaterally no pneumonia consolidation or atelectasis cardiac size normal no pleural effusion",
    "the lungs are hyperinflated but clear no focal consolidation or pleural effusion cardiac silhouette within normal limits bony thorax is intact",
    "lungs are clear no airspace disease identified heart size is normal mediastinum is within normal limits no pleural effusion or pneumothorax",
    "no acute pulmonary disease lungs clear bilateral no pleural effusion cardiac silhouette and mediastinum within normal limits",
    "clear lung fields bilaterally no infiltrates no effusions no pneumothorax cardiac silhouette normal in size",
    "no evidence of active pulmonary disease the lungs are clear bilaterally cardiac silhouette is normal in size no pleural effusion identified",
    "bilateral lung fields are clear no focal opacity or consolidation cardiac silhouette is within normal limits no bony abnormality detected",
]

PNEUMONIA_BACTERIAL_REPORTS = [
    "right lower lobe consolidation is present consistent with bacterial pneumonia no pleural effusion cardiac silhouette is normal",
    "there is a left lower lobe opacity consistent with lobar pneumonia no pleural effusion the cardiac silhouette is within normal limits",
    "dense consolidation is identified in the right middle lobe consistent with bacterial pneumonia heart size is normal",
    "patchy airspace opacity in the left lower lobe consistent with pneumonia or atelectasis no pleural effusion cardiac silhouette normal",
    "there is consolidation in the right lower and middle lobe consistent with pneumonia cardiac silhouette is within normal limits",
    "lobar consolidation is present in the left lower lobe consistent with bacterial pneumonia no pleural effusion identified",
    "right upper lobe consolidation is seen consistent with pneumonia cardiac silhouette is normal no pleural effusion",
    "bilateral lower lobe opacities are present consistent with multifocal bacterial pneumonia cardiac silhouette is normal",
    "there is dense airspace consolidation in the right lower lobe consistent with pneumonia no pleural effusion cardiac size normal",
    "left lower lobe consolidation consistent with pneumonia the right lung is clear no pleural effusion cardiac silhouette is normal",
]

PNEUMONIA_VIRAL_REPORTS = [
    "bilateral interstitial opacities are present consistent with viral pneumonia cardiac silhouette is mildly enlarged",
    "patchy bilateral airspace opacities are present consistent with multifocal pneumonia or viral infection",
    "bilateral peribronchial infiltrates are noted consistent with viral or atypical pneumonia cardiac silhouette is normal",
    "diffuse bilateral interstitial markings are increased consistent with viral pneumonitis or early pulmonary edema",
    "bilateral patchy ground glass opacities are identified consistent with viral pneumonia cardiac silhouette is within normal limits",
    "reticulonodular opacities are present bilaterally consistent with interstitial pneumonia or viral infection",
    "bilateral perihilar infiltrates are noted consistent with viral or atypical pneumonia heart size is normal",
    "there are bilateral interstitial infiltrates consistent with viral pneumonia no pleural effusion cardiac silhouette is normal",
    "patchy opacities in bilateral lung fields consistent with viral or atypical pneumonia cardiac size is normal",
    "bilateral airspace disease with ground glass opacities consistent with viral pneumonia cardiac silhouette is normal in size",
]


def get_report(label: str, filename: str) -> str:
    """Assign a report based on the label and filename hint."""
    if label == "NORMAL":
        return random.choice(NORMAL_REPORTS)
    # PNEUMONIA — try to guess bacteria vs virus from filename
    fname_lower = filename.lower()
    if "bacteria" in fname_lower:
        return random.choice(PNEUMONIA_BACTERIAL_REPORTS)
    elif "virus" in fname_lower:
        return random.choice(PNEUMONIA_VIRAL_REPORTS)
    else:
        # Mix both
        return random.choice(PNEUMONIA_BACTERIAL_REPORTS + PNEUMONIA_VIRAL_REPORTS)


def resize_and_copy(src: Path, dst: Path, size: int = 224) -> bool:
    """Resize image to size x size RGB and save. Returns True on success."""
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, "PNG")
        return True
    except Exception as e:
        print(f"  [SKIP] {src.name}: {e}")
        return False


def process_split(
    split_dir: Path,
    out_img_dir: Path,
    split_name: str,
    image_size: int = 224,
    max_per_class: int = None,
) -> list:
    """Process one split (train/val/test) and return list of record dicts."""
    records = []
    for label in ["NORMAL", "PNEUMONIA"]:
        class_dir = split_dir / label
        if not class_dir.exists():
            print(f"  [WARN] {class_dir} not found, skipping.")
            continue

        images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpeg", ".jpg", ".png"}
        ])

        if max_per_class:
            random.shuffle(images)
            images = images[:max_per_class]

        print(f"  {split_name}/{label}: {len(images)} images")

        for img_path in tqdm(images, desc=f"  {split_name}/{label}", leave=False):
            uid = f"{split_name}_{label}_{img_path.stem}"
            dst = out_img_dir / f"{uid}.png"

            if not dst.exists():
                if not resize_and_copy(img_path, dst, image_size):
                    continue

            report = get_report(label, img_path.name)
            records.append({
                "uid":        uid,
                "image_path": str(dst),
                "report":     report,
                "label":      label,
            })

    return records


def build_annotations(
    records: list,
    vocab: Vocabulary,
    max_seq_len: int,
    train_uids: set,
    val_uids: set,
    test_uids: set,
) -> dict:
    uid_to_idx = {r["uid"]: i for i, r in enumerate(records)}

    train_idx = [uid_to_idx[r["uid"]] for r in records if r["uid"] in train_uids]
    val_idx   = [uid_to_idx[r["uid"]] for r in records if r["uid"] in val_uids]
    test_idx  = [uid_to_idx[r["uid"]] for r in records if r["uid"] in test_uids]

    ann_records = []
    for rec in records:
        ann_records.append({
            "uid":        rec["uid"],
            "image_path": rec["image_path"],
            "report":     rec["report"],
            "label":      rec["label"],
            "tokens":     vocab.encode(rec["report"], max_seq_len),
        })

    return {
        "vocab_size":  len(vocab),
        "max_seq_len": max_seq_len,
        "splits": {
            "train": train_idx,
            "val":   val_idx,
            "test":  test_idx,
        },
        "records": ann_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle Chest X-ray dataset")
    parser.add_argument("--data_dir",    default="datasets/chest_xray",
                        help="Root of the chest_xray folder (contains train/val/test)")
    parser.add_argument("--out_dir",     default="datasets/processed",
                        help="Output directory for processed data")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--max_seq_len", type=int, default=150)
    parser.add_argument("--min_freq",    type=int, default=1)
    parser.add_argument("--max_train_per_class", type=int, default=None,
                        help="Cap training images per class (e.g. 1000 for quick runs)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    img_dir  = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Processing images from: {data_dir}")

    train_records = process_split(data_dir / "train", img_dir, "train",
                                  args.image_size, args.max_train_per_class)
    val_records   = process_split(data_dir / "val",   img_dir, "val",
                                  args.image_size)
    test_records  = process_split(data_dir / "test",  img_dir, "test",
                                  args.image_size)

    all_records = train_records + val_records + test_records
    print(f"\nTotal records: {len(all_records)}  "
          f"(train={len(train_records)}, val={len(val_records)}, test={len(test_records)})")

    print("\n[2/4] Building vocabulary...")
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build([r["report"] for r in all_records])
    vocab_path = str(out_dir / "vocab.pkl")
    vocab.save(vocab_path)
    print(f"  Vocabulary: {len(vocab)} tokens saved -> {vocab_path}")

    print("\n[3/4] Encoding tokens and building annotations...")
    train_uids = {r["uid"] for r in train_records}
    val_uids   = {r["uid"] for r in val_records}
    test_uids  = {r["uid"] for r in test_records}

    annotations = build_annotations(
        all_records, vocab, args.max_seq_len, train_uids, val_uids, test_uids
    )

    ann_path = str(out_dir / "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    splits = annotations["splits"]
    print(f"  Annotations saved -> {ann_path}")
    print(f"  Train / Val / Test : {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

    print("\n[4/4] Summary")
    print(f"  Images   : {len(all_records)}  ->  {img_dir}")
    print(f"  Vocab    : {len(vocab)} tokens")
    print(f"  Labels   : NORMAL / PNEUMONIA (bacteria + virus)")
    print("\n[OK] Dataset ready!")
    print("  Next: python quick_train.py --epochs 20 --batch_size 16")


if __name__ == "__main__":
    main()
