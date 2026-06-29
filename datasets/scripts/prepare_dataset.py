"""
datasets/scripts/prepare_dataset.py
====================================
Handles download, preprocessing, tokenization and splitting for:
  - OpenI Indiana University Chest X-ray dataset (default, freely available)
  - MIMIC-CXR (requires PhysioNet credentials)

Usage:
    python datasets/scripts/prepare_dataset.py --dataset openI
    python datasets/scripts/prepare_dataset.py --dataset mimic_cxr --data_dir /path/to/mimic
"""

import os
import re
import json
import pickle
import argparse
import logging
import zipfile
import urllib.request
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Special tokens ──────────────────────────────────────────────────────────
PAD_TOKEN  = "<PAD>"
SOS_TOKEN  = "<SOS>"
EOS_TOKEN  = "<EOS>"
UNK_TOKEN  = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ─── Vocabulary ──────────────────────────────────────────────────────────────
class Vocabulary:
    """Word-level vocabulary with encode / decode helpers."""

    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._counter: Counter = Counter()

        for tok in SPECIAL_TOKENS:
            self._add(tok)

    # ── private helpers ──────────────────────────────────────────────────────
    def _add(self, word: str) -> None:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\-\/]", " ", text)
        return text.split()

    # ── public API ───────────────────────────────────────────────────────────
    def build(self, reports: List[str]) -> None:
        """Count words across all reports, then keep words >= min_freq."""
        for report in reports:
            for word in self._tokenize(report):
                self._counter[word] += 1
        for word, cnt in self._counter.items():
            if cnt >= self.min_freq:
                self._add(word)
        logger.info("Vocabulary built: %d words (min_freq=%d)", len(self), self.min_freq)

    def encode(self, text: str, max_len: int = 150) -> List[int]:
        tokens = [SOS_IDX]
        for w in self._tokenize(text):
            tokens.append(self.word2idx.get(w, UNK_IDX))
        tokens.append(EOS_IDX)
        tokens = tokens[:max_len]
        tokens += [PAD_IDX] * (max_len - len(tokens))
        return tokens

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if skip_special and word in SPECIAL_TOKENS:
                if word == EOS_TOKEN:
                    break
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Vocabulary saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── OpenI Dataset Processor ─────────────────────────────────────────────────
class OpenIProcessor:
    """
    Indiana University Chest X-ray Dataset.
    XML reports + PNG images from:
    https://openi.nlm.nih.gov/faq#collection
    Kaggle mirror: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
    """

    KAGGLE_DATASET = "raddar/chest-xrays-indiana-university"

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.images_dir    = self.processed_dir / "images"
        self.reports_dir   = self.processed_dir / "reports"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        """Attempt download via kaggle CLI (must have kaggle.json configured)."""
        try:
            import subprocess
            dest = str(self.raw_dir)
            os.makedirs(dest, exist_ok=True)
            logger.info("Downloading OpenI from Kaggle …")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.KAGGLE_DATASET,
                 "--unzip", "-p", dest],
                check=True
            )
            logger.info("Download complete.")
        except Exception as exc:
            logger.warning("Kaggle download failed: %s", exc)
            logger.warning(
                "Please manually download the dataset from:\n"
                "  https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university\n"
                "and extract it to: %s", self.raw_dir
            )

    def parse_reports(self) -> List[Dict]:
        """Parse XML reports into a list of dicts {uid, impression, findings}."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise RuntimeError("xml.etree.ElementTree not available")

        xml_files = list(self.raw_dir.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {self.raw_dir}")

        records = []
        for xml_path in tqdm(xml_files, desc="Parsing XML reports"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                uid         = root.findtext(".//uId[@id]") or xml_path.stem
                impression  = root.findtext(".//AbstractText[@Label='IMPRESSION']") or ""
                findings    = root.findtext(".//AbstractText[@Label='FINDINGS']") or ""
                image_nodes = root.findall(".//parentImage")
                image_ids   = [n.get("id", "") for n in image_nodes]

                # Combine impression + findings as the full report
                report = " ".join(filter(None, [findings.strip(), impression.strip()]))
                if report and image_ids:
                    records.append({
                        "uid":       uid,
                        "report":    self._clean_report(report),
                        "image_ids": image_ids,
                    })
            except Exception as exc:
                logger.debug("Skip %s: %s", xml_path.name, exc)

        logger.info("Parsed %d valid records from %d XML files", len(records), len(xml_files))
        return records

    @staticmethod
    def _clean_report(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"xxxx", "unknown", text, flags=re.IGNORECASE)
        return text.strip()

    def preprocess_images(self, records: List[Dict], target_size: int = 224) -> List[Dict]:
        """Copy & resize images; keep only records where the image exists."""
        all_img_dirs = list(self.raw_dir.rglob("images")) + [self.raw_dir]
        valid_records = []

        for rec in tqdm(records, desc="Preprocessing images"):
            matched = None
            for img_id in rec["image_ids"]:
                for search_dir in all_img_dirs:
                    for ext in (".png", ".jpg", ".jpeg"):
                        candidate = search_dir / (img_id + ext)
                        if candidate.exists():
                            matched = candidate
                            break
                    if matched:
                        break
                if matched:
                    break

            if matched is None:
                continue

            dest = self.images_dir / (rec["uid"] + "_" + matched.stem + ".png")
            if not dest.exists():
                try:
                    img = Image.open(matched).convert("RGB")
                    img = img.resize((target_size, target_size), Image.LANCZOS)
                    img.save(dest, "PNG")
                except Exception as exc:
                    logger.debug("Image error %s: %s", matched, exc)
                    continue

            valid_records.append({**rec, "image_path": str(dest)})

        logger.info("Valid records with images: %d", len(valid_records))
        return valid_records

    def build_annotations(
        self,
        records:      List[Dict],
        vocab:        Vocabulary,
        train_split:  float = 0.80,
        val_split:    float = 0.10,
        max_seq_len:  int   = 150,
        out_path:     str   = "datasets/processed/annotations.json",
    ) -> Dict:
        """Build train/val/test splits and return the annotations dict."""
        np.random.seed(42)
        indices = np.random.permutation(len(records))
        n_train = int(len(records) * train_split)
        n_val   = int(len(records) * val_split)

        splits = {
            "train": indices[:n_train].tolist(),
            "val":   indices[n_train:n_train + n_val].tolist(),
            "test":  indices[n_train + n_val:].tolist(),
        }

        annotations = {
            "vocab_size": len(vocab),
            "max_seq_len": max_seq_len,
            "splits": splits,
            "records": [
                {
                    "uid":        r["uid"],
                    "image_path": r["image_path"],
                    "report":     r["report"],
                    "tokens":     vocab.encode(r["report"], max_seq_len),
                }
                for r in records
            ],
        }

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(annotations, f, indent=2)

        logger.info(
            "Annotations saved → %s  (train=%d  val=%d  test=%d)",
            out_path, len(splits["train"]), len(splits["val"]), len(splits["test"])
        )
        return annotations


# ─── CLI entry-point ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare radiology dataset")
    parser.add_argument("--dataset",     default="openI",
                        choices=["openI", "mimic_cxr"],
                        help="Which dataset to process")
    parser.add_argument("--data_dir",    default="datasets/raw/",
                        help="Path to raw dataset directory")
    parser.add_argument("--out_dir",     default="datasets/processed/",
                        help="Output directory for processed data")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--max_seq_len", type=int, default=150)
    parser.add_argument("--min_freq",    type=int, default=3)
    parser.add_argument("--download",    action="store_true",
                        help="Attempt Kaggle download")
    args = parser.parse_args()

    if args.dataset == "openI":
        processor = OpenIProcessor(args.data_dir, args.out_dir)

        if args.download:
            processor.download()

        logger.info("=== Step 1/4: Parsing reports ===")
        records = processor.parse_reports()

        logger.info("=== Step 2/4: Preprocessing images ===")
        records = processor.preprocess_images(records, target_size=args.image_size)

        logger.info("=== Step 3/4: Building vocabulary ===")
        vocab = Vocabulary(min_freq=args.min_freq)
        vocab.build([r["report"] for r in records])
        vocab.save(os.path.join(args.out_dir, "vocab.pkl"))

        logger.info("=== Step 4/4: Building annotations ===")
        processor.build_annotations(
            records,
            vocab,
            max_seq_len=args.max_seq_len,
            out_path=os.path.join(args.out_dir, "annotations.json"),
        )

    elif args.dataset == "mimic_cxr":
        logger.warning(
            "MIMIC-CXR requires PhysioNet credentials.\n"
            "Please download from: https://physionet.org/content/mimic-cxr/2.0.0/\n"
            "Then adapt this script to your local path layout."
        )

    logger.info("Dataset preparation complete ✓")


if __name__ == "__main__":
    main()
