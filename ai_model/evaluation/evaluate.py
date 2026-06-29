"""
ai_model/evaluation/evaluate.py
=================================
Standalone evaluation script.
Loads the best checkpoint and runs full evaluation on the test set.

Usage:
    python ai_model/evaluation/evaluate.py
    python ai_model/evaluation/evaluate.py --config config/config.yaml --subset 200
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_model.models.radiology_model import build_model
from ai_model.evaluation.metrics import evaluate_all, print_results
from datasets.scripts.dataloader import build_dataloaders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluate")


def run_evaluation(cfg: dict, subset: int = -1) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load vocabulary
    with open(cfg["dataset"]["paths"]["vocab"], "rb") as f:
        vocab = pickle.load(f)

    # Load model
    model = build_model(cfg["model"], vocab_size=len(vocab)).to(device)

    ckpt_path = cfg["training"]["best_model_path"]
    if not Path(ckpt_path).exists():
        logger.warning("No checkpoint found at %s", ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        logger.info("Loaded checkpoint (epoch %s)", ckpt.get("epoch", "?"))

    model.eval()

    # Dataloaders
    _, _, test_dl = build_dataloaders(
        annotations_path=cfg["dataset"]["paths"]["annotations"],
        vocab_path       =cfg["dataset"]["paths"]["vocab"],
        batch_size       =cfg["training"]["batch_size"],
        image_size       =cfg["dataset"]["image_size"],
        max_seq_len      =cfg["dataset"]["max_seq_len"],
        num_workers      =cfg["dataset"]["num_workers"],
    )

    hypotheses, references = [], []
    count = 0

    logger.info("Running inference on test set …")
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            if subset > 0 and count >= subset:
                break

            images = batch["image"].to(device)
            seqs, _ = model.generate(
                images,
                beam_size=cfg["inference"].get("beam_size", 5),
                max_len  =cfg["inference"].get("max_len", 120),
                min_len  =cfg["inference"].get("min_len", 15),
            )

            for seq, ref in zip(seqs, batch["report"]):
                hypotheses.append(vocab.decode(seq.cpu().tolist()))
                references.append(ref)
                count += 1

    logger.info("Evaluated %d samples", len(hypotheses))

    # Compute metrics
    results = evaluate_all(hypotheses, references, compute_met=True)
    print_results(results, title=f"Test Evaluation ({len(hypotheses)} samples)")

    # Show example outputs
    print("\n── Sample Outputs ──────────────────────────────────────────────")
    for i in range(min(3, len(hypotheses))):
        print(f"\n[{i+1}] REFERENCE : {references[i][:100]}…")
        print(f"    GENERATED : {hypotheses[i][:100]}…")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--subset",  type=int, default=-1,
                        help="Evaluate on first N samples (-1 = full test set)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg, subset=args.subset)


if __name__ == "__main__":
    main()
