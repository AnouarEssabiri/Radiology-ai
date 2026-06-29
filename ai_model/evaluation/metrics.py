"""
ai_model/evaluation/metrics.py
================================
Evaluation metrics for radiology report generation:
  - BLEU-1/2/3/4
  - ROUGE-L
  - METEOR (if nltk data available)
  - ClinicalBERT-based semantic similarity (optional)

All functions accept:
    hypotheses : List[str]         – generated reports
    references : List[List[str]]   – list of reference lists (BLEU convention)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── BLEU ─────────────────────────────────────────────────────────────────────
def compute_bleu(
    hypotheses: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU-N score using NLTK.
    Returns BLEU-{max_n} (0–1 scale).
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        logger.warning("NLTK not installed. pip install nltk")
        return 0.0

    smoother = SmoothingFunction().method1
    weights  = tuple([1.0 / max_n] * max_n)

    tok_hyps = [h.lower().split() for h in hypotheses]
    tok_refs = [[r.lower().split() for r in ref_list] for ref_list in references]

    score = corpus_bleu(tok_refs, tok_hyps, weights=weights, smoothing_function=smoother)
    return round(score, 4)


def compute_all_bleu(
    hypotheses: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Returns BLEU-1, BLEU-2, BLEU-3, BLEU-4."""
    return {
        f"bleu_{n}": compute_bleu(hypotheses, references, max_n=n)
        for n in range(1, 5)
    }


# ─── ROUGE ────────────────────────────────────────────────────────────────────
def compute_rouge(
    hypotheses: List[str],
    references: List[str],          # single reference per sample
) -> Dict[str, float]:
    """
    Returns ROUGE-1, ROUGE-2, ROUGE-L F1 scores (averaged over corpus).
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed. pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for k in totals:
            totals[k] += scores[k].fmeasure

    n = max(len(hypotheses), 1)
    return {k: round(v / n, 4) for k, v in totals.items()}


# ─── METEOR ──────────────────────────────────────────────────────────────────
def compute_meteor(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """
    Corpus-level METEOR score using NLTK.
    Requires: nltk.download('wordnet') and nltk.download('omw-1.4')
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        scores = [
            meteor_score([ref.split()], hyp.split())
            for hyp, ref in zip(hypotheses, references)
        ]
        return round(sum(scores) / max(len(scores), 1), 4)
    except Exception as exc:
        logger.warning("METEOR computation failed: %s", exc)
        return 0.0


# ─── Exact match accuracy ─────────────────────────────────────────────────────
def compute_accuracy(
    hypotheses: List[str],
    references: List[str],
    token_level: bool = False,
) -> float:
    """
    If token_level=True: per-token accuracy (excluding padding).
    Else: exact sentence match.
    """
    if not token_level:
        matches = sum(h.strip() == r.strip() for h, r in zip(hypotheses, references))
        return round(matches / max(len(hypotheses), 1), 4)

    correct = total = 0
    for h, r in zip(hypotheses, references):
        h_toks, r_toks = h.split(), r.split()
        length = max(len(h_toks), len(r_toks))
        for i in range(min(len(h_toks), len(r_toks))):
            correct += int(h_toks[i] == r_toks[i])
            total   += 1
    return round(correct / max(total, 1), 4)


# ─── Full evaluation suite ────────────────────────────────────────────────────
def evaluate_all(
    hypotheses:   List[str],
    references:   List[str],
    compute_met:  bool = True,
) -> Dict[str, float]:
    """
    Run all metrics and return a consolidated dict.

    Args:
        hypotheses  : list of generated reports
        references  : list of ground-truth reports (one per sample)
        compute_met : whether to compute METEOR (requires NLTK data)
    """
    refs_bleu = [[r] for r in references]    # wrap for corpus_bleu

    results = {}
    results.update(compute_all_bleu(hypotheses, refs_bleu))
    results.update(compute_rouge(hypotheses, references))
    results["token_accuracy"] = compute_accuracy(hypotheses, references, token_level=True)

    if compute_met:
        results["meteor"] = compute_meteor(hypotheses, references)

    return results


# ─── Per-sample report evaluation ────────────────────────────────────────────
def evaluate_sample(hypothesis: str, reference: str) -> Dict[str, float]:
    """Evaluate a single generated report against a reference."""
    return evaluate_all([hypothesis], [reference])


# ─── Print formatted results ─────────────────────────────────────────────────
def print_results(results: Dict[str, float], title: str = "Evaluation Results") -> None:
    print(f"\n{'=' * 45}")
    print(f"  {title}")
    print(f"{'=' * 45}")
    for metric, value in results.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:<18} {value:.4f}  {bar}")
    print(f"{'=' * 45}\n")


if __name__ == "__main__":
    # Smoke test
    hyps = [
        "the lungs are clear no pleural effusion or pneumothorax is seen",
        "cardiomegaly is present with mild pulmonary edema",
    ]
    refs = [
        "lungs are clear no pneumothorax or pleural effusion",
        "mild cardiomegaly with pulmonary vascular congestion",
    ]
    results = evaluate_all(hyps, refs)
    print_results(results)
