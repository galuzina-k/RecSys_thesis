import torch
from recsys_metrics import expected_popularity_complement, hit_rate, normalized_dcg

from .bahavioral_metrics import intraListDiversity


def reccomendation_report(
    preds, target, pred_items, similarities, popularities, k=5, reduction="mean"
):
    """
    Report all the metrics at once."""

    return {
        f"Hit rate @ {k}": hit_rate(preds, target, k, reduction),
        f"NDCG @ {k}": normalized_dcg(preds, target, k, reduction),
        "Diversity (ILD)": intraListDiversity(pred_items, similarities, k, reduction),
        "Novelty (EPC)": expected_popularity_complement(
            preds, popularities[pred_items.to(torch.long)], k
        ),
    }
