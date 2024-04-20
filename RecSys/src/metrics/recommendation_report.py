import torch
from recsys_metrics import hit_rate, normalized_dcg, expected_popularity_complement
from .bahavioral_metrics import intraListDiversity


def reccomendation_report(
    preds, target, pred_items, similarities, popularities, k=5, reduction="mean"
):
    """
    Report all the metrics at once."""

    return {
        "Hit rate @ K": hit_rate(preds, target, k, reduction),
        "NDCG @ K": normalized_dcg(preds, target, k, reduction),
        "Diversity (ILD)": intraListDiversity(pred_items, similarities, k, reduction),
        "Novelty (EPC)": expected_popularity_complement(
            preds, popularities[pred_items.to(torch.long)], k
        ),
    }
