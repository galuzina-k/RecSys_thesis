from recsys_metrics import hit_rate, normalized_dcg


def reccomendation_report(preds, target, k=5, reduction="mean"):
    """
    Report all the metrics at once."""

    return {
        "Hit rate @ K": hit_rate(preds, target, k, reduction),
        "NDCG @ K": normalized_dcg(preds, target, k, reduction),
    }
