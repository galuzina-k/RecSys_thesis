from torch import Tensor
import torch
from tqdm import tqdm


def intraListDiversity(rec_items: Tensor, similarities: Tensor, k=20, reduction="mean"):
    num_users = rec_items.shape[0]
    num_rec = k
    d_total = 0
    for i in range(rec_items.shape[0]):
        comb = torch.combinations(rec_items[i, :k], r=2)
        d_total += torch.sum(
            similarities[comb[:, 0].to(torch.long), comb[:, 1].to(torch.long)]
        )

    d_total = d_total / (num_rec * (num_rec - 1))
    if reduction != "mean":
        raise ValueError("Only `mean` reduction is implemented")
    return d_total / num_users
