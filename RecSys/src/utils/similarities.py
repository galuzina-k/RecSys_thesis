import pandas as pd
import torch
from torch import Tensor


def computeCosineSimilarities(
    df_train: pd.DataFrame, user_col, item_col, num_items
) -> Tensor:

    df_ind_count = (
        df_train[[item_col]].assign(cnt=1).groupby(by=item_col, as_index=False).sum()
    )
    num_i = torch.ones(num_items)
    num_i[df_ind_count[item_col].values] = torch.tensor(
        df_ind_count.cnt.values, dtype=torch.float
    )

    df_pairwise = (
        df_train.loc[:, [user_col, item_col]]
        .rename(columns={item_col: "i1"})
        .merge(
            (df_train.loc[:, [user_col, item_col]].rename(columns={item_col: "i2"})),
            on=[user_col],
            how="inner",
        )
        .loc[:, ["i1", "i2"]]
        .assign(cnt=1)
        .groupby(by=["i1", "i2"], as_index=False)
        .sum()
    )
    similariy = torch.zeros((num_items, num_items))
    similariy[df_pairwise.i1.values, df_pairwise.i2.values] = torch.tensor(
        df_pairwise["cnt"].values, dtype=torch.float
    )
    similariy = similariy / torch.sqrt(num_i * num_i.reshape(-1, 1))

    return similariy
