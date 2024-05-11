"""Basic utility functions for SVD"""

import torch


def split_test_df(df, user_col, item_col, pred_col, target_col):
    """Splits test df with prediction on pred and target"""

    n_obs_per_user = df.loc[df.userId == df.iloc[0][user_col]].shape[0]
    target = torch.zeros(df.userId.nunique(), n_obs_per_user)
    pred = torch.zeros(df.userId.nunique(), n_obs_per_user)
    items = torch.zeros(df.userId.nunique(), n_obs_per_user)

    for i, user in enumerate(sorted(df.userId.unique())):
        target[i] = torch.tensor(
            df.loc[df.userId == user]
            .sort_values(by=[pred_col], ascending=[False])[target_col]
            .values
        )
        pred[i] = torch.tensor(
            df.loc[df.userId == user]
            .sort_values(by=[pred_col], ascending=[False])[pred_col]
            .values
        )
        items[i] = torch.tensor(
            df.loc[df.userId == user]
            .sort_values(by=[pred_col], ascending=[False])[item_col]
            .values
        )

    return pred, target, items
