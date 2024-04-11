"""Basic utility functions"""

import torch


def split_test_df(df):
    """Splits test df with prediction on pred and target"""

    n_obs_per_user = df.loc[df.userId == df.iloc[0]["userId"]].shape[0]
    target = torch.zeros(df.userId.nunique(), n_obs_per_user)
    pred = torch.zeros(df.userId.nunique(), n_obs_per_user)

    for i, user in enumerate(sorted(df.userId.unique())):
        target[i] = torch.tensor(
            df.loc[df.userId == user]
            .sort_values(by=["rating_pred"], ascending=[False])["action"]
            .values
        )
        pred[i] = torch.tensor(
            df.loc[df.userId == user]
            .sort_values(by=["rating_pred"], ascending=[False])["rating_pred"]
            .values
        )

    return pred, target
