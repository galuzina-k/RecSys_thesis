from collections import OrderedDict

import torch
from torch import nn

from ..utils import MLP


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        mf_dim=32,
        n_mlp_layers=4,
        mlp_layers_dim=32,
        mlp_kwargs={},
    ):  # pylint:disable=unused-argument
        super().__init__()

        self.mf_user_embed = nn.Embedding(num_users, mf_dim)
        self.mf_item_embed = nn.Embedding(num_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(num_users, mlp_layers_dim // 2)
        self.mlp_item_embed = nn.Embedding(num_items, mlp_layers_dim // 2)
        # self.dropout = dropout

        self.mlp = OrderedDict()
        for i in range(n_mlp_layers):
            self.mlp[f"MLP_layer_{i}"] = MLP(
                mlp_layers_dim, mlp_layers_dim, **mlp_kwargs
            )
        self.mlp = nn.Sequential(self.mlp)

        self.final = MLP(
            mlp_layers_dim + mf_dim, 1, activation=False, dropout=False, batchnorm=False
        )

    def forward(self, user, item):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.hstack((xmlpu, xmlpi))
        xmlp = self.mlp(xmlp)

        x = torch.hstack((xmf, xmlp))
        x = self.final(x)
        x = torch.sigmoid(x)
        return x
