from collections import OrderedDict

import torch
from torch import nn

from ..utils import MLP


class wideAndDeep(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        cross_feats_dim,
        n_mlp_layers,
        mlp_layers_dim,
        mlp_kwargs={},
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users, embedding_dim=mlp_layers_dim // 2
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items, embedding_dim=mlp_layers_dim // 2
        )

        self.mlp = OrderedDict()
        for i in range(n_mlp_layers):
            self.mlp[f"MLP_layer_{i}"] = MLP(
                mlp_layers_dim, mlp_layers_dim, **mlp_kwargs
            )
        self.mlp = nn.Sequential(self.mlp)

        self.cross = MLP(
            cross_feats_dim,
            mlp_layers_dim,
            activation=False,
            dropout=False,
            batchnorm=False,
        )

        self.final = MLP(
            2 * mlp_layers_dim,
            1,
            activation=False,
            dropout=False,
            batchnorm=False,
        )

    def forward(self, user_input, item_input, cross_input):
        # Deep part
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        deep = torch.hstack((user_embedded, item_embedded))
        deep = self.mlp(deep)

        # Wide part
        wide = self.cross(cross_input)

        out = torch.hstack((wide, deep))

        # Output layer
        pred = torch.sigmoid(self.final(out))
        return pred
