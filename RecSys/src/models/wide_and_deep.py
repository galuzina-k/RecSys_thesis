"""
Blyat nu eto polnaya huina. Ele nashel infy voobsche v inete. 

Po itogy:
Deep part: MLP([embed(userId), embed(movieId)])
Wide part: one-hot genres + user feats

"""

from collections import OrderedDict

import torch
from torch import nn


class wideAndDeep(nn.Module):
    def __init__(
        self, num_users, num_items, cross_feats_dim, mlp_layer_sizes=[16, 64, 32, 8]
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users, embedding_dim=mlp_layer_sizes[0] // 2
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items, embedding_dim=mlp_layer_sizes[0] // 2
        )

        self.mlp = OrderedDict()
        for i in range(1, len(mlp_layer_sizes)):
            self.mlp[f"MLP_layer_{i}"] = nn.Linear(
                mlp_layer_sizes[i - 1], mlp_layer_sizes[i]
            )
            self.mlp[f"Activation_layer_{i}"] = nn.ReLU()
        self.mlp = nn.Sequential(self.mlp)

        self.cross = nn.Linear(cross_feats_dim, mlp_layer_sizes[-1])

        self.final = nn.Linear(mlp_layer_sizes[-1] + self.cross.out_features, 1)

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
