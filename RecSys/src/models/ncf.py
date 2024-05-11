from collections import OrderedDict

import torch
from torch import nn

from ..utils import MLP


class NCF(nn.Module):

    def __init__(
        self, num_users, num_items, n_mlp_layers=4, mlp_layers_dim=32, mlp_kwargs={}
    ):  # pylint:disable=dangerous-default-value
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

        self.final = nn.Linear(in_features=mlp_layers_dim, out_features=1)
        self.final = MLP(mlp_layers_dim, 1, False, False, False)

    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.hstack((user_embedded, item_embedded))

        # Pass through dense layer
        vector = self.mlp(vector)

        # Output layer
        pred = torch.sigmoid(self.final(vector))

        return pred
