from collections import OrderedDict

import torch
from torch import nn

from ..utils import MLP


class NCF(nn.Module):

    def __init__(
        self, num_users, num_items, mlp_layer_sizes=[16, 32, 64, 32]
    ):  # pylint:disable=dangerous-default-value
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

        self.final = nn.Linear(in_features=mlp_layer_sizes[-1], out_features=1)

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


class NCF_new(nn.Module):

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
        for i in range(1, n_mlp_layers):
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


class NeuMF(nn.Module):
    def __init__(
        self, nb_users, nb_items, mf_dim, mlp_layer_sizes, dropout=0
    ):  # pylint:disable=unused-argument

        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError("u dummy, mlp_layer_sizes[0] % 2 != 0")
        super().__init__()

        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)
        # self.dropout = dropout

        self.mlp = OrderedDict()
        for i in range(1, len(mlp_layer_sizes)):
            self.mlp[f"MLP_layer_{i}"] = nn.Linear(
                mlp_layer_sizes[i - 1], mlp_layer_sizes[i]
            )
            self.mlp[f"Activation_layer_{i}"] = nn.ReLU()
        self.mlp = nn.Sequential(self.mlp)

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        # self.mf_user_embed.weight.data.normal_(0.0, 0.01)
        # self.mf_item_embed.weight.data.normal_(0.0, 0.01)
        # self.mlp_user_embed.weight.data.normal_(0.0, 0.01)
        # self.mlp_item_embed.weight.data.normal_(0.0, 0.01)

    def forward(self, user, item, sigmoid=True):
        xmfu = self.mf_user_embed(user)
        xmfi = self.mf_item_embed(item)
        xmf = xmfu * xmfi

        xmlpu = self.mlp_user_embed(user)
        xmlpi = self.mlp_item_embed(item)
        xmlp = torch.hstack((xmlpu, xmlpi))
        xmlp = self.mlp(xmlp)

        x = torch.hstack((xmf, xmlp))
        x = self.final(x)
        if sigmoid:
            x = torch.sigmoid(x)
        return x
