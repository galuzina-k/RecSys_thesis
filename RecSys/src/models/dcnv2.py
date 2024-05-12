from collections import OrderedDict

import torch
from torch import nn

from ..utils import MLP


class DCNv2(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_numeric_feats,
        cat_feature_vocab,
        l=4,
        embedding_dim=5,
        n_mlp_layers=4,
        mlp_layers_dim=32,
        mlp_kwargs={},
    ):

        super().__init__()

        self.l = l

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Define numeric embeddings
        self.numerical_embeddings = nn.ModuleList()
        for _ in range(num_numeric_feats):
            self.numerical_embeddings.append(
                MLP(1, embedding_dim, activation=False, dropout=False, batchnorm=False)
            )

        # Define categorical embeddings
        self.categorical_embeddings = nn.ModuleList()
        for dim in cat_feature_vocab:
            self.categorical_embeddings.append(nn.Embedding(dim, embedding_dim))

        # Define deep part (MLP)
        self.mlp = OrderedDict()
        self.mlp["MLP_layer_0"] = MLP(
            (2 + num_numeric_feats + len(cat_feature_vocab)) * embedding_dim,
            mlp_layers_dim,
        )

        for i in range(1, n_mlp_layers):
            self.mlp[f"MLP_layer_{i}"] = MLP(
                mlp_layers_dim, mlp_layers_dim, **mlp_kwargs
            )
        self.mlp = nn.Sequential(self.mlp)

        # Define cross networks (Wx + b is implemented by MLP)
        d = (2 + num_numeric_feats + len(cat_feature_vocab)) * embedding_dim
        self.feature_crossing = nn.ModuleList()
        for i in range(l):
            self.feature_crossing.append(
                MLP(d, d, activation=False, batchnorm=False, dropout=False)
            )

        self.final = MLP(
            d + mlp_layers_dim, 1, activation=False, batchnorm=False, dropout=False
        )

    def forward(self, user_input, item_input, numeric_feats, categorical_feats):
        # Embed users and items
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Encode numeric
        num_vec = torch.zeros(0, dtype=torch.float)
        for i, emb in enumerate(self.numerical_embeddings):
            num_vec = torch.hstack((num_vec, emb(numeric_feats[:, [i]])))

        # Encode categorical
        cat_vec = torch.zeros(0, dtype=torch.float)
        for i, emb in enumerate(self.categorical_embeddings):
            cat_vec = torch.hstack((cat_vec, emb(categorical_feats[:, i])))

        total_embed = torch.hstack((user_embedded, item_embedded, num_vec, cat_vec))
        x_0 = total_embed
        x = x_0

        # cross
        for i in range(self.l):
            x = x_0 * self.feature_crossing[i](x) + x

        # dense
        x_dense = self.mlp(total_embed)

        x_final = torch.hstack((x, x_dense))
        out = torch.sigmoid(self.final(x_final))
        return out
