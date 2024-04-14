# import torch
# from torch import nn


# class DeepFM(nn.Module):
#     """
#     DeepFM: A Neural Network for recommendations with both Low and High-Order Feature Interactions.

#     Parameters
#     ----------
#     num_numeric_feats : int
#         Number of numerical features.
#     cat_feature_vocab : list of int
#         List of vocabulary sizes for each categorical feature.
#     embedding_dim : int, optional
#         Dimension of embeddings. Default is 5.
#     mlp_layer_sizes : list of int, optional
#         List of layer sizes for the MLP component. Default is [16, 32, 64, 1].

#     Attributes
#     ----------
#     numerical_embeddings : nn.ModuleList
#         List of linear layers for numerical features.
#     categorical_embeddings : nn.ModuleList
#         List of embedding layers for categorical features.
#     mlp : nn.Sequential
#         Sequential module containing the MLP layers.
#     fm_linear : nn.Linear
#         Linear layer for the FM component.
#     final_sigmoid : nn.Sigmoid
#         Sigmoid activation function for the final output.
#     embedding_dim : int
#         Dimension of embeddings.
#     cat_feature_vocab : list of int
#         List of vocabulary sizes for each categorical feature.

#     """

#     def __init__(
#         self,
#         num_users,
#         num_items,
#         num_numeric_feats,
#         cat_feature_vocab,
#         embedding_dim=5,
#         mlp_layer_sizes=[16, 32, 64, 1],
#     ):
#         super().__init__()

#         # user and movie embeddings
#         self.user_embedding = nn.Embedding(
#             num_embeddings=num_users, embedding_dim=embedding_dim
#         )
#         self.item_embedding = nn.Embedding(
#             num_embeddings=num_items, embedding_dim=embedding_dim
#         )

#         # Define numeric embeddings
#         self.numerical_embeddings = nn.ModuleList()
#         for _ in range(num_numeric_feats):
#             self.numerical_embeddings.append(nn.Linear(1, embedding_dim))

#         # Define categorical embeddings
#         self.categorical_embeddings = nn.ModuleList()
#         for dim in cat_feature_vocab:
#             self.categorical_embeddings.append(nn.Embedding(dim, embedding_dim))

#         # Define dense part (MLP)
#         self.mlp = nn.Sequential()
#         self.mlp.add_module(f"MLP_layer_{0}", nn.Linear((2 + num_numeric_feats + len(cat_feature_vocab)) * embedding_dim,
#                                                         mlp_layer_sizes[0]))
#         self.mlp.add_module(f"Activation_layer_{0}", nn.ReLU())

#         for i, size in enumerate(mlp_layer_sizes[:-1], 1):
#             self.mlp.add_module(f"MLP_layer_{i}", nn.Linear(size, mlp_layer_sizes[i]))
#             self.mlp.add_module(f"Activation_layer_{i}", nn.ReLU())

#         # FM linear layer
#         self.fm_linear = nn.Linear(num_numeric_feats + sum(cat_feature_vocab), 1)
#         self.final_sigmoid = nn.Sigmoid()

#         self.embedding_dim = embedding_dim
#         self.cat_feature_vocab = cat_feature_vocab

#     def forward(self, user_input, item_input, numeric_feats, categorical_feats):
#         """

#         Args:
#             categorical_feats: ordinal encoded
#         """
#         batch_size = user_input.shape[0]
#         # Encode users and item
#         user_embedded = self.user_embedding(user_input)
#         item_embedded = self.item_embedding(item_input)

#         # Encode numeric
#         num_vec = torch.zeros(0, dtype=torch.float)
#         for i, emb in enumerate(self.numerical_embeddings):
#             num_vec = torch.hstack((num_vec, emb(numeric_feats[:, [i]])))
#         # Encode categorical
#         cat_vec = torch.zeros(0, dtype=torch.float)
#         for i, emb in enumerate(self.categorical_embeddings):
#             cat_vec = torch.hstack((cat_vec, emb(categorical_feats[:, i])))
#         total_embed = torch.hstack((user_embedded, item_embedded, num_vec, cat_vec))

#         # FM component
#         # inner product
#         y_fm_embed = torch.zeros((batch_size, 1), dtype=torch.float)
#         # iterate over all embeddings
#         for i in range(2 + len(self.numerical_embeddings) + len(self.categorical_embeddings) - 1):
#             embed_1 = total_embed[
#                 :, i * self.embedding_dim : (i + 1) * self.embedding_dim
#             ]
#             # iterate over remaining embeddings
#             for j in range(i + 1, 2 + len(self.numerical_embeddings) + len(self.categorical_embeddings)):
#                 embed_2 = total_embed[
#                 :, j * self.embedding_dim : (j + 1) * self.embedding_dim
#                 ]
#                 y_fm_embed += torch.sum(embed_1 * embed_2, dim=1).reshape(batch_size, 1)


#         # FM sparse component
#         # construct one-hot-encoded categorical features
#         one_hot_cat = torch.zeros(0, dtype=torch.float)
#         for i, dim in enumerate(self.cat_feature_vocab):
#             one_hot_i = torch.zeros((batch_size, dim), dtype=torch.float)
#             one_hot_i[torch.arange(batch_size), categorical_feats[:, i]] = 1
#             one_hot_cat = torch.hstack((one_hot_cat, one_hot_i))

#         y_fm_sparse = self.fm_linear(torch.hstack((numeric_feats, one_hot_cat)))
#         y_fm = y_fm_embed + y_fm_sparse

#         # Deep component
#         y_deep = self.mlp(total_embed)

#         # Final summation
#         y = self.final_sigmoid(y_fm + y_deep)
#         return y


import torch
from torch import nn


class DeepFM(nn.Module):
    """
    DeepFM: A Neural Network for Hybrid Recommendations with both Low and High-Order Feature Interactions.

    Parameters
    ----------
    num_numeric_feats : int
        Number of numerical features.
    cat_feature_vocab : list of int
        List of vocabulary sizes for each categorical feature.
    embedding_dim : int, optional
        Dimension of embeddings. Default is 5.
    mlp_layer_sizes : list of int, optional
        List of layer sizes for the MLP component. Default is [16, 32, 64, 1].

    Attributes
    ----------
    numerical_embeddings : nn.ModuleList
        List of linear layers for numerical features.
    categorical_embeddings : nn.ModuleList
        List of embedding layers for categorical features.
    mlp : nn.Sequential
        Sequential module containing the MLP layers.
    fm_linear : nn.Linear
        Linear layer for the FM component.
    final_sigmoid : nn.Sigmoid
        Sigmoid activation function for the final output.
    embedding_dim : int
        Dimension of embeddings.
    cat_feature_vocab : list of int
        List of vocabulary sizes for each categorical feature.

    """

    def __init__(
        self,
        num_users,
        num_items,
        num_numeric_feats,
        cat_feature_vocab,
        embedding_dim=5,
        mlp_layer_sizes=[16, 32, 64, 1],
    ):
        super().__init__()

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Define numeric embeddings
        self.numerical_embeddings = nn.ModuleList()
        for _ in range(num_numeric_feats):
            self.numerical_embeddings.append(nn.Linear(1, embedding_dim))

        # Define categorical embeddings
        self.categorical_embeddings = nn.ModuleList()
        for dim in cat_feature_vocab:
            self.categorical_embeddings.append(nn.Embedding(dim, embedding_dim))

        # Define dense part (MLP)
        self.mlp = nn.Sequential()
        self.mlp.add_module(
            "MLP_layer_0",
            nn.Linear(
                (2 + num_numeric_feats + len(cat_feature_vocab)) * embedding_dim,
                mlp_layer_sizes[0],
            ),
        )
        self.mlp.add_module("Activation_layer_0", nn.ReLU())

        for i, size in enumerate(mlp_layer_sizes[1:], 1):
            self.mlp.add_module(
                f"MLP_layer_{i}", nn.Linear(mlp_layer_sizes[i - 1], size)
            )
            self.mlp.add_module(f"Activation_layer_{i}", nn.ReLU())

        # FM linear layer
        self.fm_linear = nn.Linear(num_numeric_feats + sum(cat_feature_vocab), 1)
        self.final_sigmoid = nn.Sigmoid()

        self.embedding_dim = embedding_dim
        self.cat_feature_vocab = cat_feature_vocab

    def forward(self, user_input, item_input, numeric_feats, categorical_feats):
        """
        Forward pass of the DeepFM model.

        Args:
            user_input (torch.Tensor): Tensor of shape (batch_size,) containing user IDs.
            item_input (torch.Tensor): Tensor of shape (batch_size,) containing item IDs.
            numeric_feats (torch.Tensor): Tensor of shape (batch_size, num_numeric_feats) containing numerical features.
            categorical_feats (torch.Tensor): Tensor of shape (batch_size, num_categorical_feats) containing ordinal encoded categorical features.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing the predicted probabilities.
        """
        batch_size = user_input.shape[0]

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

        # FM component
        y_fm_embed = torch.zeros((batch_size, 1), dtype=torch.float)
        for i in range(
            2 + len(self.numerical_embeddings) + len(self.categorical_embeddings) - 1
        ):
            embed_1 = total_embed[
                :, i * self.embedding_dim : (i + 1) * self.embedding_dim
            ]
            for j in range(
                i + 1,
                2 + len(self.numerical_embeddings) + len(self.categorical_embeddings),
            ):
                embed_2 = total_embed[
                    :, j * self.embedding_dim : (j + 1) * self.embedding_dim
                ]
                y_fm_embed += torch.sum(embed_1 * embed_2, dim=1).reshape(batch_size, 1)

        # FM sparse component
        one_hot_cat = torch.zeros(0, dtype=torch.float)
        for i, dim in enumerate(self.cat_feature_vocab):
            one_hot_i = torch.zeros((batch_size, dim), dtype=torch.float)
            one_hot_i[torch.arange(batch_size), categorical_feats[:, i]] = 1
            one_hot_cat = torch.hstack((one_hot_cat, one_hot_i))

        y_fm_sparse = self.fm_linear(torch.hstack((numeric_feats, one_hot_cat)))
        y_fm = y_fm_embed + y_fm_sparse

        # Deep component
        y_deep = self.mlp(total_embed)

        # Final summation
        y = self.final_sigmoid(y_fm + y_deep)
        return y


class DeepFMImp(nn.Module):
    """
    DeepFMImp: Improved version of DeepFM with weighted summations

    Parameters
    ----------
    num_numeric_feats : int
        Number of numerical features.
    cat_feature_vocab : list of int
        List of vocabulary sizes for each categorical feature.
    embedding_dim : int, optional
        Dimension of embeddings. Default is 5.
    mlp_layer_sizes : list of int, optional
        List of layer sizes for the MLP component. Default is [16, 32, 64, 1].

    Attributes
    ----------
    numerical_embeddings : nn.ModuleList
        List of linear layers for numerical features.
    categorical_embeddings : nn.ModuleList
        List of embedding layers for categorical features.
    mlp : nn.Sequential
        Sequential module containing the MLP layers.
    fm_linear : nn.Linear
        Linear layer for the FM component.
    final_sigmoid : nn.Sigmoid
        Sigmoid activation function for the final output.
    embedding_dim : int
        Dimension of embeddings.
    cat_feature_vocab : list of int
        List of vocabulary sizes for each categorical feature.

    """

    def __init__(
        self,
        num_users,
        num_items,
        num_numeric_feats,
        cat_feature_vocab,
        embedding_dim=5,
        mlp_layer_sizes=[16, 32, 64, 1],
    ):
        super().__init__()

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Define numeric embeddings
        self.numerical_embeddings = nn.ModuleList()
        for _ in range(num_numeric_feats):
            self.numerical_embeddings.append(nn.Linear(1, embedding_dim))

        # Define categorical embeddings
        self.categorical_embeddings = nn.ModuleList()
        for dim in cat_feature_vocab:
            self.categorical_embeddings.append(nn.Embedding(dim, embedding_dim))

        # Define dense part (MLP)
        self.mlp = nn.Sequential()
        self.mlp.add_module(
            "MLP_layer_0",
            nn.Linear(
                (2 + num_numeric_feats + len(cat_feature_vocab)) * embedding_dim,
                mlp_layer_sizes[0],
            ),
        )
        self.mlp.add_module("Activation_layer_0", nn.ReLU())

        for i, size in enumerate(mlp_layer_sizes[1:], 1):
            self.mlp.add_module(
                f"MLP_layer_{i}", nn.Linear(mlp_layer_sizes[i - 1], size)
            )
            self.mlp.add_module(f"Activation_layer_{i}", nn.ReLU())

        # FM linear layer
        self.fm_sparse = nn.Linear(num_numeric_feats + sum(cat_feature_vocab), 1)
        self.fm_linear = nn.Linear(2, 1)

        self.final_linear = nn.Linear(2, 1)
        self.final_sigmoid = nn.Sigmoid()

        self.embedding_dim = embedding_dim
        self.cat_feature_vocab = cat_feature_vocab

    def forward(self, user_input, item_input, numeric_feats, categorical_feats, device):
        """
        Forward pass of the DeepFM model.

        Args:
            user_input (torch.Tensor): Tensor of shape (batch_size,) containing user IDs.
            item_input (torch.Tensor): Tensor of shape (batch_size,) containing item IDs.
            numeric_feats (torch.Tensor): Tensor of shape (batch_size, num_numeric_feats) containing numerical features.
            categorical_feats (torch.Tensor): Tensor of shape (batch_size, num_categorical_feats) containing ordinal encoded categorical features.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing the predicted probabilities.
        """
        batch_size = user_input.shape[0]

        # Embed users and items
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Encode numeric
        num_vec = torch.zeros(0, dtype=torch.float).to(device)
        for i, emb in enumerate(self.numerical_embeddings):
            num_vec = torch.hstack((num_vec, emb(numeric_feats[:, [i]])))

        # Encode categorical
        cat_vec = torch.zeros(0, dtype=torch.float).to(device)
        for i, emb in enumerate(self.categorical_embeddings):
            cat_vec = torch.hstack((cat_vec, emb(categorical_feats[:, i])))

        total_embed = torch.hstack((user_embedded, item_embedded, num_vec, cat_vec))

        # FM component
        y_fm_embed = torch.zeros((batch_size, 1), dtype=torch.float).to(device)
        for i in range(
            2 + len(self.numerical_embeddings) + len(self.categorical_embeddings) - 1
        ):
            embed_1 = total_embed[
                :, i * self.embedding_dim : (i + 1) * self.embedding_dim
            ]
            for j in range(
                i + 1,
                2 + len(self.numerical_embeddings) + len(self.categorical_embeddings),
            ):
                embed_2 = total_embed[
                    :, j * self.embedding_dim : (j + 1) * self.embedding_dim
                ]
                y_fm_embed += torch.sum(embed_1 * embed_2, dim=1).reshape(batch_size, 1)

        # FM sparse component
        one_hot_cat = torch.zeros(0, dtype=torch.float).to(device)
        for i, dim in enumerate(self.cat_feature_vocab):
            one_hot_i = torch.zeros((batch_size, dim), dtype=torch.float).to(device)
            one_hot_i[torch.arange(batch_size), categorical_feats[:, i]] = 1
            one_hot_cat = torch.hstack((one_hot_cat, one_hot_i))

        y_fm_sparse = self.fm_sparse(torch.hstack((numeric_feats, one_hot_cat)))
        y_fm = self.fm_linear(torch.hstack((y_fm_embed, y_fm_sparse)))

        # Deep component
        y_deep = self.mlp(total_embed)

        # Final summation
        y = self.final_sigmoid(self.final_linear(torch.hstack((y_fm, y_deep))))
        return y
