from collections import OrderedDict
import torch
from torch import nn 

   
class DeepFM(nn.Module):
    """
    DeepFM: A Neural Network for recommendations with both Low and High-Order Feature Interactions.

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
    def __init__(self, num_numeric_feats, cat_feature_vocab,
                 embedding_dim=5, mlp_layer_sizes=[16, 32, 64, 1]):
        super().__init__()

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
        for i, size in enumerate(mlp_layer_sizes[:-1], 1):
            self.mlp.add_module(f"MLP_layer_{i}", nn.Linear(size, mlp_layer_sizes[i]))
            self.mlp.add_module(f"Activation_layer_{i}", nn.ReLU())

        # FM linear layer
        self.fm_linear = nn.Linear(num_numeric_feats + sum(cat_feature_vocab), 1)
        self.final_sigmoid = nn.Sigmoid()
        
        self.embedding_dim = embedding_dim
        self.cat_feature_vocab = cat_feature_vocab

    def forward(self, numeric_feats, categorical_feats):
        """

        Args:
            categorical_feats: one hot encoded
        """
        # Encode numeric
        num_vec = torch.zeros(0, dtype=torch.float32)
        for i, emb in enumerate(self.numerical_embeddings):
            num_vec = torch.hstack((num_vec, emb(numeric_feats[:, i])))

        # Encode categorical
        cat_vec = torch.zeros(0, dtype=torch.float32)
        for i, emb in enumerate(self.categorical_embeddings):
            start_idx = sum(self.cat_feature_vocab[:i])
            end_idx = start_idx + self.cat_feature_vocab[i]
            
            # Extract the one-hot encoded feature and convert it to labels
            feature = categorical_feats[:, start_idx: end_idx]
            labels = torch.where(feature == 1)[1]
            
            cat_vec = torch.hstack((cat_vec, emb(labels)))
        embed = torch.hstack((num_vec, cat_vec))

        # FM component
        # inner product
        y_fm_embed = torch.tensor(0, dtype=torch.float32)
        # iterate over all numerical embeddings
        for i in range(self.numerical_embeddings.shape[0]):
            temp_num_embed = num_vec[:, i * self.embedding_dim: (i + 1) * self.embedding_dim]
            # iterate over all categorical embeddings
            for j in range(self.categorical_embeddings.shape[0]):
                temp_cat_embed = cat_vec[:, j * self.embedding_dim: (j + 1) * self.embedding_dim]
                y_fm_embed += torch.sum(temp_num_embed * temp_cat_embed, dim=1)
        # sparse component
        y_fm_sparse = self.fm_linear(torch.hstack((numeric_feats, categorical_feats)))
        y_fm = y_fm_embed + y_fm_sparse

        # Deep component
        y_deep = self.mlp(embed)

        # Final summation
        y = self.final_sigmoid(y_fm + y_deep)
        return y
        
        
        
        
        