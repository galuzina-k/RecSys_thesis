from collections import OrderedDict

from torch import nn


class MLP(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        activation=True,
        dropout=True,
        batchnorm=True,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.block = OrderedDict()
        self.block["Linear"] = nn.Linear(in_features, out_features)

        if activation:
            self.block["Activation"] = nn.ReLU()

        if dropout:
            self.block["Dropout"] = nn.Dropout(p=dropout_rate)

        if batchnorm:
            self.block["BatchNorm"] = nn.BatchNorm1d(num_features=out_features)

        self.block = nn.Sequential(self.block)

    def forward(self, x):
        return self.block(x)
