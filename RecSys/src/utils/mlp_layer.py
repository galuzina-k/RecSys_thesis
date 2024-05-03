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
        self.layer = OrderedDict()
        self.layer["Linear"] = nn.Linear(in_features, out_features)

        if activation:
            self.layer["Activation"] = nn.ReLU()

        if dropout:
            self.layer["Dropout"] = nn.Dropout(p=dropout_rate)

        if batchnorm:
            self.layer["BatchNorm"] = nn.BatchNorm1d(num_features=out_features)

        self.layer = nn.Sequential(self.layer)

    def forward(self, x):
        return self.layer(x)
