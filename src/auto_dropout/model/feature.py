from typing import List

import torch
from torch import nn


class FeatureLayer(nn.Module):
    def __init__(self, feature_processors: List[nn.Module],
                 latent_dim: int = 1024):
        super(FeatureLayer, self).__init__()

        # size of one vector
        self.latent_dim = latent_dim
        self.n_features = len(feature_processors)
        self.feature_processors: List[nn.Module] = feature_processors

    def forward(self, x):
        # feature = (batch_size, latent_dim)
        # features = Tuple[feature]
        # x -> Tuple[feature]

        # process each separate vector
        out = [processor(feature) for processor, feature in zip(self.feature_processors, x)]

        # out -> List of (batch_size, 1)
        return out


class FeatureClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes, dropout_rate=0.5):
        super(FeatureClassifier, self).__init__()
        self.n_classes = n_classes

        self.lin1 = nn.Linear(latent_dim, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout_layer(x)
        x = self.lin1(x)
        x = self.sigmoid(x)
        return x

    def calculate_loss(self, x, target: torch.Tensor):
        criterion = nn.BCELoss()
        target -= 1
        # x.shape: (batch_size, n_classes)
        # target.shape: (batch_size, 1)
        target = torch.nn.functional.one_hot(target.long(), self.n_classes).float()

        loss = criterion(x, target)
        return loss


class FeatureRegressor(nn.Module):
    def __init__(self, latent_dim, dropout_rate=0.5):
        super(FeatureRegressor, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.lin1 = nn.Linear(latent_dim, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.dropout_layer(x)
        x = self.lin1(x)
        return x

    def calculate_loss(self, x, target):
        criterion = nn.MSELoss()
        loss = criterion(x.squeeze(1), target.float())
        return loss
