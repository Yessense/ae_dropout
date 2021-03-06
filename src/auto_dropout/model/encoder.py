from typing import Tuple

import numpy as np

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64), latent_dim: int = 1024, n_features=5):
        super(Encoder, self).__init__()
        """
        Color: white
        Shape: square, ellipse, heart
        Scale: 6 values linearly spaced in [0.5, 1]
        Orientation: 40 values in [0, 2 pi]
        Position X: 32 values in [0, 1]
        Position Y: 32 values in [0, 1]
        """
        # Layer parameters
        hidden_channels = 32
        kernel_size = 4
        hidden_dim = 256

        self.latent_dim = latent_dim
        self.n_features = n_features
        self.image_size = image_size
        self.reshape = (hidden_channels, kernel_size, kernel_size)

        n_channels = self.image_size[0]

        # Convolutional layers
        cnn_kwargs = dict(kernel_size=kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_channels, hidden_channels, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, **cnn_kwargs)
        self.conv_64 = nn.Conv2d(hidden_channels, hidden_channels, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.latent_layer = nn.Linear(hidden_dim, self.n_features * self.latent_dim)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with activation
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))

        # Fully connected layer for log variance and mean
        # x.shape -> (batch_size, latent_dim * n_features)
        # x.shape -> (128, 1024 * 5)
        x = self.latent_layer(x)

        # x.shape -> (batch_size, n_features, latent_dim)
        # x.shape -> (128, 5, 1024)
        x = x.view(-1, self.n_features, self.latent_dim)

        # feature = (batch_size, latent_dim)
        # features = Tuple[feature]
        features = x.unbind(1)
        features = list(features)

        return features
