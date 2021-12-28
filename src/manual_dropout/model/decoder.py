from typing import Tuple

import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64),
               latent_dim=1024):
        super(Decoder, self).__init__()

        # Layer parameters
        hidden_channels = 32
        kernel_size = 4
        hidden_dim = 256

        self.image_size = image_size
        self.reshape = (hidden_channels, kernel_size, kernel_size)
        n_channels = self.image_size[0]

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)

        self.convT_64 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)
        self.convT1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hidden_channels, n_channels, kernel_size, **cnn_kwargs)

        self.activation = torch.nn.GELU()

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = self.activation(self.lin1(z))
        x = self.activation(self.lin2(x))
        x = self.activation(self.lin3(x))

        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        x = self.activation(self.convT_64(x))
        x = self.activation(self.convT1(x))
        x = self.activation(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x
