from typing import Tuple, List, Union

import torch
from torch import nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 64, 64), latent_dim: int = 5, n_features=5):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_pixels = self.image_size[1] * self.image_size[2]
        self.encoder = Encoder(image_size=image_size, n_features=n_features)
        self.decoder = Decoder(image_size, latent_dim)

    def get_latent_vector(self, x):
        return self.encoder(x)

    def decode_latent(self, x):
        return self.decoder(x)

    def latent_operations(self, x1, x2, properties: List[bool]):
        sample_x1 = self.get_latent_vector(x1)
        sample_x2 = self.get_latent_vector(x2)

        latent_out = torch.zeros_like(sample_x1)

        for i in range(len(properties)):
            latent_out[0][i] = sample_x1[0][i] if properties[i] else sample_x2[0][i]

        out = self.decode_latent(latent_out)

        return out

    def forward(self, x):
        x = self.encoder(x)

        reconstruction = self.decoder(x)

        return reconstruction
