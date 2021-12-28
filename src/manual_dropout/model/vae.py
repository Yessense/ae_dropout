from typing import Tuple, List

import torch
from torch import nn

from src.manual_dropout.model.decoder import Decoder
from src.manual_dropout.model.encoder import Encoder
from src.manual_dropout.model.feature import FeatureLayer


class VAE(nn.Module):
    def __init__(self, feature_processors: List[nn.Module],
                 image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 5):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.n_features = len(feature_processors)

        self.encoder = Encoder(image_size=self.image_size,
                               latent_dim=self.latent_dim,
                               n_features=self.n_features)
        self.feature_layer = FeatureLayer(feature_processors=feature_processors,
                                          latent_dim=latent_dim)
        self.decoder = Decoder(image_size=image_size,
                               latent_dim=latent_dim)

        # generate 1d vectors for future torch.repeat command
        self.manual_drop = [torch.randint(2, (latent_dim,)).float()
                            for _ in range(self.n_features)]

    def get_latent_vector(self, x):
        # x -> (1, 5, 1024)
        return self.encoder(x)


    def decode_latent(self, x):
        batch_size = x[0].shape[0]

        # prepare for decoder
        for i in range(self.n_features):
            drop = self.manual_drop[i].repeat(batch_size, 1).to('cuda:0')
            x[i] = x[i] * drop

        scene = sum(x)

        return self.decoder(scene)

    def latent_operations(self, x1, x2, properties: List[bool]):
        n_features = len(properties)
        sample_x1 = self.get_latent_vector(x1)
        sample_x2 = self.get_latent_vector(x2)

        latent_out = [torch.zeros_like(sample_x1[i]) for i in range(n_features)]

        for i in range(n_features):
            latent_out[i][0] = sample_x1[i][0] if properties[i] else sample_x2[i][0]

        out = self.decode_latent(latent_out)

        return out

    def forward(self, x):
        batch_size = x.shape[0]

        # x -> [(-1, 1024), ...]
        x = list(self.encoder(x))

        # feature_out -> [class, value, value, value, value]
        feature_out = self.feature_layer(x)

        # prepare for decoder
        for i in range(self.n_features):
            drop = self.manual_drop[i].repeat(batch_size, 1).to('cuda:0')
            x[i] = x[i] * drop

        scene = sum(x)

        reconstruction = self.decoder(scene)

        return reconstruction, feature_out
