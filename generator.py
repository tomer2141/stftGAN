import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import cfg


# Input: (BatchSize,100)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(cfg.gen_latent_dim, 256 * cfg.gf),
            nn.ReLU(inplace=True),
        )
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(8 * cfg.gf, 8 * cfg.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
            # nn.LeakyReLU(cfg.alpha, inplace=True),
            nn.ReLU(inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(8 * cfg.gf, 4 * cfg.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
            # nn.LeakyReLU(cfg.alpha, inplace=True),
            nn.ReLU(inplace=True),
        )
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(4 * cfg.gf, 2 * cfg.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
            # nn.LeakyReLU(cfg.alpha, inplace=True),
            nn.ReLU(inplace=True),
        )
        self.l4 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(2 * cfg.gf, cfg.gf, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
            # nn.LeakyReLU(cfg.alpha, inplace=True),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(cfg.gf, cfg.ch, (12, 3), stride=2, padding=(5, 1), output_padding=(0, 1))),
        )
        
        def pad(x):
            return F.pad(x, (0, 1), "constant", 0.)
        self.pad = pad

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 8 * cfg.gf, 8, 4)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.final(x)
        x = torch.tanh(x)
        return x


class GeneratorMNIST(nn.Module):
    def __init__(self):
        super(GeneratorMNIST, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class GeneratorMNISTconv(nn.Module):
    def __init__(self):
        super(GeneratorMNISTconv, self).__init__()
        dim = 16
        self.latent_to_features = nn.Sequential(
            nn.Linear(cfg.gen_latent_dim, 8 * dim * 2 * 2),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 1, 4, 2, 1),
            nn.Sigmoid()
        )

        self.latent_to_features = nn.Sequential(
            nn.Linear(cfg.gen_latent_dim, 8 * dim * 2 * 2),
            nn.ReLU()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * 16, 2, 2)
        # Return generated image
        return self.features_to_image(x)
