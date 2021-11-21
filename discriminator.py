import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import cfg


# Input: (BatchSize,256,128,1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(512 * cfg.gf, 1)
        )
        self.l1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(cfg.ch, cfg.gf, (12, 3), stride=2, padding=(5, 1))),
            nn.LeakyReLU(cfg.alpha, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(cfg.gf, 2 * cfg.gf, (12, 3), stride=2, padding=(5, 1))),
            nn.LeakyReLU(cfg.alpha, inplace=True),
        )
        self.l3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2 * cfg.gf, 4 * cfg.gf, (12, 3), stride=2, padding=(5, 1))),
            nn.LeakyReLU(cfg.alpha, inplace=True),
        )
        self.l4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4 * cfg.gf, 8 * cfg.gf, (12, 3), stride=2, padding=(5, 1))),
            nn.LeakyReLU(cfg.alpha, inplace=True),
        )
        self.final = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(8 * cfg.gf, 16 * cfg.gf, (12, 3), stride=2, padding=(5, 1))),
            nn.LeakyReLU(cfg.alpha, inplace=True),
        )

        def apply_phaseshuffle(x, rad=2, pad_type='reflect'):
            W = x.shape[-1]
            phase = torch.randint(-rad, rad + 1, (1,))
            pad_l = torch.maximum(phase, torch.tensor(0))
            pad_r = torch.maximum(-phase, torch.tensor(0))
            phase_start = pad_r
            x = F.pad(x, (pad_l, pad_r, 0, 0), mode='reflect')
            x = x[:, :, :, phase_start:phase_start + W]
            return x

        self.apply_phaseshuffle = apply_phaseshuffle

    def forward(self, x):
        # x = self.apply_phaseshuffle(self.l1(x))
        # x = self.apply_phaseshuffle(self.l2(x))
        # x = self.apply_phaseshuffle(self.l3(x))
        # x = self.apply_phaseshuffle(self.l4(x))
        # x = self.apply_phaseshuffle(self.final(x))
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class DiscriminatorMNIST(nn.Module):
    def __init__(self):
        super(DiscriminatorMNIST, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)


class DiscriminatorMNISTconv(nn.Module):
    def __init__(self):
        super(DiscriminatorMNISTconv, self).__init__()
        dim = 16
        self.image_to_features = nn.Sequential(
            nn.Conv2d(1, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        output_size = 8 * dim * 2 * 2
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
