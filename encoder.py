import numpy as np
import torch
import torch.nn as nn
from args import cfg


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(512 * cfg.gf, cfg.encoder_out_dim)
        )
        self.l1 = nn.Sequential(
            nn.Conv2d(cfg.ch, cfg.gf, (12, 3), stride=2, padding=(5, 1)),
            nn.LeakyReLU(cfg.alpha),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(cfg.gf, 2 * cfg.gf, (12, 3), stride=2, padding=(5, 1)),
            nn.LeakyReLU(cfg.alpha),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(2 * cfg.gf, 4 * cfg.gf, (12, 3), stride=2, padding=(5, 1)),
            nn.LeakyReLU(cfg.alpha),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(4 * cfg.gf, 8 * cfg.gf, (12, 3), stride=2, padding=(5, 1)),
            nn.LeakyReLU(cfg.alpha),
        )
        self.final = nn.Sequential(
            nn.Conv2d(8 * cfg.gf, 16 * cfg.gf, (12, 3), stride=2, padding=(5, 1)),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.final(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        dict = {}
        from_idx = 0
        for i, f in cfg.param_buckets_tal:
            to_index = from_idx + cfg.param_buckets_tal[f]
            dict[f] = nn.Softmax(x[:, from_idx:to_index - 1])
            from_idx = to_index + 1

        return dict
