from generator import Generator, Generator
from discriminator import Discriminator
from args import cfg
from torchvision.utils import save_image

import torch
# netG = Generator()
# netG.load_state_dict(torch.load('checkpoints/273000_G'))
# netG.eval()
# # netD = DiscriminatorMNISTconv().load_state_dict(torch.load('checkpoints/273000_D'))
#
# fixed_noise = torch.randn(400, cfg.gen_latent_dim, device=cfg.device)
# save_image(netG(fixed_noise).data[:], "evaluation/try2.png", nrow=20, normalize=True)

netG = Generator()
netG.load_state_dict(torch.load('checkpoints/92000_G'))
netG.eval()
# netD = DiscriminatorMNISTconv().load_state_dict(torch.load('checkpoints/273000_D'))

fixed_noise = torch.randn(400, cfg.gen_latent_dim, device=cfg.device)
save_image(netG(fixed_noise).data[:], "evaluation/try2.png", nrow=20, normalize=True)
