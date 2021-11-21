from model.generator import Generator, GeneratorMNISTconv
from model.discriminator import DiscriminatorMNISTconv

import torch
from args import cfg
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.nn.functional as functional

# if torch.cuda.is_available():
#     cfg.device = 'cuda:0'

# gen = Generator()
# z = torch.randn((cfg.batch, cfg.gen_latent_dim))
# print(gen)
# out = gen(z)
# print(z.shape)

# Data Preprocess
# real_imgs = data_preprocess_zero2nine()
# real_imgs = data_preprocess_TAL()
mnist_dataset = datasets.MNIST('MNIST', train=True, download=True, transform=None)
real_imgs = (mnist_dataset.train_data).unsqueeze(1).type(torch.float32)
real_imgs = functional.pad(real_imgs, (2, 2, 2, 2), mode='constant', value=0.)
real_imgs = real_imgs/real_imgs.max()

# Convert to torch
# real_imgs = torch.from_numpy(real_imgs).type(torch.float32)

dataloader = DataLoader(real_imgs, batch_size=cfg.batch, shuffle=True, drop_last=True)

netG = GeneratorMNISTconv().to(cfg.device)
netD = DiscriminatorMNISTconv().to(cfg.device)

# Initialize weights
# netG.apply(weights_init)
# netD.apply(weights_init)

# Track losses
G_losses = []
D_losses = []

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(26, cfg.gen_latent_dim, device=cfg.device)

# Setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

batches_done = 0
print("Starting training loop...")
for epoch in range(cfg.num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_data = data.to(cfg.device)
        fake_z = torch.randn(cfg.batch, cfg.gen_latent_dim, device=cfg.device)
        fake_data = netG(fake_z)
        D_loss = get_lossD(netD, real_data, fake_data)

        D_loss.backward(retain_graph=True)
        optimizerD.step()

        optimizerG.zero_grad()
        if i % cfg.n_critic == 0:
            G_loss = get_lossG(netD, fake_data)
            G_loss.backward()
            optimizerG.step()

            if batches_done % (10 * cfg.n_critic) == 0:
                save_image(netG(fixed_noise).data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            if batches_done % (100 * cfg.n_critic) == 0:
                torch.save(netG.state_dict(), "checkpoints/%d_G" % batches_done)
                torch.save(netD.state_dict(), "checkpoints/%d_D" % batches_done)
                # save_img(netG(fixed_noise).data[:25], batches_done)
                # save_image(netG(fixed_noise).data[0], "images/%d.png" % batches_done, normalize=True)

            batches_done += cfg.n_critic

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.num_epochs, i, len(dataloader), D_loss.item(), G_loss.item())
        )



print('DONE')

# GAN Running