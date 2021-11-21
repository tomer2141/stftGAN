from generator import Generator
from discriminator import Discriminator

from PIL import Image
import pickle
import torch
from args import cfg
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.nn.functional as functional
from torchsummary import summary

# if torch.cuda.is_available():
#     cfg.device = 'cuda:0'

import wandb
wandb.init(project="SERVER_TAL", entity="shlomis")

wandb.config = {
    "gf": 64,
    "batch": 64,
    "num_epochs": 10000,
    "alpha": 0.2,
    "gamma_gp": 10,
    "learning_rate": 1e-4,
}

print("device: ", cfg.device)
# Data Preprocess
# real_imgs = data_preprocess_TAL()
# real_imgs = preprocess_like_article()
real_imgs = torch.load('data/logspec_tal_new.pt')
real_imgs = real_imgs/np.abs(cfg.clipBelow/2)+1
real_imgs = real_imgs.type(torch.FloatTensor)
# real_imgs_1 = torch.load('data/logspec_tal_tensor.pt')
print("min: %f, max: %f" % (real_imgs.min(), real_imgs.max()))

# Convert to torch
# real_imgs = torch.from_numpy(real_imgs).type(torch.float32)

dataloader = DataLoader(real_imgs, batch_size=cfg.batch, shuffle=True, drop_last=True)

netG = Generator().to(cfg.device)
netD = Discriminator().to(cfg.device)

# Initialize weights
netG.apply(weights_init)
netD.apply(weights_init)
print('xavier initialization done!')

summary(netG, (64, 100))
summary(netD, (1, 256, 128))


# Track losses
G_losses = []
D_losses = []

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(1, cfg.gen_latent_dim, device=cfg.device)

# Setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

batches_done = 0
summary = {}
cm = plt.get_cmap('viridis')
print("Starting training loop...")
for epoch in range(cfg.num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_data = data.to(cfg.device)
        fake_z = torch.randn(cfg.batch, cfg.gen_latent_dim, device=cfg.device)
        fake_data = netG(fake_z).to(cfg.device)
        D_loss, D_gp = get_lossD(netD, real_data, fake_data)

        D_loss.backward(retain_graph=True)
        optimizerD.step()

        optimizerG.zero_grad()
        if i % cfg.n_critic == 0:
            fake_z = torch.randn(cfg.batch, cfg.gen_latent_dim, device=cfg.device)
            fake_data = netG(fake_z).to(cfg.device)
            G_loss = get_lossG(netD, fake_data)
            G_loss.backward()
            optimizerG.step()

            real_consistency = consistency((real_data-1)*np.abs(cfg.clipBelow/2))
            fake_consistency = consistency((fake_data-1)*np.abs(cfg.clipBelow/2))
            mean_real_consistency = torch.mean(real_consistency)
            mean_fake_consistency = torch.mean(fake_consistency)
            G_reg = torch.abs(mean_real_consistency - mean_fake_consistency)
            G_reg_rel = torch.abs(1 - mean_fake_consistency/mean_real_consistency)
            G_reg_rel1 = torch.mean(torch.abs(mean_real_consistency-mean_fake_consistency)**2)
            G_reg_rel1 = G_reg_rel1 / torch.mean(torch.abs(mean_real_consistency)**2)

            # if batches_done % (10 * cfg.n_critic) == 0:
            #     im = netG(fixed_noise).cpu().data[:]
            #     PIL_image = Image.fromarray(np.uint8(cm(im[0][0]) * 255))
            #     PIL_image.save('images/new_data/%d.png' % batches_done)
            #     # save_img(netG(fixed_noise).cpu().data[:], batches_done)

            if batches_done % (100 * cfg.n_critic) == 0:
                summary['G'] = netG.state_dict()
                summary['D'] = netD.state_dict()
                summary['loos_G'] = G_loss.item()
                summary['loss_D'] = D_loss.item()
                summary['gradient_penalty'] = D_gp.item()
                summary['G_reg'] = G_reg.item()
                with open('checkpoints/with_spec_norm/%d.pkl' % batches_done, 'wb') as handle:
                    pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)
                wandb.log({"loss_G": G_loss.item(), "loss_D": D_loss.item(), "gradient_penalty": D_gp.item(),
                           "G_reg": G_reg.item(), "G_reg_rel": G_reg_rel.item(), "G_reg_rel1": G_reg_rel1.item()})
                wandb.watch(netG)
                # torch.save(netG.state_dict(), "checkpoints/%d_G" % batches_done)
                # torch.save(netD.state_dict(), "checkpoints/%d_D" % batches_done)
                # save_img(netG(fixed_noise).data[:25], batches_done)
                # save_image(netG(fixed_noise).data[0], "images/%d.png" % batches_done, normalize=True)

            batches_done += cfg.n_critic

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, cfg.num_epochs, i, len(dataloader), D_loss.item(), G_loss.item())
        )



print('DONE')

# GAN Running