from argparse import Namespace
import torch
cfg = Namespace(
    device=torch.device("cuda:2"),
    fft_length=512,
    fft_hop_size=128,
    L=16384,
    generator_input_dim=[1, 256, 128],
    gen_latent_dim=100,
    gf=64,
    batch=64,
    num_epochs=10000,
    alpha=0.2,
    ch=1,
    loss='wasserstein_gp',
    gamma_gp=10,
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9,
    n_critic=5,
    clipBelow=-10,



)

# gf - generator features
