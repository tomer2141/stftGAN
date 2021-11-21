from argparse import Namespace
import torch
cfg = Namespace(
    device=torch.device("cuda:0"),
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
    mapper_fc_out_dim=5,

    param_buckets_tal={
        '24.osc2waveform': 4, '26.lfo1waveform': 4,
        '32.lfo1destination': 2, '30.lfo1amount': 16, '28.lfo1rate': 16, '3.cutoff': 16, '4.resonance': 16},



)


cfg.encoder_out_dim = sum(cfg.param_buckets_tal.values())
cfg.num_params = len(cfg.param_buckets_tal)
cfg.csv_path = r'/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly.csv'
# gf - generator features
