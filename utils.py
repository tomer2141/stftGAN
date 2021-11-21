import scipy
import torch
from torch import autograd
from args import cfg
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os
import scipy.signal as signal
from PIL.Image import Image
import pandas as pd
import torch.nn as nn


def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
    eps = eps.expand_as(real_data)

    # Interpolation between real data and fake data.
    interpolation = (eps * real_data + (1 - eps) * fake_data).to(cfg.device)

    # get logits for interpolated images
    interp_logits = netD(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # Compute and return Gradient Norm
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


def compute_gp_mapper(real_data, fake_data):
    # get logits for interpolated images
    interp_logits = (torch.sum((real_data - fake_data) ** 2, (1, 2, 3))).requires_grad(True)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=fake_data,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # Compute and return Gradient Norm
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


def data_preprocess():
    path = 'C:\\Users\\sshlo\\PycharmProjects\\RadarChallenge\\raw_data\\MAFAT RADAR Challenge - Training Set V1.pkl'
    with open(path, 'rb') as f:
        out = pickle.load(f)
    x = np.abs(out['iq_sweep_burst'])
    x = np.expand_dims(x, axis=1)
    return x


def data_preprocess_TAL():
    path = r'/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly/'
    specs = []
    for i in range(150000):
        fs, data = wavfile.read(os.path.join(path, str(i) + '.wav'))
        spec_i = signal.stft(data, fs, window='hann', nperseg=512, noverlap=512 - 128)
        spec_i = np.abs(spec_i[2][:-1, :-1]) ** 2
        spec_i = spec_i / np.max(spec_i) + np.finfo(float).eps
        spec_i = np.log10(spec_i)
        spec_i = np.maximum(spec_i, cfg.clipBelow)
        spec_i = spec_i / (np.abs(cfg.clipBelow) / 2) + 1
        specs.append(spec_i)
    out = np.expand_dims(np.array(specs), 1)
    out1 = torch.from_numpy(out).type(torch.float32)
    torch.save(out1, 'data/logspec_tal_tensor.pt')
    # with open('data/logspec_tal.pkl', 'wb') as handle:
    #     pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def data_preprocess_zero2nine():
    path0 = r'C:\Users\sshlo\PycharmProjects\stftGAN\stftGAN\data\sc09\train'
    specs = []
    nb_bits = 16
    max_nb_bit = float(2 ** (nb_bits - 1))
    for path, currentDirectory, files in os.walk(path0):
        for file in files:
            if file.startswith("Zero"):
                fs, data = wavfile.read(os.path.join(path, file))
                data = data.astype(float) / max_nb_bit
                if data.size > 16384:
                    data = data[:16384]
                elif data.size < 16384:
                    data = np.append(data, np.zeros((16384 - data.size,)))
                spec_i = signal.stft(data, fs, window='hann', nperseg=512, noverlap=512 - 128)
                spec_i = np.abs(spec_i[2][:-1, :-1]) ** 2
                spec_i = spec_i / np.max(spec_i) + np.finfo(float).eps
                spec_i = np.log10(spec_i)
                spec_i = np.maximum(spec_i, cfg.clipBelow)
                spec_i = spec_i / (cfg.clipBelow / 2) + 1
                specs.append(spec_i)
    return np.expand_dims(np.array(specs), 1)


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


def get_lossD(netD, real_img, fake_img, loss=cfg.loss):
    if loss == 'wasserstein_gp':
        D_fake = netD(fake_img)
        D_real = netD(real_img)

        gamma_gp = cfg.gamma_gp
        D_gp = compute_gp(netD, real_img, fake_img)

        D_loss_fake = torch.mean(D_fake)
        D_loss_real = torch.mean(D_real)

        D_loss = -(D_loss_real - D_loss_fake) + gamma_gp * D_gp
        # G_loss = -D_loss_fake

        return D_loss, D_gp


def get_lossG(netD, fake_img, loss=cfg.loss):
    if loss == 'wasserstein_gp':
        D_fake = netD(fake_img)
        D_loss_fake = torch.mean(D_fake)
        G_loss = -D_loss_fake

        return G_loss


def save_img(x, b):
    # fig = plt.figure(figsize=(8, 8))
    # columns = 5
    # rows = 5
    # for i in range(1, columns * rows + 1):
    #     img = x[i-1][0].detach().numpy()
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.savefig("images/%d.png" % b)

    # fig = plt.figure(figsize=(8, 8))
    # columns = 1
    # rows = 1
    # for i in range(1, 2):
    #     img = x[i - 1][0].detach().numpy()
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.savefig("images/%d.png" % b)
    # del fig

    img = torch.squeeze(x)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("images/%d.png" % b, transparent=True)
    plt.clf()


import ltfatpy
from funcs.ourLTFATStft import LTFATStft
from funcs.modGabPhaseGrad import modgabphasegrad
from funcs.numba_pghi import pghi

ltfatpy.gabphasegrad = modgabphasegrad


def generate_audio_with_phase(generated_signals, fft_hop_size=cfg.fft_hop_size, fft_window_length=cfg.fft_length,
                              L=cfg.L):
    anStftWrapper = LTFATStft()

    # Compute Tgrad and Fgrad from the generated spectrograms
    tgrads = np.zeros_like(generated_signals)
    fgrads = np.zeros_like(generated_signals)
    gs = {'name': 'gauss', 'M': fft_window_length}

    for index, magSpectrogram in enumerate(generated_signals):
        tgrads[index], fgrads[index] = ltfatpy.gabphasegrad('abs', magSpectrogram, gs, fft_hop_size)

    reconstructed_audios = np.zeros([len(generated_signals), L])
    for index, magSpectrogram in enumerate(generated_signals):
        logMagSpectrogram = np.log(magSpectrogram.astype(np.float64))
        phase = pghi(logMagSpectrogram, tgrads[index], fgrads[index], fft_hop_size, fft_window_length, L, tol=10)
        reconstructed_audios[index] = anStftWrapper.reconstructSignalFromLoggedSpectogram(logMagSpectrogram, phase,
                                                                                          windowLength=fft_window_length,
                                                                                          hopSize=fft_hop_size)
        print(reconstructed_audios[index].max())

    print("reconstructed audios!")

    scipy.io.savemat('generated_wavs/commands_listen.mat',
                     {"reconstructed": reconstructed_audios, "generated_spectrograms": generated_signals})


def consistency(spectrogram):
    # Input: logspec tensor (B,H,W), Channels=1, range=(-Inf,0] i.e. [-10,0]
    spectrogram = torch.squeeze(spectrogram)
    ttderiv = spectrogram[:, 1:-1, :-2] - 2 * spectrogram[:, 1:-1, :-2] + spectrogram[:, 1:-1, 2:] + torch.pi / 4
    ffderiv = spectrogram[:, :-2, 1:-1] - 2 * spectrogram[:, 1:-1, 1:-1] + spectrogram[:, 2:, 1:-1] + torch.pi / 4

    ttderiv = torch.abs(ttderiv)
    ffderiv = torch.abs(ffderiv)

    absttderiv = (ttderiv - torch.mean(ttderiv, (1, 2), True)) / torch.std(ttderiv, (1, 2), keepdim=True)
    absffderiv = (ffderiv - torch.mean(ffderiv, (1, 2), True)) / torch.std(ffderiv, (1, 2), keepdim=True)

    consistencies = torch.sum(absttderiv * absffderiv, (1, 2))
    return consistencies


def save_img_with_PIL(x, file_name='new_image.png'):
    # Input x: tensor
    cm = plt.get_cmap('viridis')
    x_cm = cm(np.squeeze(x.detach().numpy()))
    PIL_img = Image.fromarray(np.uint8(x_cm * 255))
    PIL_img.save(file_name)


def csv2dic(path=cfg.csv_path):
    df = pd.read_csv(path)
    dic = {}
    for field, col in df.iteritems():
        if not (field == 'wav_id'):
            dic[field] = torch.from_numpy(pd.get_dummies(col).to_numpy()).to(cfg.device)
    return dic


import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def preprocess_like_article():
    path = r'/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly/'
    len_audios = 150000
    fft_hop_size = 128
    fft_window_length = 512
    L = 16384
    clipBelow = -10

    anStftWrapper = LTFATStft()
    spectrograms = np.zeros([len_audios, int(fft_window_length // 2 + 1), int(L / fft_hop_size)], dtype=np.float64)
    gs = {'name': 'gauss', 'M': 512}

    for index in range(len_audios):
        fs, audio = wavfile.read(os.path.join(path, str(index) + '.wav'))
        audio = audio.astype(np.float64)
        realDGT = anStftWrapper.oneSidedStft(signal=audio, windowLength=fft_window_length, hopSize=fft_hop_size)
        spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e ** clipBelow, normalize=True)
        spectrograms[index] = spectrogram
    spectrograms = np.expand_dims(spectrograms, axis=1)
    spectrograms = torch.from_numpy(spectrograms[:, :, :-1, :])
    torch.save(spectrograms, 'data/logspec_tal_new.pt')
    return spectrograms


# def get_initializer(type='xavier'):
#     if type == 'xavier':
#         return nn.init.xavier_uniform

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
