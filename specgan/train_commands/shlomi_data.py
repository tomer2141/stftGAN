import sys
sys.path.insert(0, '../../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import SpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.io
import scipy.signal
import librosa
from data.strechableNumpyArray import StrechableNumpyArray
from data.ourLTFATStft import LTFATStft
from data.modGabPhaseGrad import modgabphasegrad
import ltfatpy
ltfatpy.gabphasegrad = modgabphasegrad # This function is not implemented for one sided stfts with the phase method on ltfatpy


downscale = 1

print('start')

i = 0
total = 0
audios = StrechableNumpyArray()

wav_dir_path = "../../data/test_shlomi_dataset"
preprocessed_images = []
for wav_filename in os.listdir(wav_dir_path):
    if wav_filename.endswith(".wav"):
        print (wav_filename)
        # sr, audio = scipy.io.wavfile.read(os.path.join(wav_dir_path, wav_filename))
        try:
            audio, sr = librosa.load(os.path.join(wav_dir_path, wav_filename), sr=None, dtype=np.float64)
        except:
            continue

        if len(audio) < 16000:
            before = int(np.floor((16000-len(audio))/2))
            after = int(np.ceil((16000-len(audio))/2))
            audio = np.pad(audio, (before, after), 'constant', constant_values=(0, 0))
        if len(audio) > 16000: 
            # print(wav_filename, "is too long: ", len(audio))
            pass
        if np.sum(np.absolute(audio)) < len(audio)*1e-4: 
            print(wav_filename, "doesn't meet the minimum amplitude requirement")
            continue

        audios.append(audio[:16000])
        i+=1

        if i > 1000:
            i -= 1000
            total += 1000
            print("1000 plus!", total)
            break

        # fs, x = scipy.io.wavfile.read(os.path.join(wav_dir_path, wav_filename))
        # x = scipy.signal.stft(x, fs=16384, noverlap=128)[2][0:-1, 0:-1]
        # preprocessed_images.append(x)
        # if (len(preprocessed_images) > 500):
        #     break
    else:
        continue
print("there were:", total+i)

audios = audios.finalize()
print (audios.shape)
audios = np.reshape(audios, (total+i, 16000)).astype(np.float64)
print("audios shape:", audios.shape)

fft_hop_size = 128
fft_window_length = 512
L = 16384
clipBelow = -10
anStftWrapper = LTFATStft()

spectrograms = np.zeros([len(audios), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
tgrads = np.zeros([len(audios), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
fgrads = np.zeros([len(audios), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
print(spectrograms.shape)
gs = {'name': 'gauss', 'M': 512}
    
for index, audio in enumerate(audios):
    realDGT = anStftWrapper.oneSidedStft(signal=audio, windowLength=fft_window_length, hopSize=fft_hop_size)
    spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e**clipBelow, normalize=True)
    spectrograms[index] = spectrogram  
    tgradreal, fgradreal = ltfatpy.gabphasegrad('phase', np.angle(realDGT), fft_hop_size,
                                                fft_window_length)
    tgrads[index] = tgradreal /64
    fgrads[index] = fgradreal /256

shiftedSpectrograms = spectrograms/(-clipBelow/2)+1

preprocessed_images = shiftedSpectrograms
print (preprocessed_images)
print(preprocessed_images.shape)
# exit()
print(np.max(preprocessed_images[:, :256, :]))
print(np.min(preprocessed_images[:, :256, :]))
print(np.mean(preprocessed_images[:, :256, :]))
dataset = Dataset(preprocessed_images[:, :256])


time_str = 'shlomi_data'
global_path = '../../saved_results'

name = time_str

from gantools import blocks
bn = False

md = 64

params_discriminator = dict()
params_discriminator['stride'] = [2,2,2,2,2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['apply_phaseshuffle'] = True
params_discriminator['spectral_norm'] = True
params_discriminator['activation'] = blocks.lrelu


params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['latent_dim'] = 100
params_generator['consistency_contribution'] = 0
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
params_generator['shape'] = [[12, 3],[12, 3], [12, 3],[12, 3],[12, 3]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['activation'] = tf.nn.relu
params_generator['data_size'] = 2
params_generator['spectral_norm'] = True 
params_generator['in_conv_shape'] =[8, 4]

params_optimization = dict()
params_optimization['batch_size'] = 64
params_optimization['epoch'] = 10000
params_optimization['n_critic'] = 5
params_optimization['generator'] = dict()
params_optimization['generator']['optimizer'] = 'adam'
params_optimization['generator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['generator']['learning_rate'] = 1e-4
params_optimization['discriminator'] = dict()
params_optimization['discriminator']['optimizer'] = 'adam'
params_optimization['discriminator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['discriminator']['learning_rate'] = 1e-4



# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [256, 128, 1] # Shape of the image
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['fs'] = 16000//downscale
params['net']['loss_type'] ='wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 500

resume, params = utils.test_resume(True, params)
params['optimization']['epoch'] = 10000

wgan = GANsystem(SpectrogramGAN, params)

wgan.train(dataset, resume=resume)