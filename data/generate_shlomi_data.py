import librosa
from strechableNumpyArray import StrechableNumpyArray
import numpy as np
import os

pathToBaseDatasetFolder = '/home/robot/dev/stftGAN_forked/data/test_shlomi_dataset/'
folderNames = ['train']#, 'test', 'valid']
dirs = [pathToBaseDatasetFolder]
audios = StrechableNumpyArray()
i = 0
total = 0
print('start')
for directory in dirs:
    print(directory)
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):      
            audio, sr = librosa.load(directory + '/' + file_name, sr=None, dtype=np.float64)

            if len(audio) < 16000:
                before = int(np.floor((16000-len(audio))/2))
                after = int(np.ceil((16000-len(audio))/2))
                audio = np.pad(audio, (before, after), 'constant', constant_values=(0, 0))
            if len(audio) > 16000: 
                # print(file_name, "is too long: ", len(audio))
                pass
            if np.sum(np.absolute(audio)) < len(audio)*1e-4: 
                print(file_name, "doesn't meet the minimum amplitude requirement")
                continue

            audios.append(audio[:16000])
            i+=1

            if i > 1000:
                i -= 1000
                total += 1000
                print("1000 plus!", total)
                break
print("there were:", total+i)

audios = audios.finalize()
print (audios.shape)
audios = np.reshape(audios, (total+i, 16000)).astype(np.float64)
print("audios shape:", audios.shape)

from ourLTFATStft import LTFATStft
import ltfatpy
from modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad # This function is not implemented for one sided stfts with the phase method on ltfatpy

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

import scipy.io

nameForFile = 'shlomi_test_spectrograms_and_derivs'

print(spectrogram.max())
print(spectrogram.min())
print(spectrogram.mean())

shiftedSpectrograms = spectrograms/(-clipBelow/2)+1
print(shiftedSpectrograms.max())
print(shiftedSpectrograms.min())
print(shiftedSpectrograms.mean())

countPerFile = 4000 # mat files sadly cannot be arbitrarily large. 4000 works for 3 matrices (mag+tderiv+fderiv).

for index in range(1 + len(spectrograms)//countPerFile):
    scipy.io.savemat(nameForFile + '_' + str(index+1) + '.mat', dict(logspecs=shiftedSpectrograms[index*countPerFile:(index+1)*countPerFile], 
                                                               tgrad=tgrads[index*countPerFile:(index+1)*countPerFile], 
                                                               fgrad=fgrads[index*countPerFile:(index+1)*countPerFile]))
