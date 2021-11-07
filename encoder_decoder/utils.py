import numpy as np
import librosa
import os
from data.strechableNumpyArray import StrechableNumpyArray
from data.ourLTFATStft import LTFATStft
from data.modGabPhaseGrad import modgabphasegrad

import ltfatpy
ltfatpy.gabphasegrad = modgabphasegrad # This function is not implemented for one sided stfts with the phase method on ltfatpy

__all__ = [
    'process_waves_folder'
]

def process_waves_folder(folder_name: str) -> np.ndarray:
    """
        Process folder of waves according to the paper processing mechanisem
        returns a Dataset of processed waves
    """
    audios = StrechableNumpyArray()
    i = 0
    total = 0

    for wav_filename in os.listdir(folder_name):
        if wav_filename.endswith(".wav"):
            try:
                audio, sr = librosa.load(os.path.join(folder_name, wav_filename), sr=None, dtype=np.float64)
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
    print(preprocessed_images.shape)
    # exit()
    print(np.max(preprocessed_images[:, :256, :]))
    print(np.min(preprocessed_images[:, :256, :]))
    print(np.mean(preprocessed_images[:, :256, :]))

    return preprocessed_images[:, :256]