__all__ = ['MappersDataset']

from collections import OrderedDict
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input

class MappersDataset():
    def __init__(self, wav_params: OrderedDict, processed_waves: np.ndarray, wav_encodings: dict) -> None:
        self.y = processed_waves
        
        # parse dict
        self.x: list = []

        n_rows = len(wav_encodings[list(wav_params.keys())[0]])
        for row in range(n_rows):
            row_list: list = []
            for col in wav_params.keys():
                row_list.append(wav_encodings[col][row].tolist())
            self.x.append(row_list)
    