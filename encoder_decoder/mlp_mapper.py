import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.models import Sequential
from collections import OrderedDict

NUMBER_OF_OUTPUTS_FROM_MAPPER = 5
WAV_PARAMS = OrderedDict()
WAV_PARAMS['24.osc2waveform'] = 4
WAV_PARAMS['26.lfo1waveform'] = 4
WAV_PARAMS['32.lfo1destination'] = 2
WAV_PARAMS['30.lfo1amount'] = 16
WAV_PARAMS['28.lfo1rate'] = 16
WAV_PARAMS['3.cutoff'] = 16
WAV_PARAMS['4.resonance'] = 16

class MlpMapper(tf.keras.Model):
    def __init__(self, num_of_params, out_features):
        super(MlpMapper, self).__init__()
        self.num_of_params = num_of_params

        self.classifier = Sequential()
        self.classifier.add(Dense(256, input_shape=(self.num_of_params,), activation="relu"))
        self.classifier.add(Dense(128, activation="relu"))
        self.classifier.add(Dense(out_features, activation="softmax"))

    def call(self, input_tensor, training=False):
        x = self.classifier(input_tensor)
        return x