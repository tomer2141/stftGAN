from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras

from .paper_GAN import Paper_GAN
from .map_to_latent import MapToLatent

__all__ = ['FitMlpMappers']

class FitMlpMappers(keras.Model):
    def __init__(self, mappers: OrderedDict, map_to_latent: MapToLatent, gan: Paper_GAN) -> None:
        super(FitMlpMappers, self).__init__()
        self.mappers = mappers
        self.map_to_latent = map_to_latent
        self.paper_Gan = gan

    def train_step(self, data):
        x, y = data
        print (x, y)
        print ("============================")
        # with tf.GradientTape() as tape:
        #      for map_model_name in self.mappers:
        #         _model = self.mappers[map_model_name]
        #         _batch = dict_of_wav_encodings[map_model_name][batch_count:batch_count+batches]
        #         _batch = tf.Variable(_batch)
        #         _output = _model(_batch)
        #         mappings.append(_output)