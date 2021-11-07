import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.models import Sequential
from collections import OrderedDict

from tensorflow.python.pywrap_tensorflow_internal import Flatten
# print (dir(ordereddict.OrderedDict))

class EncoderModel(tf.keras.Model):
    def __init__(self, od_params_dict, sum_of_params):
        super(EncoderModel, self).__init__()
        assert type(od_params_dict) == OrderedDict

        self.od_params_dict: OrderedDict = od_params_dict

        # count the sum of params
        self.sum_of_params: int = sum_of_params

        self.features = Sequential()
        self.features.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding="valid", input_shape=(256, 128, 1)))
        self.features.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding="valid", input_shape=(127, 63, 64)))
        self.features.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 3), activation='relu', padding="valid", input_shape=(63, 31, 128)))
        self.features.add(tf.keras.layers.Conv2D(256, (3, 2), strides=(2, 1), activation='relu', padding="valid", input_shape=(31, 10, 256)))
        self.features.add(tf.keras.layers.Conv2D(256, (3, 2), strides=(1, 2), activation='relu', padding="valid", input_shape=(15, 5, 256)))
        self.features.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding="valid", input_shape=(13, 4, 256)))
        self.features.add(tf.keras.layers.Flatten())

        self.classifier = Sequential()
        self.classifier.add(Dense(self.sum_of_params, input_shape=(3072,), activation="relu"))
        self.classifier.add(Dense(self.sum_of_params, activation="softmax"))

    def call(self, input_tensor, training=False):
        x = self.features(input_tensor)
        x = self.classifier(x)
        
        r_ob = OrderedDict()
        offset = 0
        for k,v in self.od_params_dict.items():
            r_ob[k] = x[0, offset:offset+v.num_of_params]
            offset += v.num_of_params

        return r_ob