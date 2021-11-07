import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.models import Sequential

class MapToLatent(tf.keras.Model):
    def __init__(self, num_of_params, out_features):
        super(MapToLatent, self).__init__()
        self.num_of_params = num_of_params

        self.classifier = Sequential()
        self.classifier.add(Dense(256, input_shape=(self.num_of_params,), activation="relu"))
        self.classifier.add(Dense(128, activation="relu"))
        self.classifier.add(Dense(out_features, activation="softmax"))

    def call(self, input_tensor, training=False):
        x = self.classifier(input_tensor)
        return x