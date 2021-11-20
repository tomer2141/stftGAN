from collections import OrderedDict
import tensorflow as tf

from encoder_decoder.mlp_mapper import NUMBER_OF_OUTPUTS_FROM_MAPPER, WAV_PARAMS, mlp_mapper_block

class MapWithGanNN(object):

    def __init__(self) -> None:
        super().__init__()

        tf.reset_default_graph()

        print ("Building MapWithGan Network...")
        self.input_ = tf.placeholder(tf.float32, shape=(1, len(WAV_PARAMS)))
        self.map_to_latent = self.build_mappers_to_latent(self.input_)

    def build_mappers_to_latent(self, x):
        print ("Building Mappers...")

        with tf.variable_scope("mappers"):
            latent_variable = []
            for idx, (k, v) in enumerate(WAV_PARAMS.items()):
                curr_mapper = mlp_mapper_block(x[:, idx], v, NUMBER_OF_OUTPUTS_FROM_MAPPER, name=k)
                latent_variable = tf.concat([latent_variable, curr_mapper], 1)
        
            return latent_variable

    def train(self, dataset, checkpoints_path):
        run_config = tf.ConfigProto()

        with tf.Session(config=run_config) as self._sess:
            self._sess.run(tf.global_variables_initializer())
            pass
    
    def _generate_sample(self):
        res = self._sess.run()
        return res

    def generate(self, sess=None):
        if sess is not None:
            self._sess = sess
        else:
            self._sess = tf.Session()

        samples = self._generate_sample()
        if sess is None:
            self._sess.close()
        return samples