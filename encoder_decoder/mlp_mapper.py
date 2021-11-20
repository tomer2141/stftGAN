from numpy.core.fromnumeric import shape
import tensorflow as tf
from collections import OrderedDict
from gantools import blocks as paperGanBlocks
import re

NUMBER_OF_OUTPUTS_FROM_MAPPER = 5
WAV_PARAMS = OrderedDict()
WAV_PARAMS['24.osc2waveform'] = 4
WAV_PARAMS['26.lfo1waveform'] = 4
WAV_PARAMS['32.lfo1destination'] = 2
WAV_PARAMS['30.lfo1amount'] = 16
WAV_PARAMS['28.lfo1rate'] = 16
WAV_PARAMS['3.cutoff'] = 16
WAV_PARAMS['4.resonance'] = 16

# class MlpMapper():
#     def __init__(self, vector_length, out_features, *args, **kwargs):
#         # super(MlpMapper, self).__init__(*args, **kwargs)

def mlp_mapper_block(x, vector_size, num_of_outputs, name = "map_block") -> tf.Variable:
    # weights_initializer = paperGanBlocks.select_initializer()
    """
    The x(input) here represents the index to fire up e.g x = 2
    and vector_size = 4
    so:
        x_ = [0 0 1 0]
    """
    with tf.variable_scope(name):
        indices = [idx for idx in range(vector_size)]
        depth = vector_size
        one_hot_mapper = tf.one_hot(indices, depth)

        x_ = one_hot_mapper

        # NOTE: We might want to gain back control over the weights_initializer
        layer_1 = paperGanBlocks.linear(x_, 128, scope="dense_1")
        layer_2 = paperGanBlocks.linear(layer_1, 64, scope="dense_2")
        output_layer = paperGanBlocks.linear(layer_2, num_of_outputs, scope="output")

        # normalize
        non_linear_func = net_dict['non_linear_func']
        if non_linear_func is not None:
            output_layer = non_linear_func(output_layer, name="non_linear_func")

        return output_layer