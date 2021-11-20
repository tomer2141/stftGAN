import sys

from tensorflow import keras

sys.path.insert(0, '../../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
from encoder_decoder import MapWithGanNN, FitMlpMappers, MappersDataset, WAV_PARAMS, NUMBER_OF_OUTPUTS_FROM_MAPPER, Paper_GAN, process_waves_folder, MapToLatent
from collections import OrderedDict
from gantools.data.Dataset import Dataset
import tensorflow as tf
from more_itertools import chunked

# graph = tf.get_default_graph()
# tf.enable_eager_execution()

def build_mappers() -> OrderedDict:
    print ("Building Mappers...")

    od_bucket_tal = OrderedDict()
    for k, v in WAV_PARAMS.items():
        mlp_mapper = MlpMapper(v, NUMBER_OF_OUTPUTS_FROM_MAPPER)
        od_bucket_tal[k] = mlp_mapper
    return od_bucket_tal

def load_excel_of_params(filename: str) -> pd.DataFrame:
    print (pd.read_csv(filename))
    return pd.read_csv(filename).drop('wav_id', 1)

def extract_encodings_from_pd(params: pd.DataFrame) -> np.ndarray:
    cols: list = list(params.columns)
    hash_table: dict = {}
    list_of_wav_keys: list = list(WAV_PARAMS.keys())
    matrix_of_encoding: np.ndarray = np.zeros((30, len(list_of_wav_keys))) # TODO: Edit on production
    for idx, col in enumerate(list_of_wav_keys):
        dummies = np.argmax(pd.get_dummies(params[col]).to_numpy().astype('float32')[:30], 1) # TODO: Remove on production!
        matrix_of_encoding[:, idx] = dummies
        
    return matrix_of_encoding

@tf.function
def mappers_loss(target_spec: tf.Tensor, gan_spec: tf.Tensor) -> tf.Tensor:
    return 1.

if __name__ == "__main__":
    print ("Train Mlp Mappers")
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../data/mlp_mapper/20210727_data_150k_constADSR_CATonly.csv')

    wavs_params: pd.DataFrame = load_excel_of_params(filename)
    wav_encodings: np.ndarray = extract_encodings_from_pd(wavs_params)
    # process waves
    filename = os.path.join(dirname, '../../data/mlp_mapper/wavs')
    processed_waves: np.ndarray = process_waves_folder(filename)

    # exit(0)
    print ("\nspectograms shape:")
    print (processed_waves.shape)

    #define models 
    model = MapWithGanNN()
    model.train(None, None)
    # map_models: OrderedDict = build_mappers()
    # paper_GAN = Paper_GAN()
    # map_to_latent = MapToLatent(NUMBER_OF_OUTPUTS_FROM_MAPPER * len(list(WAV_PARAMS.items())), 100)
    
    # dataset = MappersDataset(WAV_PARAMS, processed_waves, wav_encodings)
    # print (dict_of_wav_encodings)
    # train
    # training = FitMlpMappers(map_models, map_to_latent, paper_GAN, dynamic=True)
    # training.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=mappers_loss,
    #     run_eagerly=True
    # ) 

    # # with tf.Session() as sess:
    # #     print(self._sess.run([mappers_loss, keras.optimizers.Adam()], feed_dict))
        
    #     # sess.run()
    # training.fit(wav_encodings, processed_waves, batch_size=15, epochs=2)
    # epochs = 1
    # batches = 15
    # for epoch in range(1, epochs+1):
    #     print (f"\nEpoch: {epoch}")
    #     batch_count = 0
    #     for spectogram_batch in chunked(processed_waves, batches):
    #         # print (wav_encodings_batch)
    #         mappings = []
    #         # with tf.GradientTape() as tape:
    #         with graph.as_default():
    #             for map_model_name in map_models:
    #                 _model = map_models[map_model_name]
    #                 _batch = dict_of_wav_encodings[map_model_name][batch_count:batch_count+batches]
    #                 _batch = tf.Variable(_batch)
    #                 _output = _model(_batch)
    #                 mappings.append(_output)

    #             print (len(mappings), mappings)
    #             concatenated_tensors: tf.Tensor = tf.concat(mappings, 1)
    #             print (concatenated_tensors)
    #             latent_space: tf.Tensor = map_to_latent(concatenated_tensors[0])
    #             print (latent_space)
    #             batch_count += batches
    #             break