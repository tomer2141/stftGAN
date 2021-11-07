import sys

sys.path.insert(0, '../../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
from encoder_decoder import MlpMapper, WAV_PARAMS, NUMBER_OF_OUTPUTS_FROM_MAPPER, Paper_GAN, process_waves_folder, MapToLatent
from collections import OrderedDict
from gantools.data.Dataset import Dataset
import tensorflow as tf
from more_itertools import chunked
graph = tf.get_default_graph()
tf.enable_eager_execution()

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

def extract_encodings_from_pd(params: pd.DataFrame) -> dict:
    cols: list = list(params.columns)
    hash_table: dict = {}
    for col in cols:
        hash_table[col] = pd.get_dummies(params[col]).to_numpy().astype('float32')[:30] # TODO: Remove on production!

    return hash_table


def mappers_loss(target_spec: tf.Tensor, gan_spec: tf.Tensor) -> tf.Tensor:
    pass

if __name__ == "__main__":
    print ("Train Mlp Mappers")

    wavs_params: pd.DataFrame = load_excel_of_params("../../data/mlp_mapper/20210727_data_150k_constADSR_CATonly.csv")
    dict_of_wav_encodings: dict = extract_encodings_from_pd(wavs_params)

    # process waves
    processed_waves: np.ndarray = process_waves_folder("../../data/mlp_mapper/wavs")

    # exit(0)
    print ("\nspectograms shape:")
    print (processed_waves.shape)

    #define models
    map_models: OrderedDict = build_mappers()
    paper_GAN = Paper_GAN()
    # map_to_latent = MapToLatent()

    # # train
    epochs = 1
    batches = 15
    for epoch in range(1, epochs+1):
        print (f"\nEpoch: {epoch}")
        batch_count = 0
        for spectogram_batch in chunked(processed_waves, batches):
            # print (wav_encodings_batch)
            mappings = []
            # with tf.GradientTape() as tape:
            for map_model_name in map_models:
                _model = map_models[map_model_name]
                _batch = dict_of_wav_encodings[map_model_name][batch_count:batch_count+batches]
                with graph.as_default():
                    _batch = tf.Variable(_batch)
                    mappings.append(_model(_batch).tolist())

            print (mappings)
            batch_count += batches
            break