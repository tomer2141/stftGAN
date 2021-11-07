from gantools.model import SpectrogramGAN
from gantools.gansystem import GANsystem
import tensorflow as tf
from gantools import utils
import os

__all__ = ['paper_GAN']

def Paper_GAN() -> GANsystem:
        
    time_str = 'shlomi_data'
    global_path = '../saved_results'

    name = time_str

    from gantools import blocks
    bn = False

    md = 64

    downscale = 1

    params_discriminator = dict()
    params_discriminator['stride'] = [2,2,2,2,2]
    params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
    params_discriminator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
    params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
    params_discriminator['full'] = []
    params_discriminator['minibatch_reg'] = False
    params_discriminator['summary'] = True
    params_discriminator['data_size'] = 2
    params_discriminator['apply_phaseshuffle'] = True
    params_discriminator['spectral_norm'] = True
    params_discriminator['activation'] = blocks.lrelu


    params_generator = dict()
    params_generator['stride'] = [2, 2, 2, 2, 2]
    params_generator['latent_dim'] = 100
    params_generator['consistency_contribution'] = 0
    params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
    params_generator['shape'] = [[12, 3],[12, 3], [12, 3],[12, 3],[12, 3]]
    params_generator['batch_norm'] = [bn, bn, bn, bn]
    params_generator['full'] = [256*md]
    params_generator['summary'] = True
    params_generator['non_lin'] = tf.nn.tanh
    params_generator['activation'] = tf.nn.relu
    params_generator['data_size'] = 2
    params_generator['spectral_norm'] = True 
    params_generator['in_conv_shape'] =[8, 4]

    params_optimization = dict()
    params_optimization['batch_size'] = 64
    params_optimization['epoch'] = 10000
    params_optimization['n_critic'] = 5
    params_optimization['generator'] = dict()
    params_optimization['generator']['optimizer'] = 'adam'
    params_optimization['generator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
    params_optimization['generator']['learning_rate'] = 1e-4
    params_optimization['discriminator'] = dict()
    params_optimization['discriminator']['optimizer'] = 'adam'
    params_optimization['discriminator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
    params_optimization['discriminator']['learning_rate'] = 1e-4

    # all parameters
    params = dict()
    params['net'] = dict() # All the parameters for the model
    params['net']['generator'] = params_generator
    params['net']['discriminator'] = params_discriminator
    params['net']['prior_distribution'] = 'gaussian'
    params['net']['shape'] = [256, 128, 1] # Shape of the image
    params['net']['gamma_gp'] = 10 # Gradient penalty
    params['net']['fs'] = 16000//downscale
    params['net']['loss_type'] ='wasserstein'

    params['optimization'] = params_optimization
    params['summary_every'] = 100 # Tensorboard summaries every ** iterations
    params['print_every'] = 50 # Console summaries every ** iterations
    params['save_every'] = 1000 # Save the model every ** iterations
    params['summary_dir'] = os.path.join(global_path, name +'_summary/')
    params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
    params['Nstats'] = 500

    resume, params = utils.test_resume(True, params)
    params['optimization']['epoch'] = 10000

    return GANsystem(SpectrogramGAN, params)