import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from matplotlib.pyplot import figure
import keras_tuner as kt
import argparse
from datetime import datetime
import random

import pandas as pd
tfpl = tfp.layers
tfd = tfp.distributions



seed_value= 2978040

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#### PARSE COMMAND LINE CONFIG:
parser = argparse.ArgumentParser()

parser.add_argument('-d', type = int, required=True,
                    help = 'Amount of data downsampling')

parser.add_argument('-c', action= 'append', type = int, required=True,
                     help= 'Columns selected econder' )

parser.add_argument('-gs', type = int, required = True,
                    help = 'Stratify by genotype')

parser.add_argument('-bs', type = int, required= True,
                    help = 'Batch size')

parser.add_argument('-beta', type = int, required= False,
                    help = 'Importance of KL divergence')

args = parser.parse_args()

my_args = {k:v for k,v in args._get_kwargs()}

DOWNSAMPLING = my_args["d"]
### CREATE EXPERIMENT NAME

cols_name = '-'.join([str(col) for col in my_args['c']])

if  my_args["beta"] is not None:
    EXPERIMENT_NAME =\
    f'MINIMAL_VAE_C{cols_name}_D{my_args["d"]}_GS{my_args["gs"]}_BS{my_args["bs"]}_Beta{my_args["beta"]}'
else:
    EXPERIMENT_NAME = \
    f'MINIMAL_VAE_C{cols_name}_D{my_args["d"]}_GS{my_args["gs"]}_BS{my_args["bs"]}'

print('----------------------------------------------------------------------')
print(EXPERIMENT_NAME)
print('----------------------------------------------------------------------')
#exit() ##TMP
if not my_args["gs"]:
    print('Loading non-stratified splits...')
    X_train = np.load('data/processed/No_stratification/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_val = np.load('data/processed/No_stratification/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_test = np.load('data/processed/No_stratification/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
    print('X shape', X_train.shape)
    Y_train = np.load('data/processed/No_stratification/X_train.npy')[:, ::DOWNSAMPLING,  9:10]
    Y_val = np.load('data/processed/No_stratification/X_val.npy')[:, ::DOWNSAMPLING,  9:10]
    Y_test = np.load('data/processed/No_stratification/X_test.npy')[:, ::DOWNSAMPLING,  9:10]
    print('Y shape', Y_train.shape)

    G_train = np.expand_dims(np.load('data/processed/No_stratification/G_train.npy'), 1)
    G_val = np.expand_dims(np.load('data/processed/No_stratification/G_val.npy'), 1)
    G_test = np.expand_dims(np.load('data/processed/No_stratification/G_test.npy'), 1)
    print('G shape', G_train.shape)

    MI_train = pd.read_csv('data/processed/No_stratification/MI_train.csv')
    MI_val = pd.read_csv('data/processed/No_stratification/MI_val.csv')
    MI_test =pd.read_csv('data/processed/No_stratification/MI_test.csv')
    print('MI shape', MI_train.shape)

else:
    ## Stratified
    print('Loading stratified splits...')
    X_train = np.load('data/processed/Genotype_stratified/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_val = np.load('data/processed/Genotype_stratified/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_test = np.load('data/processed/Genotype_stratified/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
    print('X shape', X_train.shape)
    Y_train = np.load('data/processed/Genotype_stratified/X_train.npy')[:, ::DOWNSAMPLING,  9:10]
    Y_val = np.load('data/processed/Genotype_stratified/X_val.npy')[:, ::DOWNSAMPLING,  9:10]
    Y_test = np.load('data/processed/Genotype_stratified/X_test.npy')[:, ::DOWNSAMPLING,  9:10]
    print('Y shape', Y_train.shape)

    G_train = np.expand_dims(np.load('data/processed/Genotype_stratified/G_train.npy'), 1)
    G_val = np.expand_dims(np.load('data/processed/Genotype_stratified/G_val.npy'), 1)
    G_test = np.expand_dims(np.load('data/processed/Genotype_stratified/G_test.npy'), 1)
    print('G shape', G_train.shape)

    MI_train = pd.read_csv('data/processed/Genotype_stratified/MI_train.csv')
    MI_val = pd.read_csv('data/processed/Genotype_stratified/MI_val.csv')
    MI_test =pd.read_csv('data/processed/Genotype_stratified/MI_test.csv')
    print('MI shape', MI_train.shape)



X_train = np.vstack((X_train, X_test)) # No need for test set!
Y_train = np.vstack((Y_train, Y_test)) # No need for test set!
G_train = np.vstack((G_train, G_test))
MI_train = np.vstack((MI_train, MI_test))

del X_test, Y_test, G_test

class simple_TCN_VAE():
    def __init__(   self, 
                    hidden_dim,
                    nb_obs,
                    input_shape,
                    nb_filters,
                    kld_weight,
                    learning_rate,
                    beta_1,
                    beta_2):
        
        self.nb_obs = nb_obs
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.build_model()

    def build_encoder(self):
        input_layer = keras.layers.Input(self.input_shape)
         ### Standard normal INDEPENDENT priors ###
        prior = tfp.distributions.MultivariateNormalDiag(
            loc = tf.zeros(self.hidden_dim,),
            scale_diag = tf.ones(self.hidden_dim))
        
        # Down-size
        # Kernel size and number of layers fixed for simplicity sake
        TC1 = keras.layers.Conv1D(  filters = self.nb_filters,
                                    kernel_size = 5,
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TC1')(input_layer)

        TC2 = keras.layers.Conv1D(  filters = self.nb_filters,
                                    kernel_size = 5,
                                    strides = 2, 
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TCN2')(TC1)
        
        TC3 = keras.layers.Conv1D(  filters = self.nb_filters,
                                    kernel_size = 5,
                                    strides = 5, 
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TCN3')(TC2)
        
        TC4 = keras.layers.Conv1D(  filters = self.nb_filters,
                                    kernel_size = 5,
                                    strides = 1, 
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TCN4')(TC3)
        
        TC5 = keras.layers.Conv1D(  filters = self.hidden_dim,
                                    kernel_size = 10,
                                    strides = 10, 
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TCN5')(TC4)
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(TC5)


        parameterization = keras.layers.Dense(
            tfpl.IndependentNormal.params_size(self.hidden_dim), #basically 2* hidden_dim
               activation=None)(gap_layer)

       
        encoder_output = tfpl.IndependentNormal(
            self.hidden_dim, 
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, 
            weight = self.kld_weight / self.nb_obs , #Divide by the size of training data!
            use_exact_kl=False)     
            )(parameterization)
        
        encoder_output = keras.layers.Reshape(
            [self.hidden_dim, 1])(encoder_output)
                                    
        self.encoder = keras.models.Model(input_layer, encoder_output)

        #return(ENCODER)

    def build_decoder(self):

        input_layer = keras.layers.Input((self.hidden_dim, 1))

        dec_upsample = keras.layers.Conv1DTranspose(
                                        filters = 64,
                                        kernel_size = 20,
                                        strides = 20,
                                        padding = 'same',
                                        activation = 'gelu')(input_layer) # 500
        
        dec_upsample = keras.layers.Conv1DTranspose(
                                        filters = 64,
                                        kernel_size = 10,
                                        strides = 10,
                                        activation = 'gelu',
                                        padding = 'same')(dec_upsample) # 5000

        
        dec_upsample = keras.layers.Conv1DTranspose(
                                    kernel_size = 5,
                                    filters = 64,
                                    strides = 5,
                                    activation = 'gelu',
                                    padding = 'same')(dec_upsample) #25000

        dec_upsample = keras.layers.Conv1D(
                                    kernel_size = 5,
                                    filters = 64,
                                    activation = 'gelu',
                                    padding = 'same')(dec_upsample) #25000

        dec_upsample = tf.keras.layers.Cropping1D((0, 2))(dec_upsample) #24998

        final_reconstruction = keras.layers.Conv1D(
                                    kernel_size = 1,
                                    filters = 1,
                                    activation = 'linear',
                                    padding = 'same')(dec_upsample)

        self.decoder = keras.models.Model(inputs=input_layer, outputs=final_reconstruction)

        #return(DECODER)

    def build_model(self):
        self.build_encoder()
        #tf.keras.utils.plot_model(  self.encoder,
        #                            show_shapes = True,
        #                            to_file= 'VAE-TCN_encoder.png')
        self.build_decoder()
        #tf.keras.utils.plot_model(  self.decoder, 
        #                            show_shapes = True, 
        #                            to_file ='VAE-TCN_decoder.png')

        VAE = keras.Sequential([self.encoder, self.decoder])


        optimizer = keras.optimizers.Adam(
                        learning_rate = self.learning_rate,
                        beta_1= self.beta_1,
                        beta_2= self.beta_2,
                        epsilon=1e-07,
                        amsgrad=False,
                        name='Adam')

        VAE.compile(loss= 'mse', metrics = 'mae', optimizer= optimizer)
        
        self.VAE = VAE


def tune_minimal_VAE():
    '''
    In: Search options, model.
    Out: Hyperparam.
    '''

    def model_builder(hp):
            

            hp_w = hp.Float(    name= 'KLD weight',
                                min_value = 0.001,
                                max_value = 1.5,
                                sampling = 'log')

            hp_lr = hp.Float(   name= 'lr',
                                min_value = 1e-5,
                                max_value = 0.001,
                                sampling = 'log')

            hp_b1 = hp.Float(   name= 'beta_1',
                                min_value = 0.2,
                                max_value = 0.999,
                                sampling = 'log')

            hp_b2 = hp.Float(   name= 'beta_2',
                                min_value = 0.2,
                                max_value = 0.999,
                                sampling = 'log')

            hp_nb_filters = hp.Int(   name= 'nb_filters',
                                        min_value = 1,
                                        max_value = 64,
                                        sampling = 'log')

            model_build =  simple_TCN_VAE(   input_shape= (
                                                            X_train.shape[1], 
                                                            X_train.shape[2]
                                                        ),
                                            kld_weight = hp_w,
                                            hidden_dim = 25,
                                            nb_obs= X_train.shape[0], 
                                            learning_rate = hp_lr,
                                            nb_filters= hp_nb_filters,
                                            beta_1= hp_b1,
                                            beta_2= hp_b2).VAE
            #print(model_build.summary())
            return model_build

    tuner = kt.BayesianOptimization(model_builder,
                objective='val_mae',
                max_trials =  100,
                directory= 'models/hyper_search/tuner_logs/self-supervised',
                project_name= EXPERIMENT_NAME)

    tuner.search(   X_train, 
                    Y_train,
                    batch_size= 32,
                    epochs= 250,
                    verbose= True,
                    validation_data=(X_val, Y_val),
                    callbacks = [tf.keras.callbacks.TensorBoard(
                        f"models/hyper_search/tuner_logs/self-supervised/{EXPERIMENT_NAME}"
                                )])


    best_hps_raw = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_hps = {x:best_hps_raw.get(x) for x in [    'KLD weight',
                                                    'lr',
                                                    'nb_filters',
                                                    'beta_1',
                                                    'beta_2']}
    print(best_hps)
    pd.DataFrame.from_dict(best_hps,
     orient="index").to_csv(
            f"models/hyper_search/best_hyper/self-supervised/{EXPERIMENT_NAME}_hyperparam.csv"
                            )

def train_minimal_VAE(hyper_path,
                    n_epochs = 3500, 
                    patience = 100,
                    b =  my_args["beta"]): 

    if b is not None: #Just a patch to try different betas but use same hyperparams
      hyper_path=\
        f'models/hyper_search/best_hyper/self-supervised/MINIMAL_VAE_C{cols_name}_D{my_args["d"]}_GS{my_args["gs"]}_BS{my_args["bs"]}_hyperparam.csv'

    my_hp = pd.read_csv(hyper_path, index_col = 0)
    
    nb_filters= int(my_hp.loc['nb_filters'][0])
    if b == None:
        kld_weight= my_hp.loc['KLD weight'][0]
    else:
        kld_weight = b
    learning_rate= my_hp.loc['lr'][0]
    beta_1 = my_hp.loc['beta_1'][0]
    beta_2 = my_hp.loc['beta_2'][0]

    my_model_1 = simple_TCN_VAE(
                            hidden_dim= 25,
                            input_shape= (
                                            X_train.shape[1], 
                                            X_train.shape[2]
                                        ),
                            nb_obs= X_train.shape[0],
                            nb_filters= nb_filters,
                            kld_weight= kld_weight,
                            learning_rate= learning_rate,
                            beta_1 = beta_1,
                            beta_2 = beta_2
                            ).VAE

    print(my_model_1.summary())
    
    
    TB = tf.keras.callbacks.TensorBoard(
        f"models/training_logs/self-supervised/{EXPERIMENT_NAME}_logs")
    

    my_early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor= 'val_loss',
                                min_delta=0,
                                patience= patience,
                                verbose=0,
                                mode= 'min',
                                baseline=None,
                                restore_best_weights= False)
    
    
    history = my_model_1.fit(x = X_train,
                    y = Y_train,
                    validation_data =  (X_val, Y_val),
                    epochs = n_epochs,
                    batch_size = 32,
                    verbose = True,
                    callbacks = [my_early_stopping, TB]
                    ).history

    my_number_of_epochs = my_early_stopping.stopped_epoch - patience
    
    
    my_all_X = np.vstack((X_train, X_val))
    my_all_Y = np.vstack((Y_train, Y_val))

    my_model_2 = simple_TCN_VAE(
                             hidden_dim= 25,
                            input_shape= (
                                            my_all_X.shape[1], 
                                            my_all_X.shape[2]
                                        ),
                            nb_obs= my_all_X.shape[0],
                            nb_filters= nb_filters,
                            kld_weight= kld_weight,
                            learning_rate= learning_rate,
                            beta_1 = beta_1,
                            beta_2 = beta_2
                            ).VAE


    print('\nTRAINING DEFINITIVE MODEL...\n')
    history_all = my_model_2.fit(x = my_all_X,
                    y = my_all_Y,
                    epochs = my_number_of_epochs,
                    batch_size = 32,
                    verbose = True,
                    ).history


    my_model_2.save(
        f'models/trained_models/Self-Supervised/{EXPERIMENT_NAME}.h5')
    

    #trivial_level = np.mean(
     #   np.absolute((X_val[:, :, 9:10] - np.mean(X_val[:, :, 9:10],
     #    1).reshape(331, 1, 1))))

    figure(figsize=(8, 6), dpi=80)
    plt.plot(history['mae'],
     color = 'blue', alpha = 1, label = 'Loss')
    plt.plot(history['val_mae'],
     color = 'red', alpha = 1, label = 'Val. Loss')
   # plt.hlines(y = trivial_level, xmin= 0, xmax = len(history['loss']),
   #  linestyle = 'dashed',
   #   colors= "black",
    #   label = 'Trivial reconstruction')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

    plt.grid(True)
    plt.legend()
    plt.savefig(f'images/self-supervised/{EXPERIMENT_NAME}.png')

    _plot_examples(my_model_2)
    




    G = np.vstack((G_train, G_val)) # Test set is already incorporated

    return my_model_2, my_all_X, G

def _plot_examples(model):
    plt.figure(figsize=(10, 10), dpi=100)
    for i in range(X_train.shape[0]):
        if i + 1 > 9:
            break
        plt.subplot(3, 3, 1 + i)
        Y_hat = model(X_train[i:i+1]).numpy()
        Y = Y_train[i, :, :]

        plt.plot(Y_hat.flatten(),
                    c = 'green',
                    label = 'pred.',
                    alpha = 1,
                    linestyle='dashed')

        plt.plot(Y.flatten(),
                c = 'black',
                label = 'real',
                alpha = 0.1,
                linestyle='dashed')
        if i in [0, 3, 6]:
            plt.ylabel('Standardized Speed')
        if i in [6, 7, 8]:
            plt.xlabel('Steps')
        #plt.axis('off')
    #plt.title('Reconstructed speed')
    plt.legend()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(f'images/self-supervised/reconstructed_speed_{EXPERIMENT_NAME}')
    plt.show()
    plt.close()


def get_hidden_dim(model, X, G):
    #import pdb
    #print(model.layers)
    #pdb.set_trace()
    extractor = tf.keras.Model(inputs= model.layers[0].inputs,
                        outputs= model.layers[0].layers[-3].output)

    Inner_dim = extractor.predict(X, batch_size = 32)

    Inner_means = Inner_dim[:, 0:25]
    Inner_std = tf.math.softplus(Inner_dim[:, 25:50]).numpy()

    my_inner_dim = np.hstack((Inner_means, Inner_std))
    #print(Inner_means)
    #print(Inner_std)

    gen_dim = pd.DataFrame(np.hstack((G, my_inner_dim)))
    gen_dim.columns = ['Genotype'] + [f'Dim_{i}' for i in range(my_inner_dim.shape[1])]
    gen_dim.to_csv(
        f'data/processed/Hidden_representations/self-supervised/{EXPERIMENT_NAME}.csv',
         index=False)
    ## Retrieving  meta_info
    MI = np.vstack((MI_train, MI_val))
    MI = pd.DataFrame(MI).iloc[:, [2, 3, 4, 5]]
    MI.columns = ['Trial', 'Arena', 'Genotype', 'Plant']
    MI.to_csv(
        f'data/processed/Hidden_representations/self-supervised/{EXPERIMENT_NAME}_metainfo.csv',
         index=False)

    print(gen_dim)



#tune_minimal_VAE()
model, X, G = train_minimal_VAE(f'models/hyper_search/best_hyper/self-supervised/{EXPERIMENT_NAME}_hyperparam.csv')
get_hidden_dim(model, X, G)

## Retrieving  meta_info
#import pdb
#MI = np.vstack((MI_train, MI_val))
##pdb.set_trace()
#MI = pd.DataFrame(MI).iloc[:, [2, 3, 4, 5]]
#MI.columns = ['Trial', 'Arena', 'Genotype', 'Plant']
#MI.to_csv(
#        f'data/processed/Hidden_representations/self-supervised/{EXPERIMENT_NAME}_metainfo.csv',
#         index=False)

#_plot_examples(tf.keras.models.load_model( f'models/trained_models/Self-Supervised/{EXPERIMENT_NAME}.h5'))