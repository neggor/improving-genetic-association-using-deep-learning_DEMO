import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import keras_tuner as kt
import argparse
from datetime import datetime
import random
import tensorflow_addons as tfa
from sklearn.manifold import TSNE

import pandas as pd



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


parser.add_argument('-bs', type = int, required= True,
                    help = 'Batch size')


args = parser.parse_args()

my_args = {k:v for k,v in args._get_kwargs()}

DOWNSAMPLING = my_args["d"]
### CREATE EXPERIMENT NAME

cols_name = '-'.join([str(col) for col in my_args['c']])


EXPERIMENT_NAME =\
f'MINIMAL_Contrastive_C{cols_name}_D{my_args["d"]}_BS{my_args["bs"]}'

print('----------------------------------------------------------------------')
print(EXPERIMENT_NAME)
print('----------------------------------------------------------------------')

## Stratified
print('Loading stratified splits...')
X_train = np.load('data/processed/Genotype_stratified/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
X_val = np.load('data/processed/Genotype_stratified/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
X_test = np.load('data/processed/Genotype_stratified/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
print('X train shape', X_train.shape)
print('X val shape', X_val.shape)
print('X test shape', X_test.shape)
# Now, the labels are the G!
Y_train = np.expand_dims(np.load('data/processed/Genotype_stratified/G_train.npy'), 1)
Y_val = np.expand_dims(np.load('data/processed/Genotype_stratified/G_val.npy'), 1)
Y_test = np.expand_dims(np.load('data/processed/Genotype_stratified/G_test.npy'), 1)
print('G shape', Y_train.shape)

MI_train = pd.read_csv('data/processed/Genotype_stratified/MI_train.csv')
MI_val = pd.read_csv('data/processed/Genotype_stratified/MI_val.csv')
MI_test =pd.read_csv('data/processed/Genotype_stratified/MI_test.csv')
print('MI shape', MI_train.shape)

# From https://keras.io/examples/vision/supervised-contrastive-learning/

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits, basically the similarity between the embeddings
        logits = tf.divide( 
            tf.matmul(
                feature_vectors_normalized[:tf.shape(labels)[0]], tf.transpose(feature_vectors_normalized[tf.shape(labels)[0]:])
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class simple_TCN_contrastive():
    def __init__(   self, 
                    hidden_dim,
                    input_shape,
                    nb_filters,
                    learning_rate,
                    beta_1,
                    beta_2,
                    temperature):
        
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.temperature = temperature
        self.build_model()


    def build_encoder(self):

        input_layer = keras.layers.Input(self.input_shape)
         #batch_size= my_args["bs"])

        augmented = keras.layers.GaussianNoise(
            0.1)(input_layer) 

        concatenation= tf.concat((input_layer, augmented), 0) # Copy with noise

       
       

        # Down-size
        # Kernel size and number of layers fixed for simplicity sake
        TC1 = keras.layers.Conv1D(  filters = self.nb_filters,
                                    kernel_size = 5,
                                    activation = 'gelu',
                                    padding = 'same',
                                    name = 'TC1')(concatenation)

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
        
        embedding = keras.layers.GlobalAveragePooling1D()(TC5)

        self.encoder = keras.models.Model(input_layer, embedding)


    def build_model(self):
        self.build_encoder()

        my_encoder = self.encoder


        optimizer = keras.optimizers.Adam(
                        learning_rate = self.learning_rate,
                        beta_1= self.beta_1,
                        beta_2= self.beta_2,
                        epsilon=1e-07,
                        amsgrad=False,
                        name='Adam')

        my_encoder.compile(loss= SupervisedContrastiveLoss(self.temperature), optimizer= optimizer)
        
        self.encoder = my_encoder


def tune_minimal_contrastive():
    '''
    In: Search options, model.
    Out: Hyperparam.
    '''

    def model_builder(hp):
            


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

            hp_t = hp.Float(   name= 't',
                                min_value = 0.01,
                                max_value = 0.15,
                                sampling = 'log')

            hp_nb_filters = hp.Int(   name= 'nb_filters',
                                        min_value = 10,
                                        max_value = 64,
                                        sampling = 'log')
            

            model_build =  simple_TCN_contrastive(
                                            hidden_dim= 10,
                                            input_shape= (
                                                X_train.shape[1], 
                                                X_train.shape[2]
                                                        ),
                                            learning_rate = hp_lr,
                                            nb_filters= hp_nb_filters,
                                            beta_1= hp_b1,
                                            beta_2= hp_b2,
                                            temperature = hp_t).encoder
            #print(model_build.summary())
            return model_build

    tuner = kt.BayesianOptimization(model_builder,
                objective='val_loss',
                max_trials =  100,
                directory= 'models/hyper_search/Contrastive/',
                project_name= EXPERIMENT_NAME)

    tuner.search(   X_train, 
                    Y_train,
                    batch_size= my_args["bs"],
                    epochs= 80,
                    verbose= True,
                    validation_data=(X_val, Y_val),
                    callbacks = [tf.keras.callbacks.TensorBoard(
                        f"models/hyper_search/tuner_logs/Contrastive/{EXPERIMENT_NAME}"
                                )])


    best_hps_raw = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_hps = {x:best_hps_raw.get(x) for x in [    'lr',
                                                    'nb_filters',
                                                    'beta_1',
                                                    'beta_2',
                                                    't']}
    print(best_hps)
    pd.DataFrame.from_dict(best_hps,
     orient="index").to_csv(
            f"models/hyper_search/best_hyper/Contrastive/{EXPERIMENT_NAME}_hyperparam.csv"
                            )

def train_minimal_contrastive(  n_epochs = 250, 
                        patience = 50): 

   
    my_hp = pd.read_csv(
        'models/hyper_search/best_hyper/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS64_hyperparam.csv',
         index_col = 0)
    
    nb_filters= int(my_hp.loc['nb_filters'][0])
    
    learning_rate= my_hp.loc['lr'][0]
    beta_1 = my_hp.loc['beta_1'][0]
    beta_2 = my_hp.loc['beta_2'][0]
    temperature = my_hp.loc['t'][0]

    # nb_filters = 64
    # learning_rate= 0.0003
    # beta_1 = 0.9
    # beta_2 = 0.99
    # temperature = 0.05
    HD = 10
    my_model_1 = simple_TCN_contrastive(
                            hidden_dim= HD,
                            input_shape= (
                                            X_train.shape[1], 
                                            X_train.shape[2]
                                        ),
                            nb_filters= nb_filters,
                            learning_rate= learning_rate,
                            beta_1 = beta_1,
                            beta_2 = beta_2,
                            temperature= temperature
                            ).encoder

    print(my_model_1.summary())

    TB = tf.keras.callbacks.TensorBoard(
        f"models/training_logs/Contrastive/{EXPERIMENT_NAME}_logs")
    

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
                    batch_size = my_args["bs"],
                    verbose = True,
                    callbacks = [my_early_stopping, TB]
                    ).history
    my_number_of_epochs = my_early_stopping.stopped_epoch - patience
    
    
    my_all_X = np.vstack((X_train, X_val))
    my_all_Y = np.vstack((Y_train, Y_val))

    my_model_2 = simple_TCN_contrastive(
                             hidden_dim= HD,
                            input_shape= (
                                            my_all_X.shape[1], 
                                            my_all_X.shape[2]
                                        ),
                            nb_filters= nb_filters,
                            learning_rate= learning_rate,
                            beta_1 = beta_1,
                            beta_2 = beta_2,
                            temperature= temperature
                            ).encoder

    print('Performance before training:')
    training_loss = my_model_2.evaluate(my_all_X, my_all_Y)
    test_loss = my_model_2.evaluate(X_test, Y_test)
    print(f'BEFORE TRAINING: Test loss: {test_loss}, Training loss: {training_loss}')

    print('\nTRAINING DEFINITIVE MODEL...\n')
    history_all = my_model_2.fit(x = my_all_X,
                    y = my_all_Y,
                    epochs = my_number_of_epochs,
                    batch_size = my_args["bs"],
                    verbose = True,
                    ).history


    print('Assessing loss in test set...')

    training_loss = my_model_2.evaluate(my_all_X, my_all_Y)
    test_loss = my_model_2.evaluate(X_test, Y_test)

    print(f'AFTER TRAINING: Test loss: {test_loss}, Training loss: {training_loss}')

    my_model_2.save(
        f'models/trained_models/Contrastive/{EXPERIMENT_NAME}.h5')
    

   
    figure(figsize=(8, 6), dpi=80)
    plt.plot(history['loss'],
     color = 'blue', alpha = 1, label = 'Loss')
    plt.plot(history['val_loss'],
     color = 'red', alpha = 1, label = 'Val. Loss')
   
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.grid(True)
    plt.legend()
    plt.savefig(f'images/Contrastive/{EXPERIMENT_NAME}.png')
    plt.close()

    # Get traits:
    X_full = np.vstack((X_train, X_val, X_test))
    Y_full = np.vstack((Y_train, Y_val, Y_test))
    
    # Remove the duplicates and reorder:
    embedding, ind = np.unique(my_model_2.predict(X_full),
                                axis = 0,
                                return_index= True)
    embedding = embedding[np.argsort(ind)]


    gen_dim = pd.DataFrame(np.hstack((Y_full, embedding)))
    gen_dim.columns = ['Genotype'] + [f'Dim_{i}' for i in range(embedding.shape[1])]
    gen_dim.to_csv(
    f'data/processed/Hidden_representations/Contrastive/{EXPERIMENT_NAME}.csv',
        index=False)

    #X_embedded = \
    #TSNE(n_components=2, learning_rate='auto', init='pca', perplexity= 6).\
    #fit_transform(embedding)

    #my_df = pd.DataFrame(np.hstack((X_embedded, Y_full)))
    #my_df.columns = ['Dim. 1', 'Dim. 2', 'Color']
    #sns.lmplot(
    #        x='Dim. 1',
    #        y='Dim. 2',
    #        data=my_df, hue='Color', fit_reg=False,
    #        legend = False)

    #plt.savefig(f'images/Contrastive/t_SNE_{EXPERIMENT_NAME}.png')    



    # Metainfo stuff
    MI = np.vstack((MI_train, MI_val, MI_test))
    MI = pd.DataFrame(MI).iloc[:, [2, 3, 4, 5]]
    MI.columns = ['Trial', 'Arena', 'Genotype', 'Plant']
    MI.to_csv(
        f'data/processed/Hidden_representations/Contrastive/{EXPERIMENT_NAME}_metainfo.csv',
         index=False)
    
   
#tune_minimal_contrastive()
train_minimal_contrastive()