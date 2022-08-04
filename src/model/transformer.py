# Adapted from https://keras.io/examples/timeseries/timeseries_transformer_classification/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import sys
import numpy as np
import pandas as pd
import random
import keras_tuner as kt
sys.path.insert(1, os.path.abspath(''))
from src.model.utils import predict, plot_history
import argparse
from datetime import datetime
import csv


seed_value= 2978040

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()

parser.add_argument('-d', type = int, required=True,
                    help = 'Amount of data downsampling')

parser.add_argument('-c', action= 'append', type = int, required=True,
                     help= 'Columns selected' )

parser.add_argument('-snp', type = int, required = True,
                    help = 'SNP (R position) used as Y')

parser.add_argument('-gs', type = int, required = True,
                    help = 'Stratify by genotype')

parser.add_argument('-bs', type = int, required= True,
                    help = 'Batch size')
                
args = parser.parse_args()

my_args = {k:v for k,v in args._get_kwargs() if v is not None}

DOWNSAMPLING = my_args["d"]

### CREATE EXPERIMENT NAME

cols_name = '-'.join([str(col) for col in my_args['c']])

my_time = datetime.now().strftime("%d%m%Y%H%M")

EXPERIMENT_NAME = f'Transformer_C{cols_name}_D{my_args["d"]}_GS{my_args["gs"]}\
_snp{my_args["snp"]}_BS{my_args["bs"]}'#_{my_time}'
print('----------------------------------------------------------------------')
print(EXPERIMENT_NAME)
print('----------------------------------------------------------------------')

##### LOAD DATA
## No stratified:
if my_args["gs"] == 0:
    print('Loading non-stratified splits...')
    X_train = np.load('data/processed/No_stratification/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_val = np.load('data/processed/No_stratification/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_test = np.load('data/processed/No_stratification/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
    print('X shape', X_train.shape)
    Y_train = np.load('data/processed/No_stratification/y_train.npy')
    Y_val = np.load('data/processed/No_stratification/y_val.npy')
    Y_test = np.load('data/processed/No_stratification/y_test.npy')
    print('Y shape', Y_train.shape)

    G_train = np.expand_dims(np.load('data/processed/No_stratification/G_train.npy'), 1)
    G_val = np.expand_dims(np.load('data/processed/No_stratification/G_val.npy'), 1)
    G_test = np.expand_dims(np.load('data/processed/No_stratification/G_test.npy'), 1)
    print('G shape', G_train.shape)

    print('Proportion train set:', np.mean(Y_train[:, 0]))
    print('Proportion val set:', np.mean(Y_val[:, 0])) 
    print('Proportion test set:', np.mean(Y_test[:, 0]))
elif my_args["gs"] == 1:
    ## Stratified
    print('Loading stratified splits...')
    X_train = np.load('data/processed/Genotype_stratified/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_val = np.load('data/processed/Genotype_stratified/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_test = np.load('data/processed/Genotype_stratified/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
    print('X shape', X_train.shape)
    Y_train = np.load('data/processed/Genotype_stratified/y_train.npy')
    Y_val = np.load('data/processed/Genotype_stratified/y_val.npy')
    Y_test = np.load('data/processed/Genotype_stratified/y_test.npy')
    print('Y shape', Y_train.shape)

    G_train = np.expand_dims(np.load('data/processed/Genotype_stratified/G_train.npy'), 1)
    G_val = np.expand_dims(np.load('data/processed/Genotype_stratified/G_val.npy'), 1)
    G_test = np.expand_dims(np.load('data/processed/Genotype_stratified/G_test.npy'), 1)
    print('G shape', G_train.shape)

    print('Proportion train set:', np.mean(Y_train[:, 0]))
    print('Proportion val set:', np.mean(Y_val[:, 0])) 
    print('Proportion test set:', np.mean(Y_test[:, 0]))
else:
    print('Loading anti-stratified splits...')
    X_train = np.load('data/processed/anti-Stratified/X_train.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_val = np.load('data/processed/anti-Stratified/X_val.npy')[:, ::DOWNSAMPLING, my_args['c']]
    X_test = np.load('data/processed/anti-Stratified/X_test.npy')[:, ::DOWNSAMPLING, my_args['c']]
    print('X shape', X_train.shape)
    
    Y_train = np.load('data/processed/anti-Stratified/y_train.npy')
    Y_val = np.load('data/processed/anti-Stratified/y_val.npy')
    Y_test = np.load('data/processed/anti-Stratified/y_test.npy')
    print('Y shape', Y_train.shape)

    G_train = np.expand_dims(np.load('data/processed/anti-Stratified/G_train.npy'), 1)
    G_val = np.expand_dims(np.load('data/processed/anti-Stratified/G_val.npy'), 1)
    G_test = np.expand_dims(np.load('data/processed/anti-Stratified/G_test.npy'), 1)
    print('G shape', G_train.shape)

    print('Proportion train set:', np.mean(Y_train[:, 0]))
    print('Proportion val set:', np.mean(Y_val[:, 0])) 
    print('Proportion test set:', np.mean(Y_test[:, 0]))

    print('Genotypes in train:\n', np.unique(G_train), '\n')
    print('Genotypes in val:\n', np.unique(G_val), '\n')
    print('Genotypes in test:\n', np.unique(G_test), '\n')

#### MODEL CLASS
class transformer_model:

    def __init__(self, input_shape, head_size, num_heads, dropout,
     ff_dims, mlp_units, batch_size, learning_rate,
     beta_1, beta_2, num_transformer_blocks):

        self.input_shape = input_shape
        self.num_transformer_blocks = num_transformer_blocks
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.mlp_units = mlp_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def transformer_encoder(self, input_tensor):
    # Normalization and Attention
        x = keras.layers.LayerNormalization(epsilon=1e-6)(input_tensor)
        x = keras.layers.MultiHeadAttention(
            key_dim= self.head_size, num_heads= self.num_heads, dropout= self.dropout
        )(x, x)
        x = keras.layers.Dropout(self.dropout)(x)
        res = x + input_tensor

        # Feed Forward Part
        x = keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = keras.layers.Conv1D(filters= self.ff_dims, kernel_size=1, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Conv1D(filters= input_tensor.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self):
        inputs = keras.Input(shape= self.input_shape)
        x = inputs
        x = keras.layers.Conv1D(filters = self.input_shape[1], groups = self.input_shape[1],
        kernel_size = 50, strides = 50, name = 'Event_generator')(x)
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = keras.layers.Dense(dim, activation="relu")(x)
            x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(2, activation="softmax")(x)

   

        model = keras.Model(inputs, outputs)

        model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(
                        learning_rate = self.learning_rate,
                        beta_1= self.beta_1,
                        beta_2= self.beta_2,
                        epsilon=1e-07,
                        amsgrad=False,
                        name='Adam'),
        metrics= ['accuracy', tfa.metrics.MatthewsCorrelationCoefficient(num_classes= 2)])

        return model

#### HYPERPARAMETER TUNER
def tune_transformer_model(num_trials, epochs):
    def model_builder(hp):
            # Hyperparameters to tune:
            hp_head_size = hp.Int(name = 'head_size', min_value = 100, max_value = 256)
            hp_num_heads = hp.Int(name = 'num_heads', min_value = 1, max_value = 5)
            hp_dropout = hp.Choice (name = 'dropout', values = [0., 0.15, 0.5])
            hp_ff_dims = hp.Int(name = 'ff_dims', min_value = 4, max_value = 32, sampling = 'log')
            hp_lr = hp.Float(name= 'learning_rate', min_value = 1e-8, max_value = 0.001, sampling = 'log')
            hp_b1 = hp.Float(name= 'beta_1', min_value = 0.1, max_value = 0.999, sampling = 'log')
            hp_b2 = hp.Float(name= 'beta_2', min_value = 0.1, max_value = 0.999, sampling = 'log')
            hp_num_transformer_blocks = hp.Int(name = 'num_transformer_blocks', min_value = 1, max_value = 4)
            my_model = transformer_model(input_shape = X_train.shape[1:],
                                                head_size = hp_head_size,
                                                num_transformer_blocks = hp_num_transformer_blocks,
                                                num_heads = hp_num_heads,
                                                dropout = hp_dropout,
                                                ff_dims = hp_ff_dims,
                                                mlp_units = [250],
                                                batch_size = 32,
                                                learning_rate = hp_lr ,
                                                beta_1 = hp_b1,
                                                beta_2 = hp_b2).build_model()

            return my_model

    tuner = kt.BayesianOptimization(model_builder,
                    objective='val_loss',
                    max_trials = num_trials,
                    directory= 'models/hyper_search/tuner_logs/supervised',
                    project_name= EXPERIMENT_NAME)

    tuner.search(X_train, Y_train, batch_size= int(my_args["bs"]), epochs= epochs,
                                    verbose= True, validation_data=(X_val, Y_val),
                                     callbacks = [tf.keras.callbacks.TensorBoard(f"models/hyper_search/tuner_logs/supervised/{EXPERIMENT_NAME}_logs")])
        
    best_hps_raw = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    best_hps = {x:best_hps_raw.get(x) for x in ['head_size', 'num_heads', 'dropout', 'ff_dims', 'learning_rate', 'beta_1', 'beta_2', 'num_transformer_blocks']}
    
    print(best_hps)
    pd.DataFrame.from_dict(best_hps, orient="index").to_csv(f"models/hyper_search/best_hyper/supervised/{EXPERIMENT_NAME}_hyperparam.csv")
    return(best_hps)

#### TRAIN
def train_transformer(path_to_hyper):

    for e in range(5):
        print(F' TRAINING MODEL {e}')
        tf.keras.backend.clear_session()
        my_hp = pd.read_csv(path_to_hyper, index_col = 0)
        #print(my_hp)
        input_shape = X_train.shape[1:]

        head_size = int(my_hp.loc['head_size'])
        num_transformer_blocks = int(my_hp.loc['num_transformer_blocks'])
        num_heads = int(my_hp.loc['num_heads'])
        dropout = float(my_hp.loc['dropout'])
        ff_dims = int(my_hp.loc['ff_dims'])
        learning_rate = float(my_hp.loc['learning_rate'])
        beta_1 = float(my_hp.loc['beta_1'])
        beta_2 = float(my_hp.loc['beta_2'])

        n_epochs = 150
        patience = 20

        my_model_1 = transformer_model(input_shape = X_train.shape[1:],
                                                    head_size = head_size,
                                                    num_transformer_blocks = num_transformer_blocks,
                                                    num_heads = num_heads,
                                                    dropout = dropout,
                                                    ff_dims = ff_dims,
                                                    mlp_units = [250],
                                                    batch_size = 32,
                                                    learning_rate = learning_rate,
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2).build_model()

        #tf.keras.utils.plot_model(my_model_1, to_file="Transformer_def.png",
        #show_shapes=True)
        #exit()                                        
        
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
                        verbose = False,
                        callbacks = [my_early_stopping]
                        ).history
        
        my_number_of_epochs = my_early_stopping.stopped_epoch -  (patience - 1)

        tf.keras.backend.clear_session()

        my_model_2 = transformer_model(input_shape = X_train.shape[1:],
                                                    head_size = head_size,
                                                    num_transformer_blocks = num_transformer_blocks,
                                                    num_heads = num_heads,
                                                    dropout = dropout,
                                                    ff_dims = ff_dims,
                                                    mlp_units = [250],
                                                    batch_size = 32,
                                                    learning_rate = learning_rate,
                                                    beta_1 = beta_1,
                                                    beta_2 = beta_2).build_model()
        
        my_all_X = np.vstack((X_train, X_val))
        my_all_Y = np.vstack((Y_train, Y_val))

        history_all = my_model_2.fit(x = my_all_X,
                        y = my_all_Y,
                        epochs = my_number_of_epochs,
                        batch_size = my_args["bs"],
                        verbose = False
                        ).history
        
        mcc, accuracy, loss, y_pred = predict(my_model_2, X_test, Y_test, batch_size = my_args["bs"])
        print('Test performance: ', {'mcc': mcc , 'acc': accuracy, 'loss': loss})
        test_results = {'num' : e, 'mcc':mcc , 'acc':accuracy, 'loss':loss}

        if e == 0:
            my_type = 'w'
        else:
            my_type = 'a'

        with open(f'models/Results/supervised/Predictive_results_{EXPERIMENT_NAME}.csv', my_type) as f:
            w = csv.DictWriter(f, test_results.keys())
            if e == 0:
                w.writeheader()
            w.writerow(test_results)
        
        plot_history(history, np.mean(Y_val[:, 0]), EXPERIMENT_NAME,
        'images/supervised/')
        
        pd.DataFrame.from_dict(history).to_csv(f'models/history_logs/train_{EXPERIMENT_NAME}_{e}.csv')
        pd.DataFrame.from_dict(history_all).to_csv(f'models/history_logs/train+val{EXPERIMENT_NAME}_{e}.csv')
        
        my_model_2.save(f'models/trained_models/Supervised/{EXPERIMENT_NAME}_{e}.h5')
    X = np.vstack((X_train, X_val, X_test))
    G = np.vstack((G_train, G_val, G_test))

    # Select best model base on MCC:
    ensemble_performance = pd.read_csv(f'models/Results/supervised/Predictive_results_{EXPERIMENT_NAME}.csv')
    row_max = np.argmax(ensemble_performance['acc'])

    e = ensemble_performance['num'].iloc[row_max]
    my_model_2 = tf.keras.models.load_model(f'models/trained_models/Supervised/{EXPERIMENT_NAME}_{e}.h5')

    return my_model_2, X, G



#### GET PHENOTYPES
def get_hidden_dim(model, X, G):
    
    extractor = tf.keras.Model(inputs= model.inputs,
                        outputs= model.layers[-2].output)

    Inner_dim = extractor.predict(X, batch_size = my_args["bs"])

    gen_dim = pd.DataFrame(np.hstack((G, Inner_dim)))
    gen_dim.columns = ['Genotype'] + [f'Dim_{i}' for i in range(Inner_dim.shape[1])]
    gen_dim.to_csv(
        f'data/processed/Hidden_representations/supervised/{EXPERIMENT_NAME}.csv',
         index=False)
    print(gen_dim)

#tune_inception_time_model(50, 50)
model, X, G = train_transformer(f'models/hyper_search/best_hyper/supervised/{EXPERIMENT_NAME}_hyperparam.csv')
get_hidden_dim(model, X, G)
