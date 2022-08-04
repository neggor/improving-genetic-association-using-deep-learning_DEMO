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
from src.features.process_data import convert_to_image
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


parser.add_argument('-snp', type = int, required = True,
                    help = 'SNP (R position) used as Y')

parser.add_argument('-gs', type = int, required = True,
                    help = 'Stratify by genotype')

parser.add_argument('-bs', type = int, required= True,
                    help = 'Batch size')

parser.add_argument('-res', type = int, required= True,
                    help = 'Image resolution')
                
args = parser.parse_args()

my_args = {k:v for k,v in args._get_kwargs() if v is not None}

### CREATE EXPERIMENT NAME


my_time = datetime.now().strftime("%d%m%Y%H%M")

EXPERIMENT_NAME = f'Xception_GS{my_args["gs"]}\
_snp{my_args["snp"]}_BS{my_args["bs"]}_RES{my_args["res"]}'#_{my_time}'
print('----------------------------------------------------------------------')
print(EXPERIMENT_NAME)
print('----------------------------------------------------------------------')

##### LOAD DATA
## No stratified:
if my_args["gs"] == 0:
    print('Loading non-stratified splits...')
    X_train = convert_to_image(np.load('data/processed/No_stratification/X_train.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_val = convert_to_image(np.load('data/processed/No_stratification/X_val.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_test = convert_to_image(np.load('data/processed/No_stratification/X_test.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))

    ## Convert to RGB to feed Xception:
    X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train))
    X_val = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_val))
    X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test))

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
    print('Loading stratified splits...')
    X_train = convert_to_image(np.load('data/processed/Genotype_stratified/X_train.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_val = convert_to_image(np.load('data/processed/Genotype_stratified/X_val.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_test = convert_to_image(np.load('data/processed/Genotype_stratified/X_test.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))

    ## Convert to RGB to feed Xception:
    X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train))
    X_val = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_val))
    X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test))

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
    print('Loading anti-Stratified splits...')
    X_train = convert_to_image(np.load('data/processed/anti-Stratified/X_train.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_val = convert_to_image(np.load('data/processed/anti-Stratified/X_val.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))
    X_test = convert_to_image(np.load('data/processed/anti-Stratified/X_test.npy')[:, :, [0, 1, 2, 9]],
        resolution = (my_args["res"], my_args["res"]))

    ## Convert to RGB to feed Xception:
    X_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_train))
    X_val = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_val))
    X_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_test))

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
    

class Xception_model:
    def __init__(self, input_shape, learning_rate, beta_1, beta_2):
        self.base_model = keras.applications.Xception(
            include_top = False,
            weights = "imagenet",
            input_shape = input_shape)

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.model = self.build(input_shape)

    def build(self, input_shape):
        inputs = keras.Input(shape = input_shape)
        x = self.base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
    
        outputs = keras.layers.Dense(2, activation = 'softmax')(x)
        
        model = keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(
                        learning_rate= self.learning_rate,
                        beta_1= self.beta_1,
                        beta_2= self.beta_2)

        model.compile(optimizer= optimizer,
        loss= 'categorical_crossentropy',
        metrics = ['accuracy', tfa.metrics.MatthewsCorrelationCoefficient(num_classes= 2)])

        # Freeze Xception weights:
        # Train top:
        print('Training top...')
        model.fit(
            X_train, Y_train, 
            validation_data = (X_val, Y_val),
            callbacks = [tf.keras.callbacks.EarlyStopping(patience= 2, restore_best_weights=True)],
            epochs = 5, verbose = False, batch_size= int(my_args["bs"]))
        
        # Un- freeze Xception weights:
        self.base_model.trainable = True


        return model

'''
def tune_Xception(num_trials, epochs):

    ## Override HyperModel class: https://github.com/keras-team/keras-tuner/blob/master/keras_tuner/engine/hypermodel.py

    def model_builder(hp):
        hp_lr = hp.Float(name= 'learning_rate', min_value = 1e-8, max_value = 0.001, sampling = 'log')
        hp_b1 = hp.Float(name= 'beta_1', min_value = 0.1, max_value = 0.999, sampling = 'log')
        hp_b2 = hp.Float(name= 'beta_2', min_value = 0.1, max_value = 0.999, sampling = 'log')
        
        my_model = Xception_model(input_shape= (int(my_args["res"]),  int(my_args["res"]), 3),
                                    learning_rate = hp_lr,
                                    beta_1 = hp_b1,
                                    beta_2 = hp_b2).model
        
        return my_model



    tuner = kt.BayesianOptimization(hypermodel = model_builder,
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
'''
#### TRAIN

def train_Xception():
    n_epochs = 150
    patience = 25

    my_model_1 = Xception_model(input_shape= (int(my_args["res"]),  int(my_args["res"]), 3),
                                    learning_rate = 0.000003,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999).model
        
    
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
    
    my_number_of_epochs = my_early_stopping.stopped_epoch - (patience - 1)
    print(my_number_of_epochs)
    tf.keras.backend.clear_session()

    my_model_2 = Xception_model(input_shape= (int(my_args["res"]),  int(my_args["res"]), 3),
                                    learning_rate = 0.000003,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999).model
    
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
    test_results = {'mcc':mcc , 'acc':accuracy, 'loss':loss}

    with open(f'models/Results/supervised/Predictive_results_{EXPERIMENT_NAME}.csv', 'w') as f:
        w = csv.DictWriter(f, test_results.keys())
        w.writeheader()
        w.writerow(test_results)
    
    plot_history(history, np.mean(Y_val[:, 0]), EXPERIMENT_NAME,
     'images/supervised/')
    
    pd.DataFrame.from_dict(history).to_csv(f'models/history_logs/train_{EXPERIMENT_NAME}.csv')
    pd.DataFrame.from_dict(history_all).to_csv(f'models/history_logs/train+val{EXPERIMENT_NAME}.csv')
    
    #my_model_2.save(f'models/trained_models/Inception_time/{EXPERIMENT_NAME}.h5')
    X = np.vstack((X_train, X_val, X_test))
    G = np.vstack((G_train, G_val, G_test))
    
    my_model_2.save(f'models/trained_models/Supervised/{EXPERIMENT_NAME}.h5')
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
model, X, G = train_Xception()
get_hidden_dim(model, X, G)


