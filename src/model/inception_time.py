## Copied and adapted from https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py .
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#### PARSE COMMAND LINE CONFIG:
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

EXPERIMENT_NAME = f'IT_C{cols_name}_D{my_args["d"]}_GS{my_args["gs"]}\
_snp{my_args["snp"]}_BS{my_args["bs"]}'#_{my_time}'
print('----------------------------------------------------------------------')
print(EXPERIMENT_NAME)
print('----------------------------------------------------------------------')

##### LOAD DATA
## No stratified:
if  my_args["gs"] == 0 :
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
class inception_time_model:

    def __init__(self,
            input_shape,
            bottlneck_size = 32,
            nb_filters=32,
            use_residual=True,
            use_bottleneck=True,
            depth=6,
            kernel_size=41,
            dropout = 0,
            nb_classes = 2,
            my_optimizer = None, 
            lr = 0.001,
            beta_1= 0.9,
            beta_2= 0.99):

            self.my_optimizer = my_optimizer
            self.nb_filters = nb_filters
            self.use_residual = use_residual
            self.use_bottleneck = use_bottleneck
            self.depth = depth
            self.kernel_size = kernel_size - 1
            self.bottleneck_size = bottlneck_size
            self.dropout = dropout
            self.lr = lr
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            

            self.model = self.build_model(input_shape, nb_classes)
    
    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Dropout(self.dropout)(x) # I don't like this
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor]) # basically sums with a list of tensors of the same shape
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes = 2):
        '''
        Just a sequential model. Returns the model.
        '''
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth): #Number of inception modules

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2: #The residual conection goes from 2 to 5 to 8...
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        
        ########## END OF ARCHITECTURE ########
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        if self.my_optimizer == None:
            self.my_optimizer = keras.optimizers.Adam(
                            learning_rate = self.lr,
                            beta_1= self.beta_1,
                            beta_2= self.beta_2,
                            epsilon=1e-07,
                            amsgrad=False,
                            name='Adam')

        model.compile(loss='categorical_crossentropy', optimizer= self.my_optimizer,
                      metrics=['accuracy', tfa.metrics.MatthewsCorrelationCoefficient(num_classes= 2)])

        return model


#### HYPERPARAMETER TUNER
def tune_inception_time_model(num_trials, epochs):

    def model_builder(hp):
            hp_depth = hp.Int(name = 'depth', min_value = 1, max_value = 4)
            hp_dropout = hp.Choice (name = 'dropout', values = [0., 0.15, 0.5])
            hp_lr = hp.Float(name= 'lr', min_value = 1e-6, max_value = 0.001, sampling = 'log')
            hp_b1 = hp.Float(name= 'beta_1', min_value = 0.2, max_value = 0.999, sampling = 'log')
            hp_b2 = hp.Float(name= 'beta_2', min_value = 0.2, max_value = 0.999, sampling = 'log')
            hp_kernel_size = hp.Int(name = 'kernel_size', min_value = 16, max_value = 61, sampling = 'log')
            hp_filter_size = hp.Int(name = 'nb_filters', min_value = 10, max_value = 64, sampling = 'log')
            hp_bottleneck_size = hp.Int(name = 'bottleneck_size', min_value = 5, max_value = 64, sampling = 'log')

            model_build = inception_time_model(input_shape= (X_train.shape[1], X_train.shape[2]),
                                                nb_classes= 2,
                                                depth = hp_depth,
                                                nb_filters = hp_filter_size,
                                                kernel_size = hp_kernel_size,
                                                bottlneck_size = hp_bottleneck_size, 
                                                dropout = hp_dropout,
                                                lr = hp_lr,
                                                beta_1= hp_b1,
                                                beta_2= hp_b2).model

            return model_build
    
    tuner = kt.BayesianOptimization(model_builder,
                objective='val_loss',
                max_trials = num_trials,
                directory= 'models/hyper_search/tuner_logs/supervised',
                project_name= EXPERIMENT_NAME)

    tuner.search(X_train, Y_train, batch_size= my_args["bs"], epochs= epochs,
                               verbose= True, validation_data=(X_val, Y_val),
                                callbacks = [
                                    tf.keras.callbacks.TensorBoard(
                                        f"models/hyper_search/tuner_logs/supervised/{EXPERIMENT_NAME}_logs")])


    best_hps_raw = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_hps = {x:best_hps_raw.get(x) for x in ['depth', 'dropout', 'lr', 'beta_1', 'beta_2', 'kernel_size', 'nb_filters', 'bottleneck_size']}
    print(best_hps)
    pd.DataFrame.from_dict(
        best_hps, orient="index").to_csv(
            f"models/hyper_search/best_hyper/supervised/{EXPERIMENT_NAME}_hyperparam.csv")
    return(best_hps)

#### TRAIN
def train_inception_time(path_to_hyper):
    # Placeholder predictions:
    #e_y_pred = []
    # Train 5 models.
    for e in range(5):
        print(F' TRAINING MODEL {e}')
        tf.keras.backend.clear_session()
        my_hp = pd.read_csv(path_to_hyper, index_col = 0)
        #print(my_hp)
        depth = int(my_hp.loc['depth'][0])
        dropout = float(my_hp.loc['dropout'][0])
        lr = float(my_hp.loc['lr'][0])
        beta_1 = float(my_hp.loc['beta_1'][0])
        beta_2 = float(my_hp.loc['beta_2'][0])
        kernel_size = int(my_hp.loc['kernel_size'][0])
        nb_filters = int(my_hp.loc['nb_filters'][0])
        bottleneck_size = int(my_hp.loc['bottleneck_size'][0])

        my_optimizer = None
        # IMPORTANT CONSTANTS: Fixed for all experiments
        n_epochs = 150
        patience = 20
        
        my_model_1 = inception_time_model(input_shape= (X_train.shape[1], X_train.shape[2]),
                                nb_classes= 2,
                                depth = depth,
                                nb_filters = nb_filters,
                                kernel_size = kernel_size,
                                bottlneck_size = bottleneck_size,
                                dropout = dropout,
                                my_optimizer = my_optimizer,
                                lr = lr,
                                beta_1= beta_1,
                                beta_2= beta_2).model
        

        #print(my_model_1.summary())
        
        
        
        #Get number of epochs.
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

        # Retrain using the optimal number of epochs:
        my_model_2 = inception_time_model(input_shape= (X_train.shape[1], X_train.shape[2]),
                                nb_classes= 2,
                                depth = depth,
                                nb_filters = nb_filters,
                                kernel_size = kernel_size,
                                bottlneck_size = bottleneck_size,
                                dropout = dropout,
                                my_optimizer = my_optimizer,
                                lr = lr,
                                beta_1= beta_1,
                                beta_2= beta_2).model

        my_all_X = np.vstack((X_train, X_val))
        my_all_Y = np.vstack((Y_train, Y_val))

        history_all = my_model_2.fit(x = my_all_X,
                        y = my_all_Y,
                        epochs = my_number_of_epochs,
                        batch_size = my_args["bs"],
                        verbose = False
                        ).history
        
        mcc, accuracy, loss, y_pred = predict(my_model_2, X_test, Y_test, batch_size = my_args["bs"])

        #e_y_pred.append(y_pred[:, 0])
        print('Test performance: ', {'mcc': mcc , 'acc': accuracy, 'loss': loss})
        test_results = {'num' : e, 'mcc':mcc , 'acc':accuracy, 'loss':loss}

        #pd.DataFrame.from_dict(test_results).to_csv(f'models/Reuslts/supervised/Predictive_results_{EXPERIMENT_NAME}.csv')
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
    print('Generating hidden dimension...')
    extractor = tf.keras.Model(inputs= model.inputs,
                        outputs= model.layers[-2].output)

    Inner_dim = extractor.predict(X, batch_size = my_args["bs"])

    gen_dim = pd.DataFrame(np.hstack((G, Inner_dim)))
    gen_dim.columns = ['Genotype'] + [f'Dim_{i}' for i in range(Inner_dim.shape[1])]
    gen_dim.to_csv(
        f'data/processed/Hidden_representations/supervised/{EXPERIMENT_NAME}.csv',
         index=False)
    print(gen_dim)

def _more_metrics():
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score

    ensemble_performance = pd.read_csv(f'models/Results/supervised/Predictive_results_{EXPERIMENT_NAME}.csv')
    row_max = np.argmax(ensemble_performance['acc'])
    e = ensemble_performance['num'].iloc[row_max]
    my_model_2 = tf.keras.models.load_model(f'models/trained_models/Supervised/{EXPERIMENT_NAME}_{e}.h5')
    y_hat = my_model_2.predict(X_test, batch_size = 10)
    report = classification_report(Y_test[:, 1], np.argmax(y_hat, 1),  output_dict=True)

    #print(report)
    pd.DataFrame(report).transpose().to_csv(f'results/Supervised/{EXPERIMENT_NAME}_class_report.csv')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_hat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(
        fpr[1],
        tpr[1],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[1],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic\n Supervised SNP 86001")
    plt.legend(loc="lower right")
    plt.savefig(f'results/Supervised/{EXPERIMENT_NAME}_ROC_AUC')
    #plt.show()

    # Weights las layer:
    pd.DataFrame(my_model_2.layers[-1].get_weights()[0]).to_csv(f'results/Supervised/{EXPERIMENT_NAME}_ll_weights.csv')
    pd.DataFrame(my_model_2.layers[-1].get_weights()[1]).to_csv(f'results/Supervised/{EXPERIMENT_NAME}_ll_biases.csv')
    print('Weights extracted!')
    #pdb.set_trace()

#_more_metrics()
### EXECUTE:
#tune_inception_time_model(50, 50)
model, X, G = train_inception_time(f'models/hyper_search/best_hyper/supervised/{EXPERIMENT_NAME}_hyperparam.csv')
get_hidden_dim(model, X, G)

#model = inception_time_model(input_shape= (X_train.shape[1], X_train.shape[2]), depth = 1).model
#dot_img_file = 'IT_Depth3.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

