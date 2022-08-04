from cProfile import label
from sklearn.metrics import matthews_corrcoef, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def predict(model, x_test, y_test, batch_size):
    y_pred = model.predict(x_test, batch_size=batch_size)
    mcc = matthews_corrcoef(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    accuracy = accuracy_score(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_test, y_pred).numpy()
    return mcc, accuracy, loss, y_pred

def plot_history(history, val_prop, exp_name, directory, show = False,):
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(30, 10, forward=True)
    fig.suptitle(exp_name, fontsize=16)
    axs[0].plot(history['loss'], color = 'blue', alpha = 1, label = 'Loss')
    axs[0].plot(history['val_loss'], color = 'red', alpha = 1, label = 'Val. Loss')
    axs[0].grid(True)

    axs[1].plot(history['accuracy'], color = 'blue', alpha = 1)
    axs[1].plot(history['val_accuracy'],  color = 'red', alpha = 1)
    axs[1].hlines(y = val_prop, xmin= 0, xmax = len(history['accuracy']), linestyle = 'dashed', colors= "black", label = 'Proportion of 1 in val.')
    axs[1].grid(True)
    
    axs[2].plot(history['MatthewsCorrelationCoefficient'],  color = 'blue', alpha = 1)
    axs[2].plot(history['val_MatthewsCorrelationCoefficient'], color = 'red', alpha = 1)
    axs[2].grid(True)

    fig.legend()

    plt.savefig(directory + exp_name + '_History.png')
    if show:
        plt.show() 