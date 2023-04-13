import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

MAX_EPOCHS = 15


def load_dataset() -> (pd.DataFrame, pd.DataFrame):
    full_df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
        header=None
    )
    return full_df.iloc[:, :-1], full_df.iloc[:, -1:],


def split_dataset(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) \
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size)
    
    # we want the labels to start from 0
    return train_X, test_X, np.ravel(train_y - 1) , np.ravel(test_y - 1)


def calculate_accuracy(y: pd.Series, predicted_y: pd.Series) -> float:
    return np.mean(y == predicted_y)


def plot_training(history: tf.keras.callbacks.History) -> None:
    for metric in ['accuracy', 'loss']:
        plt.plot(history.history[f'{metric}'])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'model {metric}')
        plt.ylabel(f'{metric}')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def plot_confusion_matrix(y: pd.Series, predicted_y: pd.Series) -> None:
    cm = metrics.confusion_matrix(y_true=y, y_pred=predicted_y)
    sn.heatmap(cm, annot=True, fmt='g')
    plt.show()
