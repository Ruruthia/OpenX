import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset() -> (pd.DataFrame, pd.DataFrame):
    full_df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
        header=None
    )
    return full_df.iloc[:, :-1], full_df.iloc[:, -1:],


def split_dataset(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) \
        -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size)
    return train_X, test_X, np.ravel(train_y), np.ravel(test_y)


def accuracy(y: pd.Series, predicted_y: pd.Series) -> float:
    return np.mean(y == predicted_y)[0]
