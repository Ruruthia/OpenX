import pandas as pd


def load_dataset() -> (pd.DataFrame, pd.DataFrame):
    full_df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
        header=None
    )
    return full_df.iloc[:, :-1], full_df.iloc[:, -1:],
