import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier


class ScikitModel:
    def __init__(self, model_type: str) -> None:
        if model_type == "knn":
            self._clf = neighbors.KNeighborsClassifier(n_neighbors=15)
        elif model_type == "random_forest":
            self._clf = RandomForestClassifier()
        else:
            raise NotImplementedError("The only implemented classifiers are knn and random_forest!")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._clf.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict(X)
