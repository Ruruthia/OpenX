import pandas as pd


def _heuristic(point: pd.Series) -> int:
    """
    Given a datapoint, assigns class based on location (encoded as one-hot in features 11-14).
    """
    # Neota
    if point[11]:
        # Spruce-Fir
        return 0
    # Rawah or Comanche
    if point[10] or point[12]:
        # Lodgepole Pine
        return 1
    # Cache la Poudre
    else:
        # Ponderosa Pine
        return 2


class SimpleHeuristic:

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(_heuristic, axis=1)
