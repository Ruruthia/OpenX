import argparse
import pickle

import numpy as np
import yaml

from src.models.neural_net import NeuralNet
from src.models.scikit_models import ScikitModel
from src.models.simple_heuristic import SimpleHeuristic
from src.utils import (
    load_dataset,
    split_dataset,
    MAX_EPOCHS,
    plot_training,
    calculate_accuracy,
    plot_confusion_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["simple_heuristic", "neural_net", "knn", "random_forest"],
        help="Type of model " "to train. ",
    )
    args = parser.parse_args()
    model_type = args.model_type

    X, y = load_dataset()
    train_X, test_X, train_y, test_y = split_dataset(X, y)

    if model_type in ["knn", "random_forest"]:
        model = ScikitModel(model_type=model_type)
        model.fit(train_X, train_y)
        with open(f"models/{model_type}.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif model_type == "neural_net":
        with open("configs/neural_net.yaml") as f:
            config = yaml.safe_load(f)
        model = NeuralNet()
        history = model.fit(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            **config,
            max_epochs=MAX_EPOCHS,
            path="models/neural_net.ckpt",
        )
        plot_training(history)

    else:
        model = SimpleHeuristic()

    if model_type == "knn":
        idx = np.random.randint(0, high=test_X.shape[0], size=10000, dtype=int)
        test_X = test_X.iloc[idx]
        test_y = test_y[idx]

    predicted_y = model.predict(test_X)
    accuracy = calculate_accuracy(y=test_y, predicted_y=predicted_y)
    print(f"Model accuracy: {accuracy}")
    plot_confusion_matrix(y=test_y, predicted_y=predicted_y)


if __name__ == "__main__":
    main()
