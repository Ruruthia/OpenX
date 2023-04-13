from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from src.models.neural_net import NeuralNet
from src.utils import calculate_accuracy, load_dataset, split_dataset

MAX_EPOCHS = 15
X, y = load_dataset()
train_X, test_X, train_y, test_y = split_dataset(X, y)

space = {
    'learning_rate': hp.choice('learning_rate', [0.0001, 0.0005, 0.001, 0.005]),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256]),
}


def objective(params):
    model = NeuralNet(batch_size=params['batch_size'], learning_rate=params['learning_rate'], max_epochs=MAX_EPOCHS)
    model.fit(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
    predicted_y = model.predict(test_X)
    accuracy = calculate_accuracy(y=test_y, predicted_y=predicted_y)
    return {'loss': -accuracy, 'status': STATUS_OK}


def main():

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, verbose=True, max_evals=1)
    print('Best: ', best)


if __name__ == '__main__':
    main()
