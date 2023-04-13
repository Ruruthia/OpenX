# OpenX internship task

Python project to compare performance of simple heuristic, two simple ML models and basic neural network on Covertype dataset.

## Models

### Simple heuristic

Assigns one of 4 classes (out of 7) based on location features.

Model accuracy: 0.535

[Confusion matrix](https://github.com/Ruruthia/OpenX/blob/master/images/simple_heuristic_confusion_matrix.png?raw=true)

As expected, does not perform very well.

### Random Forest

Model accuracy: 0.941

[Confusion matrix](https://github.com/Ruruthia/OpenX/blob/master/images/random_forest_confusion_matrix.png?raw=true)

### KNN

Model accuracy: 0.95

[Confusion matrix](https://github.com/Ruruthia/OpenX/blob/master/images/knn_confusion_matrix.png?raw=true)

### NN

Very small & simple neural network. Trained only for 15 epochs. Results could probably be much better given longer training times & more advanced architecture.
In current state, it performs worse than KNNs and Random Forest.

Model accuracy: 0.910

[Confusion matrix](https://github.com/Ruruthia/OpenX/blob/master/images/neural_net_confusion_matrix.png?raw=true)

[Accuracy during training](https://github.com/Ruruthia/OpenX/blob/master/images/neural_net_accuracy.png?raw=true)

[Loss during training](https://github.com/Ruruthia/OpenX/blob/master/images/neural_net_loss.png?raw=true)

## How to run the code

Install the dependencies and the package:
```
pip install -r requirements.txt 
pip install -e .
```

Then, train the models:
```
python scripts/train_and_evaluate.py --model-type simple_heuristic
python scripts/train_and_evaluate.py --model-type knn
python scripts/train_and_evaluate.py --model-type random_forest
python scripts/train_and_evaluate.py --model-type neural_net
```

You can also optimize the hyperparameters for the neural network by running the `scripts/optimize_hyperparams.py` script.

To run the server you can either run the `src/run_server.py` file or use a Docker container:
```
docker build -t my-app .
docker run -p 5000:5000 my-app
```

To test the server, run the `scr/test_server.py` file.
