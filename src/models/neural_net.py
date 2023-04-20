import numpy as np
import pandas as pd
import tensorflow as tf


class NeuralNet:

    def __init__(self) -> None:
        self._model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(7)
        ])

    def load_from_checkpoint(self, path: str) -> None:
        self._model = tf.keras.models.load_model(path)

    def fit(self, train_X: pd.DataFrame, train_y: np.ndarray, test_X: pd.DataFrame, test_y: np.ndarray,
            learning_rate: float, batch_size: int, max_epochs: int,
            path=None) -> tf.keras.callbacks.History:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
        self._model.compile(optimizer=optimizer,
                            loss=loss_fn,
                            metrics=['accuracy'])
        history = self._model.fit(train_X, train_y, validation_data=(test_X, test_y),
                                  epochs=max_epochs, batch_size=batch_size, callbacks=[stop_early])
        if path is not None:
            self._model.save(path)
        return history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logits = self._model.predict(X)
        return np.argmax(logits, axis=1)
