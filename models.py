# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model utilities."""

import functools
from typing import Callable, Dict

from sklearn import ensemble
from sklearn import linear_model
import tensorflow as tf


class KerasModelWrapper:
    """Wraps a Keras model to have the sklearn model interface."""

    def __init__(self,
                 model_build_fn: Callable,
                 sequence_length: int,
                 vocab_size: int,
                 fit_kwargs: Dict = dict()):
        """Initialize a KerasModelWrapper.

        Args:
          model_build_fn: A function that when called with arguments
            `model_build_fn(sequence_length, vocab_size)` returns a Keras model.
          sequence_length: The length of input sequences.
          vocab_size: The one-hot dimension size for input sequences.
          fit_kwargs: An optional dictionary of keyword arguments passed to the
            Keras model.fit(**fit_kwargs). See
              https://keras.io/api/models/model_training_apis/ for more details.
        """
        self._model_build_fn = model_build_fn
        self._fit_kwargs = fit_kwargs
        self._sequence_length = sequence_length
        self._vocab_size = vocab_size

    # We capitalize .fit(X, y) and .predict(X) to reflect the sklearn API
    # pylint: disable=invalid-name
    def fit(self, X, y):
        # Reinitialize the model for each call to .fit().
        self._model = self._model_build_fn(
            self._sequence_length, self._vocab_size)
        self._model.fit(X, y, **self._fit_kwargs)

    def predict(self, X):
        return self._model.predict(x=X).squeeze(axis=1)

    # pylint: enable=invalid-name


def build_cnn_model(sequence_length: int,
                    vocab_size: int,
                    num_filters: int,
                    kernel_size: int,
                    hidden_size: int,
                    adam_learning_rate: float):
    """Returns a 1D CNN model.

    This model consists of 3 layers of 1D convs, followed by a dense layer.
    The optimizer is configured to be Adam.

    For example, for an input sequence of length 118, with vocab size 20, 32 filters,
    kernel_size=5, hidden_size=64, model.summary() returns:

      Layer (type)                 Output Shape              Param #
    =================================================================
    conv1d_78 (Conv1D)           (None, 118, 32)           3232
    _________________________________________________________________
    conv1d_79 (Conv1D)           (None, 118, 32)           5152
    _________________________________________________________________
    conv1d_80 (Conv1D)           (None, 118, 32)           5152
    _________________________________________________________________
    flatten_24 (Flatten)         (None, 3776)              0
    _________________________________________________________________
    dense_53 (Dense)             (None, 64)                241728
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 64)                0
    _________________________________________________________________
    dense_54 (Dense)             (None, 1)                 65
    =================================================================
    Total params: 255,329
    Trainable params: 255,329
    Non-trainable params: 0

    Args:
      sequence_length: The input sequence length.
      vocab_size: The dimension of the 1-hot encoding.
    """
    model = tf.keras.models.Sequential()
    input_shape = (sequence_length, vocab_size)
    dropout_prob = 0.25

    model.add(tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', input_shape=input_shape, padding='same'))
    model.add(tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(num_filters, kernel_size, activation='relu', padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_prob, seed=0))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=adam_learning_rate),
        loss='mse',
        metrics=['mse'])
    return model


def build_linear_model(model_kwargs):
    # set defaults
    default_kwargs = {
        'ridge_alpha': 1.0,
        'ridge_fit_intercept': True,
    }
    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = linear_model.Ridge(alpha=kwargs['ridge_alpha'], fit_intercept=kwargs['ridge_fit_intercept'])
    flatten_inputs = True
    return model, flatten_inputs


def build_cnn(sequence_length, vocab_size, model_kwargs):
    default_kwargs = {
        'cnn_batch_size': 64,
        'cnn_num_epochs': 500,
        'cnn_num_filters': 32,
        'cnn_kernel_size': 5,
        'cnn_hidden_size': 64,
        'cnn_adam_learning_rate': 0.0001
    }

    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]
    build_model = functools.partial(build_cnn_model,
                                    num_filters=kwargs['cnn_num_filters'],
                                    kernel_size=kwargs['cnn_kernel_size'],
                                    hidden_size=kwargs['cnn_hidden_size'],
                                    adam_learning_rate=kwargs['cnn_adam_learning_rate'])
    fit_kwargs = {'batch_size': kwargs['cnn_batch_size'], 'epochs': kwargs['cnn_num_epochs']}
    model = KerasModelWrapper(build_model, sequence_length, vocab_size, fit_kwargs)
    flatten_inputs = False
    return model, flatten_inputs


def get_model(model_name,
              sequence_length: int,
              vocab_size: int,
              model_kwargs: Dict):
    """Returns model, flatten_inputs."""
    if model_name == 'linear':
        return build_linear_model(model_kwargs)
    elif model_name == 'cnn':
        return build_cnn(sequence_length, vocab_size, model_kwargs)
    elif model_name == 'random_forest':
        flatten_inputs = True
        return ensemble.RandomForestRegressor(), flatten_inputs
    else:
        raise NotImplementedError
