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

"""Tests for models."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import models
import utils


class ModelsTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='one_sequence',
            sequences=[[1, 0, 0]],
        ),
        dict(
            testcase_name='two_sequences',
            sequences=[[0, 0, 0], [1, 0, 0]],
        ),
        dict(
            testcase_name='three_sequences',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
        ),
    )
    def test_fit_predict(self, sequences):
        sequence_length = 3
        vocab_size = 3
        for model_type in ['linear', 'cnn']:
            model_kwargs = dict(cnn_num_filters=2,
                                cnn_kernel_size=1,
                                cnn_hidden_size=12,
                                cnn_batch_size=1,
                                cnn_num_epochs=1)
            model, flatten_inputs = models.get_model(
                model_type,
                sequence_length,
                vocab_size,
                model_kwargs)
            if flatten_inputs:
                x = utils.one_hot_and_flatten(np.array(sequences), num_classes=vocab_size)
            else:
                x = utils.onehot(np.array(sequences), num_classes=vocab_size)
        y = np.ones(len(sequences))
        model.fit(x, y)
        output_shape = (len(sequences),)
        self.assertEqual(model.predict(x).shape, output_shape)


if __name__ == '__main__':
    absltest.main()
