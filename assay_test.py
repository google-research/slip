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

"""Tests for landscape."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import assay


class ConstantLandscape:
    """Evaluates to a constant value."""

    def __init__(self, fitness):
        self._fitness = fitness

    def evaluate(self, sequences):
        return np.array([self._fitness] * sequences.shape[0])


class AssayTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='clip_to_5',
            constant_fitness=10,
            min_fitness_threshold=0,
            max_fitness_threshold=5,
        ),
        dict(
            testcase_name='clip_up_to_0',
            constant_fitness=-10,
            min_fitness_threshold=0,
            max_fitness_threshold=5,
        ),
    )
    def test_thresholded_assay(self, constant_fitness, min_fitness_threshold,
                               max_fitness_threshold):
        sequences = np.array([[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
        mock_landscape = ConstantLandscape(constant_fitness)

        thresholded_assay = assay.ThresholdedAssay(mock_landscape,
                                                   min_fitness_threshold,
                                                   max_fitness_threshold)
        actual_fitnesses = thresholded_assay.evaluate(sequences)
        for actual_fitness in actual_fitnesses:
            self.assertLessEqual(actual_fitness, max_fitness_threshold)
            self.assertGreaterEqual(actual_fitness, min_fitness_threshold)


if __name__ == '__main__':
    absltest.main()
