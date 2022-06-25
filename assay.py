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

"""Modified landscapes."""
import abc
from typing import Optional

import numpy as np

import potts_model


class Assay(abc.ABC):
    """Assay base class.

    An Assay is a view on an underlying fitness landscape, intended to simulate
    experimental (i.e. in-vitro) assessments of a sequence. Assays induce an
    arbitrary transformation on the underlying fitness function. For example,
    they can add noise, have a limited dynamic range, and/or convert floating
    point fitnesses into discrete counts via a stochastic process.
    """

    def __init__(self, landscape):
        self._landscape = landscape

    @abc.abstractmethod
    def evaluate(self, sequences):
        """Returns the assayed function of a set of sequences, input as  a 2D array.

        Args:
          sequences: A 2D array of integer encoded sequences.

        Returns:
          A 1D np.ndarray.
        """


class ThresholdedAssay(Assay):
    """An assay where fitness cannot be below and/or above a threshold."""

    def __init__(self, landscape,
                 min_fitness_threshold,
                 max_fitness_threshold):
        super().__init__(landscape)
        self._min_fitness_threshold = min_fitness_threshold
        self._max_fitness_threshold = max_fitness_threshold

    def evaluate(self, sequences):
        fitnesses = self._landscape.evaluate(sequences)
        clipped_fitnesses = np.clip(
            fitnesses,
            a_min=self._min_fitness_threshold,
            a_max=self._max_fitness_threshold)
        return clipped_fitnesses
