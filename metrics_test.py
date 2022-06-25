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

"""Tests for synthetic_protein_landscapes.metrics.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd

import metrics


class MetricsTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='2_clusters',
            sequences=[[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=2,
        ),
        dict(
            testcase_name='3_clusters',
            sequences=[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]],
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=3,
        ),
        dict(
            testcase_name='1_cluster',
            sequences=[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]],
            max_intra_cluster_hamming_distance=2,
            expected_num_clusters=1,
        ),
    )
    def test_num_clusters(self, sequences, max_intra_cluster_hamming_distance,
                          expected_num_clusters):
        sequences = np.vstack(sequences)
        pdist = metrics.pairwise_hamming_distance(sequences)
        num_clusters = metrics.num_clusters(
            pdist,
            max_intra_cluster_hamming_distance=max_intra_cluster_hamming_distance)
        self.assertEqual(num_clusters, expected_num_clusters)

    @parameterized.named_parameters(
        dict(
            testcase_name='1_cluster',
            sequences=[[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            fitnesses=[0, 1, 1],
            min_fitness=0.5,
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=1,
        ),
        dict(
            testcase_name='1_cluster_1_hit',
            sequences=[[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            fitnesses=[0, 1, 0],
            min_fitness=0.5,
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=1,
        ),
        dict(
            testcase_name='2_clusters',
            sequences=[[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            fitnesses=[1, 1, 1],
            min_fitness=0.5,
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=2,
        ),
        dict(
            testcase_name='no_clusters',
            sequences=[[0, 0, 0, 0], [1, 2, 0, 0], [1, 2, 0, 0]],
            fitnesses=[0, 0, 0],
            min_fitness=0.5,
            max_intra_cluster_hamming_distance=1,
            expected_num_clusters=0,
        ),
    )
    def test_num_clusters_for_min_fitness(self, sequences, fitnesses, min_fitness,
                                          max_intra_cluster_hamming_distance,
                                          expected_num_clusters):
        df = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        num_clusters = metrics.num_clusters_for_min_fitness(
            df,
            min_fitness=min_fitness,
            max_intra_cluster_hamming_distance=max_intra_cluster_hamming_distance)
        self.assertEqual(num_clusters, expected_num_clusters)


if __name__ == '__main__':
    absltest.main()
