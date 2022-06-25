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

"""Tests for synthetic_protein_landscapes.sampling.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from scipy.spatial import distance as spd

import sampling


class SamplingTest(parameterized.TestCase):

    @parameterized.parameters(
        (0, 0),
        (0, 9),
        (3, 7),
        (1, 9),
    )
    def test_sample_within_hamming_radius(self, min_mutations, max_mutations):
        vocab_size = 4
        num_samples = 27
        seq_length = 9
        sequence = np.random.randint(0, vocab_size, seq_length)
        samples = sampling.sample_within_hamming_radius(
            sequence,
            num_samples,
            vocab_size,
            min_mutations=min_mutations,
            max_mutations=max_mutations)
        dists = spd.cdist([sequence], samples, metric='hamming')[0]
        num_mutations = dists * seq_length
        np.testing.assert_allclose((num_samples, len(sequence)), samples.shape)
        self.assertTrue(
            all([min_mutations <= n <= max_mutations for n in num_mutations]))

    @parameterized.parameters((0,), (33,))
    def test_sample_within_hamming_radius_random_state(self, seed):

        def _sample(seed):
            random_state = np.random.RandomState(seed)
            return sampling.sample_within_hamming_radius([0, 1, 0, 1],
                                                         10,
                                                         4,
                                                         min_mutations=1,
                                                         max_mutations=4,
                                                         random_state=random_state)

        np.testing.assert_allclose(_sample(seed), _sample(seed))

    @parameterized.named_parameters(
        dict(
            testcase_name='one_position',
            reference_sequence=[0],
            vocab_size=5,
            expected=[[1], [2], [3], [4]],
        ),
        dict(
            testcase_name='two_position',
            reference_sequence=[0, 0],
            vocab_size=3,
            expected=[[0, 1], [0, 2], [1, 0], [2, 0]],
        ),
        dict(
            testcase_name='two_position_different',
            reference_sequence=[0, 2],
            vocab_size=3,
            expected=[[0, 1], [0, 0], [1, 2], [2, 2]],
        ))
    def test_get_all_single_mutants(self, reference_sequence, vocab_size,
                                    expected):
        all_singles = sampling.get_all_single_mutants(reference_sequence,
                                                      vocab_size)

        actual_set = set(tuple(s) for s in all_singles)
        expected_set = set(tuple(s) for s in expected)

        self.assertSetEqual(actual_set, expected_set)


if __name__ == '__main__':
    absltest.main()
