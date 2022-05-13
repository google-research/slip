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

"""Tests for synthetic_protein_landscapes.utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import utils


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='3 classes',
          class_vector=[1, 0, 2],
          num_classes=3,
          expected=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),
      dict(
          testcase_name='4 classes',
          class_vector=[1, 3],
          num_classes=4,
          expected=np.array([[0, 1, 0, 0], [0, 0, 0, 1]])),
      dict(
          testcase_name='2d in [100, 110]',
          class_vector=np.array([[1, 0, 0], [1, 1, 0]]),
          num_classes=2,
          expected=np.array([[[0, 1], [1, 0], [1, 0]], [[0, 1], [0, 1], [1,
                                                                         0]]]),
      ),
  )
  def test_one_hot(self, class_vector, num_classes, expected):
    np.testing.assert_allclose(
        utils.onehot(class_vector, num_classes), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='no skip',
          seq_start=[1, 2, 3],
          seq_end=[4, 5, 6],
          crossover_idx=3,
          expected=[1, 2, 3]),
      dict(
          testcase_name='full skip',
          seq_start=[1, 2, 3],
          seq_end=[4, 5, 6],
          crossover_idx=0,
          expected=[4, 5, 6]),
      dict(
          testcase_name='1',
          seq_start=[1, 2, 3],
          seq_end=[4, 5, 6],
          crossover_idx=1,
          expected=[1, 5, 6]),
  )
  def test_crossover_at_index(self, seq_start, seq_end, crossover_idx,
                              expected):
    seq_start = np.array(seq_start)
    seq_end = np.array(seq_end)
    expected = np.array(expected)
    recombined_seq = utils._crossover_at_index(seq_start, seq_end,
                                               crossover_idx)
    np.testing.assert_array_equal(recombined_seq, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='recombination distribution',
          seq_start=np.array([0, 0, 0, 0]),
          seq_end=np.array([1, 1, 1, 1]),
          seed=0,
      ),)
  def test_recombine_seqs_dist(self, seq_start, seq_end, seed):
    random_state = np.random.RandomState(seed)

    num_replicates = 1000
    recombined_seqs = []
    for _ in range(num_replicates):
      recombined_seqs.append(
          utils.recombine_seqs(seq_start, seq_end, random_state))
    recombined_seqs = np.vstack(recombined_seqs)

    # should be ~50% 1s in first column
    num_1s_in_first_position = recombined_seqs[:, 0].sum()
    tolerance = 50
    self.assertIn(
        num_1s_in_first_position,
        range(num_replicates // 2 - tolerance, num_replicates // 2 + tolerance))

    # should be ~uniform 1s across columns
    num_1s_in_each_position = recombined_seqs.sum(axis=0)
    freq_1s_in_each_position = num_1s_in_each_position / recombined_seqs.shape[0]
    dist_1s = freq_1s_in_each_position / freq_1s_in_each_position.sum()
    uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(dist_1s, uniform_dist, atol=0.02)

  @parameterized.named_parameters(
      dict(testcase_name='3 hamming', x=[1, 2, 3], y=[4, 5, 6], expected=3),
      dict(testcase_name='2 hamming', x=[1, 2, 3], y=[4, 5, 3], expected=2),
      dict(testcase_name='0 hamming', x=[3, 1, 2], y=[3, 1, 2], expected=0),
  )
  def test_hamming_distance(self, x, y, expected):
    self.assertEqual(utils.hamming_distance(x, y), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='2_combos',
          mutations_a=[(0, 2), (1, 3)],
          mutations_b=[(0, 4), (3, 11)],
          expected_output=[((0, 2), (1, 3), (3, 11)),
                           ((0, 4), (1, 3), (3, 11))]),
      dict(
          testcase_name='easy_add',
          mutations_a=[
              (0, 1),
          ],
          mutations_b=[
              (1, 2),
          ],
          expected_output=[
              ((0, 1), (1, 2)),
          ]),
      dict(
          testcase_name='add_with_duplicate',
          mutations_a=[(0, 1), (1, 3)],
          mutations_b=[
              (1, 2),
          ],
          expected_output=[
              ((0, 1), (1, 2)),
              ((0, 1), (1, 3)),
          ]),
  )
  def test_merge_mutation_sets(self, mutations_a, mutations_b, expected_output):
    actual = utils.merge_mutation_sets(mutations_a, mutations_b)
    self.assertSetEqual(set(actual), set(expected_output))

  @parameterized.parameters(
      ([1, 2, 3], [1, 2, 3], []),
      ([1, 2, 4], [1, 2, 3], [2]),
      ([1, 2, 3], [1, 2, 4], [2]),
      ([2, 2, 3], [1, 2, 4], [0, 2]),
      ([1, 1, 2], [1, 0, 0], [1, 2]),
  )
  def test_get_mutation_positions(self, sequence, parent, expected_output):
    sequence = np.array(sequence)
    parent = np.array(parent)

    actual_output = utils.get_mutation_positions(sequence, parent)
    self.assertEqual(list(actual_output), expected_output)

  def test_get_mutation_positions_unequal_lengths(self):
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    with self.assertRaisesRegex(AssertionError, 'equal length'):
      utils.get_mutation_positions(a, b)

  @parameterized.parameters(
      ([0, 1, 2], [0, 1, 2], []),
      ([0, 1, 3], [0, 1, 2], [(2, 3)]),
      ([0, 1, 2], [0, 1, 3], [(2, 2)]),
      ([1, 1, 2], [0, 1, 3], [(0, 1), (2, 2)]),
      ([1, 1, 2], [1, 0, 0], [(1, 1), (2, 2)]),
  )
  def test_get_mutations(self, sequence, parent, expected_output):
    sequence = np.array(sequence)
    parent = np.array(parent)

    actual_output = utils.get_mutations(sequence, parent)
    self.assertEqual(actual_output, expected_output)

  def test_get_mutations_unequal_lengths(self):
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    with self.assertRaisesRegex(AssertionError, 'equal length'):
      utils.get_mutations(a, b)

  @parameterized.parameters(
      ([], [0, 1, 2]),
      ([(0, 1)], [1, 1, 2]),
      ([(0, 1), (2, 0)], [1, 1, 0]),
  )
  def test_apply_mutations(self, mutations, expected_output):
    parent = np.array([0, 1, 2])
    expected_output = np.array(expected_output)
    actual_output = utils.apply_mutations(parent, mutations)
    np.testing.assert_allclose(actual_output, expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='additive',
          seq_a=[1, 0, 0],
          seq_b=[0, 0, 2],
          ref_seq=[0, 0, 0],
          expected=[[1, 0, 2]]),
      dict(
          testcase_name='duplicate_position',
          seq_a=[1, 1, 0],
          seq_b=[0, 2, 2],
          ref_seq=[0, 0, 0],
          expected=[[1, 1, 2], [1, 2, 2]]),
      dict(
          testcase_name='additive_2_positions',
          seq_a=[1, 0],
          seq_b=[0, 2],
          ref_seq=[0, 0],
          expected=[[1, 2]]),
      dict(
          testcase_name='2_positions',
          seq_a=[1, 1],
          seq_b=[2, 2],
          ref_seq=[0, 0],
          expected=[[1, 1], [1, 2], [2, 1], [2, 2]]),
      dict(
          testcase_name='combine_with_ref',
          seq_a=[0, 0],
          seq_b=[2, 2],
          ref_seq=[0, 0],
          expected=[[2, 2]]),
      dict(
          testcase_name='all_combos',
          seq_a=[1, 1, 1],
          seq_b=[2, 0, 2],
          ref_seq=[0, 0, 0],
          expected=[[1, 1, 2], [2, 1, 1], [1, 1, 1], [2, 1, 2]]),
  )
  def test_add_seqs(self, seq_a, seq_b, ref_seq, expected):
    seq_a = np.array(seq_a)
    seq_b = np.array(seq_b)
    ref_seq = np.array(ref_seq)

    combined_seqs = utils.add_seqs(seq_a, seq_b, ref_seq)

    expected_set = set(tuple(s) for s in expected)
    actual_set = set(tuple(s) for s in combined_seqs)

    self.assertSetEqual(actual_set, expected_set)

  @parameterized.named_parameters(
      dict(
          testcase_name='unequal_lengths',
          seq_a=[1, 0, 0, 1],
          seq_b=[0, 0, 2],
          ref_seq=[0, 0, 0],
      ),
      dict(
          testcase_name='unequal_ref_length',
          seq_a=[1, 0, 0],
          seq_b=[0, 0, 2],
          ref_seq=[0, 0, 0, 0],
      ),
  )
  def test_add_seqs_wrong_length(self, seq_a, seq_b, ref_seq):
    seq_a = np.array(seq_a)
    seq_b = np.array(seq_b)
    ref_seq = np.array(ref_seq)

    with self.assertRaisesRegex(AssertionError, 'equal length'):
      utils.add_seqs(seq_a, seq_b, ref_seq)


class TensorUtilsTest(parameterized.TestCase):
  # 2x2x3x3
  mock_tensor = np.array([
                    [
                        [
                            [-10, 1, 1],
                            [1, 8, 1],
                            [2, 0, 1],
                        ],
                        [
                            [0, 1, 6],
                            [1, 1, -10],
                            [2, 0, 0],
                        ],
                    ],
                    [
                        [
                            [0, 1, 1],
                            [1, 1, 1],
                            [0, 0, 10],
                        ],
                        [
                            [0, 1, 6],
                            [1, 1, 1],
                            [2, 0, -9],
                        ],
                    ]
                ])

  def _get_tensor_idx_from_pair(self, pair):
      return (pair[0][0], pair[1][0], pair[0][1], pair[1][1])


  def test_get_top_n_mutation_pairs(self):
    best_interactions = utils.get_top_n_mutation_pairs(self.mock_tensor, 2, lowest=False)
    best_pair = best_interactions[0]
    self.assertEqual(self.mock_tensor[self._get_tensor_idx_from_pair(best_pair)], 10)

    second_best_pair = best_interactions[1]
    self.assertEqual(self.mock_tensor[self._get_tensor_idx_from_pair(second_best_pair)], 8)

  def test_get_top_n_mutation_pairs_lowest(self):
    worst_interactions = utils.get_top_n_mutation_pairs(self.mock_tensor, 3, lowest=True)
    self.assertEqual(self.mock_tensor[self._get_tensor_idx_from_pair(worst_interactions[0])], -10)
    self.assertEqual(self.mock_tensor[self._get_tensor_idx_from_pair(worst_interactions[1])], -10)
    self.assertEqual(self.mock_tensor[self._get_tensor_idx_from_pair(worst_interactions[2])], -9)

if __name__ == '__main__':
  absltest.main()
