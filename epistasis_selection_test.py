"""Tests for epistasis_selection.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd


import epistasis_selection
import potts_model

class SelectionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='4_mutants',
          epistatic_pairs=[((1, 1), (2, 2)),
                           ((3, 3), (4, 4)),
                           ((5, 5), (6, 6)),
                           ((7, 7), (8, 8))],
          distance=4,
          num_rounds=1,
          expected_set=[
              ((1, 1), (2, 2), (3, 3), (4, 4)),
              ((1, 1), (2, 2), (5, 5), (6, 6)),
              ((1, 1), (2, 2), (7, 7), (8, 8)),
              ((3, 3), (4, 4), (5, 5), (6, 6)),
              ((3, 3), (4, 4), (7, 7), (8, 8)),
              ((5, 5), (6, 6), (7, 7), (8, 8)),
          ]
      ),
      dict(
          testcase_name='8_mutants',
          epistatic_pairs=[((1, 1), (2, 2)),
                           ((3, 3), (4, 4)),
                           ((5, 5), (6, 6)),
                           ((7, 7), (8, 8))],
          distance=8,
          num_rounds=3,
          expected_set=[
              ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8))
          ]
      ),
  )
  def test_combine_k_rounds(self, num_rounds, distance, epistatic_pairs, expected_set):
      actual_mutants = epistasis_selection.combine_k_rounds(num_rounds=num_rounds, mutations=epistatic_pairs)
      actual_mutants_at_distance = [m for m in actual_mutants if len(m)==distance]
      self.assertSetEqual(set(actual_mutants_at_distance), set(expected_set))

  @parameterized.named_parameters(
      dict(
          testcase_name='add_pairs',
          epistatic_pairs=[((1, 1), (2, 2)),
                           ((2, 2), (3, 3)),
                           ((5, 5), (6, 6)),
                           ((6, 6), (8, 8))],
          distance=4,
          num_rounds=1,
          expected_set=[
              ((1, 1), (2, 2), (5, 5), (6, 6)),
              ((2, 2), (3, 3), (5, 5), (6, 6)),
              ((1, 1), (2, 2), (6, 6), (8, 8)),
              ((2, 2), (3, 3), (6, 6), (8, 8)),
          ]
      ),
      dict(
          testcase_name='cant_use_one',
          epistatic_pairs=[((1, 1), (2, 2)),
                           ((3, 3), (4, 4)),
                           ((4, 4), (5, 5)),  # this overlaps with everything
                           ((5, 5), (6, 6)),
                           ((7, 7), (8, 8))],
          num_rounds=3,
          distance=8,
          expected_set=[
              ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)),
          ]
      ),
      dict(
          testcase_name='no_mutants',
          epistatic_pairs=[((1, 1), (2, 2)),
                           ((3, 3), (4, 4)),
                           ((4, 4), (5, 5)),
                           ((7, 7), (8, 8))],
          num_rounds=3,
          distance=8,
          expected_set=[]
      ),
  )
  def test_combine_k_rounds_overlap(self, num_rounds, distance, epistatic_pairs, expected_set):
      actual_mutants = epistasis_selection.combine_k_rounds(
          num_rounds=num_rounds, mutations=epistatic_pairs)
      # only test a particular distance
      actual_mutants_at_distance = [
          m for m in actual_mutants if len(m) == distance]
      self.assertSetEqual(set(actual_mutants_at_distance), set(expected_set))

  @parameterized.named_parameters(
      dict(
          testcase_name='limit_1_pairs',
          mutation_sets=[((1, 1), (2, 2)),
                           ((1, 1), (4, 4)),
                           ((1, 1), (6, 6)),
                           ((7, 7), (8, 8))],
          limit=1,
          expected_set=[((1, 1), (2, 2)),
                        ((7, 7), (8, 8))],
      ),
      dict(
          testcase_name='limit_2_pairs',
          mutation_sets=[((1, 1), (2, 2)),
                           ((1, 1), (4, 4)),
                           ((1, 1), (6, 6)),
                           ((7, 7), (8, 8))],
          limit=2,
          expected_set=[((1, 1), (2, 2)),
                        ((1, 1), (4, 4)),
                        ((7, 7), (8, 8))],
      ),
  )
  def test_filter_mutation_set_by_position(self, mutation_sets, limit, expected_set):
      actual = epistasis_selection.filter_mutation_set_by_position(mutation_sets, limit)
      self.assertSetEqual(set(actual), set(expected_set))


class GetEpistaticSeqsIntegrationTest(parameterized.TestCase):

  def _basic_params(self):
    """Weight matrix and field vector."""
    rng = np.random.default_rng(0)
    weight_matrix = rng.normal(size=(4, 4, 20, 20))
    # make symmetric
    weight_matrix = weight_matrix + np.moveaxis(weight_matrix, (0, 1, 2, 3), (1, 0, 3, 2))
    field_vec = rng.normal(size=(4, 20))
    return weight_matrix, field_vec

  def _get_landscape(self):
    """Return a small PottsModel landscape."""
    weight_matrix, field_vec = self._basic_params()

    return potts_model.PottsModel(
        weight_matrix,
        field_vec,
        distance_threshold_for_nearby_residues=0,
        coupling_scale=1.0,
        field_scale=1.0,
        wt_seq=[0, 0, 0, 0])

  def test_get_epistatic_seqs(self):
    expected_len = 2
    epistatic_seqs = epistasis_selection.get_epistatic_seqs_for_landscape(self._get_landscape(), top_k=5, distance=2, n=2)
    self.assertLen(epistatic_seqs, expected_len)

  def test_get_adaptive_seqs(self):
    expected_len = 2
    adaptive_seqs = epistasis_selection.get_adaptive_seqs_for_landscape(self._get_landscape(), top_k=10, distance=2, n=2)
    self.assertLen(adaptive_seqs, expected_len)


if __name__ == '__main__':
  absltest.main()
