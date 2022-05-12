"""Tests for epistasis_selection.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd


import epistasis_selection


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


if __name__ == '__main__':
  absltest.main()
