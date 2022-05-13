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

import itertools
from typing import Optional, Iterable, Tuple, List, Sequence

import numpy as np

import potts_model
import utils


def combine_k_rounds(num_rounds: int, mutations: Iterable[Sequence[Tuple[int, int]]]) -> List[Sequence[Tuple[int, int]]]:
  """Return the result of combining `mutations` for `num_rounds`.

  Starting with a pool of M `mutations` m_1 ... m_M, stack them for K=`num_rounds` rounds. For example,
  for K=3 rounds of combination, this will result in every variant (m_i + m_j + m_k), for i, j, k \\in M.
  Be careful of memory usage, as this can be very large due to combinatorial possibilities.
  In the best case, this scales with {M \\choose K}. But if mutations overlap at P positions,
  combining them produces 1 + 2^{P} variants. So in the worst case, this will produce
  {M \\choose K} * 2^{P} variants. See the definition for `utils.merge_mutation_sets` for more on
  mutation merging.

  Args:
    num_rounds: The number of rounds of combination
    mutations: The starting pool of mutations, where each mutation is an iterable of
      tuples encoding mutations (position, mutation).

  Returns:
    A list of tuples of mutations, where each element will be a combination of
    `num_rounds` mutations from `mutations`. Note that each tuple will possibly be of different lengths.

  """
  if num_rounds == 0:
    return list(mutations)
  mutations_to_combine = itertools.combinations(mutations, num_rounds + 1)

  all_samples = []

  for mutant in mutations_to_combine:
    mutant = list(mutant)
    prev_round = [()]
    # iteratively add the mutations together
    for i in range(num_rounds + 1):
      next_round = []
      new_mutation = mutant.pop()
      for merged in prev_round:
        next_round.extend(utils.merge_mutation_sets(merged, new_mutation))
      prev_round = next_round
    all_samples.extend(prev_round)
  return all_samples


def get_epistatic_seqs_for_landscape(landscape: potts_model.PottsModel,
                                distance: int,
                                n: int,
                                adaptive: bool = True,
                                top_k: Optional[int] = None,
                                random_state: np.random.RandomState = np.random.RandomState(0)):
  """Return `n` variants at `distance` that are enriched for epistasis on `landscape`.

  To construct epistatic sequences, the top epistatic pairs are taken directly from the landscape
  epistasis tensor, and used as building blocks for higher order mutants.

  Args:
    landscape: The landscape.
    distance: The number of mutations from the landscape wildtype. Raises a ValueError if not an even number.
    n: The number of variants in the test set.
    adaptive: When True (False), return sequences enriched for adaptive (deleterious) epistasis
    top_k: The number of highest magnitude interactions to use for sampling.
    random_state: An instance of np.random.RandomState

  Return:
    A List of sequences.
  """
  if distance % 2 != 0:
    raise ValueError('Odd distance not supported.')

  # TODO(nthomas) another option is to do this combination in batches, until the test set is full or the
  # input mutations are exhausted
  if not top_k:
    top_k = n
  mutation_pairs = utils.get_top_n_mutation_pairs(landscape.epistasis_tensor, top_k, lowest=not adaptive)

  num_rounds = distance // 2
  all_combined = combine_k_rounds(num_rounds, mutation_pairs)
  all_combined = [element for element in all_combined if len(element) == distance]

  if len(all_combined) < n:
    raise ValueError(f'Not enough ({len(all_combined)} < {n}) mutants at distance {distance}, try increasing `top_k`.')
  subset = random_state.choice(all_combined, n, replace=False)
  seqs = [utils.apply_mutations(landscape.wildtype_sequence, m) for m in subset]
  return seqs
