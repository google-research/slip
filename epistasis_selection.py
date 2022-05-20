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

from collections import Counter
import functools
import itertools
from typing import Optional, Iterable, Tuple, List


import numpy as np

import potts_model
import sampling
import utils


def combine_k_rounds(num_rounds: int, mutations: Iterable[Tuple[Tuple[int, int], ...]]) -> List[Tuple[Tuple[int, int], ...]]:
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
  mutation_combinations = itertools.combinations(mutations, num_rounds + 1)

  all_samples = []
  for mutation_combination in mutation_combinations:
    all_samples.extend(utils.merge_multiple_mutation_sets(mutation_combination))
  return all_samples


def filter_mutation_set_by_position(mutation_sets: Iterable[Tuple[Tuple[int, int], ...]], limit: int = 10):
  """Return a filtered mutation set, where each position is used a maximum of `limit` times."""
  filtered_mutation_sets = []
  position_counter = Counter()
  for mutation_set in mutation_sets:
    positions = [m[0] for m in mutation_set]
    if any([position_counter[position] >= limit for position in positions]):
      continue
    else:
      position_counter.update(positions)
      filtered_mutation_sets.append(mutation_set)
  return filtered_mutation_sets


def get_epistatic_seqs_for_landscape(landscape: potts_model.PottsModel,
                                     distance: int,
                                     n: int,
                                     adaptive: bool = True,
                                     max_reuse: Optional[int] = None,
                                     top_k: Optional[int] = None,
                                     random_state: np.random.RandomState = np.random.RandomState(0)
                                     ) -> List[np.ndarray]:
  """Return `n` variants at `distance` that are enriched for epistasis on `landscape`.

  To construct epistatic sequences, the top epistatic pairs are taken directly from the landscape
  epistasis tensor, and used as building blocks for higher order mutants. If `max_reuse` is set, the
  top epistatic pairs are filtered greedily to only reuse the same positions `max_reuse` times.

  Args:
    landscape: The landscape.
    distance: The number of mutations from the landscape wildtype. Raises a ValueError if not an even number.
    n: The number of variants in the test set.
    adaptive: When True (False), return sequences enriched for adaptive (deleterious) epistasis
    max_reuse: An integer indicating the maximum number of times a position can be reused in the starting pool
      of epistatic pairs.
    top_k: The number of highest magnitude interactions to use for sampling. All epistatic pairs included in the
     resulting variants are guaranteed to be within the `top_k` highest magnitude.
    random_state: An instance of np.random.RandomState

  Return:
    A List of sequences.
  """
  if distance % 2 != 0:
    raise ValueError('Odd distance not supported.')

  if not top_k:
    top_k = n
  mutation_pairs = utils.get_top_n_mutation_pairs(landscape.epistasis_tensor, top_k, lowest=not adaptive)
  if max_reuse is not None:
    assert max_reuse > 0
    mutation_pairs = filter_mutation_set_by_position(mutation_pairs, limit=max_reuse)
    print(f'{len(mutation_pairs)} after filtering {top_k}')

  num_rounds = distance // 2
  all_combined = combine_k_rounds(num_rounds, mutation_pairs)
  all_combined = [element for element in all_combined if len(element) == distance]

  if len(all_combined) < n:
    raise ValueError(f'Not enough ({len(all_combined)} < {n}) mutants at distance {distance}, try increasing `top_k`.')
  # TODO(nthomas) after switching to np.random.Generator, we can do rng.choice(all_combined)
  subset_idxs = random_state.choice(len(all_combined), n, replace=False)
  subset = [all_combined[i] for i in subset_idxs]

  seqs = [utils.apply_mutations(landscape.wildtype_sequence, m) for m in subset]
  return seqs


def get_adaptive_seqs_for_landscape(landscape: potts_model.PottsModel,
                                    distance: int,
                                    n: Optional[int] = None,
                                    adaptive: bool = True,
                                    max_reuse: Optional[int] = None,
                                    top_k: Optional[int] = None,
                                    random_state: np.random.RandomState = np.random.RandomState(0)
                                    ) -> List[np.ndarray]:
  """Return `n` variants at `distance` that are enriched for adaptive singles on `landscape`.

  To construct adaptive sequences, the top single mutants are taken directly from the landscape,
  and used as building blocks for higher order mutants. If `max_reuse` is set, the
  top singles are filtered greedily to only reuse the same positions `max_reuse` times.

  Args:
    landscape: The landscape.
    distance: The number of mutations from the landscape wildtype. Raises a ValueError if not an even number.
    n: The number of variants.
    adaptive: When True (False), return sequences enriched for adaptive (deleterious) singles.
    max_reuse: An integer indicating the maximum number of times a position can be reused in the starting pool
      of epistatic pairs.
    top_k: The number of singles to use as building blocks.
    random_state: An instance of np.random.RandomState.

  Return:
    A List of sequences.
  """
  all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
  fitnesses = landscape.evaluate(all_singles)

  if not top_k:
    top_k = n
  if adaptive:
    top_k_indexes = np.argsort(-1 * fitnesses)[:top_k]
  else:
    top_k_indexes = np.argsort(fitnesses)[:top_k]

  get_mutations_from_wt = functools.partial(utils.get_mutations, parent=landscape.wildtype_sequence)
  mutation_set = [get_mutations_from_wt(single) for single in all_singles[top_k_indexes]]

  if max_reuse is not None:
    assert max_reuse > 0
    mutation_set = filter_mutation_set_by_position(mutation_set, limit=max_reuse)
    print(f'{len(mutation_set)} after filtering {top_k}')

  num_rounds = distance
  all_combined = combine_k_rounds(num_rounds, mutation_set)
  all_combined = [element for element in all_combined if len(element) == distance]

  if n is not None:
    if len(all_combined) < n:
      raise ValueError(f'Not enough ({len(all_combined)} < {n}) mutants at distance {distance}, try increasing `top_k`.')
    # TODO(nthomas) after switching to np.random.Generator, we can do rng.choice(all_combined)
    subset_idxs = random_state.choice(len(all_combined), n, replace=False)
    subset = [all_combined[i] for i in subset_idxs]
    seqs = [utils.apply_mutations(landscape.wildtype_sequence, m) for m in subset]
  else:
    seqs = [utils.apply_mutations(landscape.wildtype_sequence, m) for m in all_combined]
  return seqs
