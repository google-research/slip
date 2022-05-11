import itertools
from typing import Optional, Iterable, Tuple, List

import numpy as np

import potts_model
import utils


def combine_k_rounds(num_rounds: int, mutations: Iterable[Iterable[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
  """Return the result of combining `mutations` for `num_rounds`.

  Starting with a pool of M `mutations` m_1 ... m_M, stack them for K=`num_rounds` rounds. For example,
  for K=3 rounds of combination, this will result in every variant (m_i + m_j + m_k), for i, j, k \in M.
  Be careful of memory usage, as this can be very large due to combinatorial possibilities.
  In the best case, this scales with {M \choose K}. But if mutations overlap at P positions,
  combining them produces 2^{P} variants. So in the worst case, this will produce
  {M \choose K} * 2^{P} variants. See the definition for `utils.merge_mutation_sets` for more on
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
    return mutations
  mutations_to_combine = list(itertools.combinations(mutations, num_rounds + 1))

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
    all_samples.extend(next_round)
  return all_samples


def get_test_seqs_for_landscape(landscape: potts_model.PottsModel,
                                distance: int,
                                n: int,
                                adaptive: bool = True,
                                top_k: Optional[int] = None,
                                rng: np.random.Generator = np.random.default_rng(0)):
  """Return `n` variants at `distance` that are enriched for epistasis on `landscape`.

  Args:
    landscape: The landscape.
    distance: The number of mutations from the landscape wildtype. Raises a ValueError if not an even number.
    n: The number of variants in the test set.
    adaptive: When True (False), return sequences enriched for adaptive (deleterious) epistasis
    top_k: The number of highest magnitude interactions to use for sampling.
    rng: Random state.

  Return:
    Sequences.
  """
  if distance % 2 != 0:
    raise ValueError('Odd distance not supported.')

  # TODO(nthomas) another option is to do this combination in batches, until the test set is full or the
  # input mutations are exhausted
  if not top_k:
    top_k = n
  tensor_indexes = utils.get_top_n_4d_tensor_indexes(landscape.epistasis_tensor, top_k, lowest=not adaptive)
  mutation_pairs = [utils.get_mutation_pair_from_tensor_index(
      t) for t in tensor_indexes]


  num_rounds = distance // 2
  all_combined = combine_k_rounds(num_rounds, mutation_pairs)
  all_combined = _filter_elements_to_length(all_combined, distance)
  if len(all_combined) < n:
    raise ValueError(f'Not enough ({len(all_combined)} < {n}) mutants at distance {distance}, try increasing `top_k`.')
  subset = rng.choice(all_combined, n, replace=False)
  test_seqs = [utils.apply_mutations(landscape.wildtype_sequence, m) for m in subset]
  return test_seqs


def _filter_elements_to_length(elements: Iterable[Iterable], length: int) -> List[Iterable]:
  lengths = [len(x) for x in elements]
  to_include = [l == length for l in lengths]
  filtered = []
  for i, include in enumerate(to_include):
    if include is True:
      filtered.append(elements[i])
  return filtered
