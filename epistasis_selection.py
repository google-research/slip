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
"""Selecting curated sets of sequences enriched for epistasis."""

from collections import Counter
import itertools
from typing import Iterable, Tuple, List

import numpy as np

import potts_model
import utils

Mutation = utils.Mutation


def combine_k_rounds(num_rounds: int,
                     mutations: Iterable[Tuple[Mutation, ...]]) -> List[Tuple[Mutation, ...]]:
    """Return the result of combining `mutations` for `num_rounds`.

    Starting with a pool of M `mutations` m_1 ... m_M, stack them for K=`num_rounds` rounds. For example,
    for K=3 rounds of combination, this will result in every variant (m_i + m_j + m_k), for i, j, k in M.
    Be careful of memory usage, as this can be very large due to combinatorial possibilities.
    In the best case, this scales with {M choose K}. But if mutations overlap at P positions,
    combining them produces 1 + 2^{P} variants. So in the worst case, this will produce
    {M choose K} * 2^{P} variants. See the definition for `utils.merge_mutation_sets` for more on
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
        all_samples.extend(
            utils.merge_multiple_mutation_sets(mutation_combination))
    return all_samples


def filter_mutation_sets_by_position(mutation_sets: Iterable[Tuple[Mutation, ...]],
                                     limit: int) -> List[Tuple[Mutation, ...]]:
    """Return a filtered list of mutation sets, where each position is used a maximum of `limit` times."""
    if limit <= 0:
        raise ValueError('Limit must be > 0.')

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


def get_top_k_epistatic_pairs(landscape: potts_model.PottsModel,
                              adaptive: bool,
                              max_reuse: int,
                              top_k: int) -> List[Tuple[Mutation, Mutation]]:
    """Returns a list of high magnitude mutation pairs.

    Args:
      landscape: The landscape.
      adaptive: When True (False), return sequences enriched for adaptive (deleterious) epistasis
      max_reuse: An integer indicating the maximum number of times a position can be reused in the starting pool
        of epistatic pairs.
      top_k: The number of highest magnitude interactions to use for sampling. All epistatic pairs included in the
       resulting variants are guaranteed to be within the `top_k` highest magnitude.

    Returns:
        A list of mutation pairs.
"""
    if max_reuse <= 0:
        raise ValueError('`max_reuse` must be > 0.')
    mutation_pairs = utils.get_top_n_mutation_pairs(landscape.epistasis_tensor, top_n=top_k, get_highest=adaptive)
    mutation_pairs = filter_mutation_sets_by_position(mutation_pairs, limit=max_reuse)
    print(f'{len(mutation_pairs)} pairs after filtering {top_k}')
    return mutation_pairs


def get_top_k_single_mutations(landscape: potts_model.PottsModel, adaptive: bool, max_reuse: int, top_k: int) \
        -> List[Tuple[Mutation]]:
    """Returns a set of high magnitude single mutations.

    Args:
      landscape: The landscape.
      adaptive: When True (False), return mutations enriched for positive (negative) single mutant effects.
      max_reuse: A positive integer indicating the maximum number of times a position can be reused in the
        returned set of sequences.
      top_k: The number of highest magnitude effects to use for sampling. All mutations included in the
       returned set are guaranteed to be within the `top_k` highest magnitude.
    """
    if max_reuse <= 0:
        raise ValueError('`max_reuse` must be > 0.')

    mutation_sets = utils.get_top_n_single_mutations(landscape.wildtype_sequence,
                                                     landscape.vocab_size,
                                                     landscape.evaluate,
                                                     top_n=top_k,
                                                     get_highest=adaptive)
    mutation_sets = filter_mutation_sets_by_position(
        mutation_sets, limit=max_reuse)
    print(f'{len(mutation_sets)} singles after filtering {top_k}')
    return mutation_sets


def combine_mutations_and_subset(mutation_sets: Iterable[Tuple[Mutation, ...]],
                                 num_rounds: int,
                                 n: int,
                                 target_distance: int,
                                 wildtype_sequence,
                                 random_state: np.random.RandomState) -> List[np.ndarray]:
    """Returns a list of sequences constructed from `mutation_sets`.

    Args:
      mutation_sets: The pool of constituent mutations.
      num_rounds: The number of rounds to combine constituent mutations.
      n: The desired number of variants in the test set.
      target_distance: The desired distance of constructed variants from the wildtype.
      wildtype_sequence: An integer encoded starting sequence to interpret mutations.
      random_state: An instance of np.random.RandomState

    Return:
      A List of sequences of distance `target_distance` from the `wildtype_sequence`.
    """
    all_combined = combine_k_rounds(num_rounds, mutation_sets)
    all_combined = [element for element in all_combined if len(
        element) == target_distance]

    if len(all_combined) < n:
        raise ValueError(
            f'Not enough ({len(all_combined)} < {n}) mutants at target_distance {target_distance}, \
                 try increasing `top_k`.')
    subset_idxs = random_state.choice(len(all_combined), n, replace=False)
    subset = [all_combined[i] for i in subset_idxs]
    seqs = [utils.apply_mutations(wildtype_sequence, m) for m in subset]
    return seqs


def get_epistatic_seqs_for_landscape(landscape: potts_model.PottsModel,
                                     distance: int,
                                     n: int,
                                     adaptive: bool,
                                     max_reuse: int,
                                     top_k: int,
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

    mutation_pairs = get_top_k_epistatic_pairs(
        landscape, adaptive, max_reuse, top_k)
    num_rounds = distance // 2
    return combine_mutations_and_subset(mutation_pairs,
                                        num_rounds,
                                        n,
                                        distance,
                                        landscape.wildtype_sequence,
                                        random_state)


def get_adaptive_seqs_for_landscape(landscape: potts_model.PottsModel,
                                    distance: int,
                                    n: int,
                                    adaptive: bool,
                                    max_reuse: int,
                                    top_k: int,
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
    mutation_sets = get_top_k_single_mutations(landscape, adaptive, max_reuse, top_k)
    num_rounds = distance
    seqs = combine_mutations_and_subset(mutation_sets,
                                        num_rounds,
                                        n,
                                        distance,
                                        landscape.wildtype_sequence,
                                        random_state)
    return seqs
