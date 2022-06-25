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

"""Solvers for proposing new sequences given a set of data."""

import abc
import itertools
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import sampling
import utils


class Solver(abc.ABC):
    """Solver base class."""

    @abc.abstractmethod
    def propose(self, data, num_samples,
                random_state):
        """Proposes `num_samples` sequences given input data.

        Args:
          data: A pd.DataFrame with columns 'sequence' and 'fitness'.
          num_samples: The number of proposals to return.
          random_state: An optional instance of np.random.RandomState.

        Returns:
          A 2D np.ndarray with integer encoded sequences with dimension
          [`num_samples`, L].
        """


def _optimize_inner_loop(solver, initial_df, score_fn,
                         num_rounds, num_samples,
                         random_state):
    """Perform `num_rounds` of inner loop optimization and return all candidates.

    Args:
      solver: A Solver.
      initial_df: A DataFrame with `sequence` and `fitness` keys.
      score_fn: A function that takes in a 2D array NxL and returns a 1D array of
        scalars of size N.
      num_rounds: The number of inner loop rounds to complete.
      num_samples: The number of samples to take in each inner loop iteration.
      random_state: A np.random.RandomState.

    Returns:
      A pd.DataFrame with all candidates seen across all rounds, including the
      initial_df. The resulting DataFrame will be of size
      (num_samples * num_rounds) + initial_df.shape[0]
    """
    all_candidates_df = initial_df.copy()
    all_candidates_df['fitness'] = score_fn(
        np.vstack(all_candidates_df.sequence.values))

    for _ in range(num_rounds):
        inner_loop_candidates = solver.propose(all_candidates_df, num_samples,
                                               random_state)
        y_pred = score_fn(inner_loop_candidates)
        candidate_df = pd.DataFrame(
            dict(sequence=list(inner_loop_candidates), fitness=y_pred))
        all_candidates_df = pd.concat([all_candidates_df, candidate_df])

    return all_candidates_df


class RecombinationSolver(Solver):
    """Solver which recombines `top_k` sequences with the highest fitness.

    For each proposal, we sample a pair of sequences A, B from the top K, then
    sample a crossover index and return the result of crossing over A and B.
    """

    def __init__(self, top_k):
        self._top_k = top_k

    def propose(
            self,
            data,
            num_samples,
            random_state=None):
        if not random_state:
            random_state = np.random.RandomState()
        if num_samples <= 0:
            raise AssertionError('num_samples must be >0')
        pool = data.sort_values(
            by='fitness', ascending=False).head(self._top_k).sequence.values

        if len(pool) < 2:
            return np.vstack(pool)

        proposals = []
        for _ in range(num_samples):
            pair = random_state.choice(pool, size=2)
            proposal = utils.recombine_seqs(pair[0], pair[1], random_state)
            proposals.append(proposal)
        return np.vstack(proposals)


class ModelBasedSolver(Solver):
    """Solver which uses a model to guide proposals.

    A ModelBasedSolver uses a set of input observations to train a model, and then
    explores sequence space in the "inner loop" to find candidate sequences
    that are scored highly by the model.
    """

    def __init__(self, model, vocab_size, flatten_inputs,
                 inner_loop_solver, inner_loop_num_rounds,
                 inner_loop_num_samples):
        """Construct a ModelBasedSolver.

        Args:
          model: A model with methods .fit(x, y) and .predict(x), where .predict(x)
            returns a float. The model is used to guide the inner loop search, so if
            .predict(x) returns an integer, this will lead to a flat (and difficult
            to optimize) landscape.
          vocab_size: The number of amino acids in the vocabulary.
          flatten_inputs: Boolean that is True if `model` takes a 1D sequence of
            one-hots (e.g. a linear model) or False if `model` takes a 2D array of
            one-hots (e.g. a CNN).
          inner_loop_solver: A Solver class with a .propose() method. Generates
            candidates to evaluate with the model.
          inner_loop_num_rounds: Number of inner loop rounds to evaluate.
          inner_loop_num_samples: Number of candidate to generate in each round of
            the inner loop.
        """
        self._model = model
        self._vocab_size = vocab_size
        self._flatten_inputs = flatten_inputs
        self._inner_loop_solver = inner_loop_solver
        self._inner_loop_num_rounds = inner_loop_num_rounds
        self._inner_loop_num_samples = inner_loop_num_samples

    def propose(
            self,
            data,
            num_samples,
            random_state=None):

        if not random_state:
            random_state = np.random.RandomState()
        if num_samples <= 0:
            raise AssertionError('num_samples must be >0')

        # Fit model on data.
        x_train, y_train = utils.get_x_y_from_df(
            data, vocab_size=self._vocab_size, flatten=self._flatten_inputs)
        self._model.fit(x_train, y_train)

        score_fn = self._get_model_predictions
        all_candidates_df = _optimize_inner_loop(self._inner_loop_solver, data,
                                                 score_fn,
                                                 self._inner_loop_num_rounds,
                                                 self._inner_loop_num_samples,
                                                 random_state)
        # Grab the top `num_samples` candidates ever seen.
        all_candidates_df['sequence'] = all_candidates_df['sequence'].apply(
            tuple)
        proposals = all_candidates_df.drop_duplicates('sequence').sort_values(
            by='fitness', ascending=False).head(num_samples).sequence.values
        return np.vstack(proposals)

    def _get_model_predictions(self, x):
        if self._flatten_inputs:
            y_pred = self._model.predict(
                utils.one_hot_and_flatten(x, self._vocab_size))
        else:
            y_pred = self._model.predict(
                utils.onehot(x, num_classes=self._vocab_size))
        return y_pred


class RandomMutationSolver(Solver):
    """Solver that randomly samples around best sequences.

    The optimization algorithm is as follows: A distance D is sampled uniformly
    between `min_distance` and `max_distance`. A sequence S is sampled uniformly
    from the `top_k` highest fitness sequences. D mutations are then sampled
    uniformly across the sequence and applied to S.
    """

    def __init__(self,
                 min_distance,
                 max_distance,
                 top_k,
                 vocab_size=20):
        """Constructs a RandomHopper Solver.

        Args:
          min_distance: The minimum Hamming distance (inclusive) to consider from a
            known sequence.
          max_distance: The maximum Hamming distance (inclusive) to consider from a
            known sequence.
          top_k: The number of top sequences to explore around.
          vocab_size: The size of the amino acid vocabulary.
        """
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._top_k = top_k
        self._vocab_size = vocab_size

    def propose(
            self,
            data,
            num_samples,
            random_state=None):

        if not random_state:
            random_state = np.random.RandomState()
        if num_samples <= 0:
            raise ValueError('num_samples must be > 0')

        proposals = []
        data['sequence'] = data.sequence.apply(tuple)
        top_k_df = data.drop_duplicates('sequence').sort_values(
            by='fitness', ascending=False).head(self._top_k)
        starting_points = top_k_df.sample(
            n=num_samples, replace=True, random_state=random_state).sequence.values
        distances = random_state.choice(
            range(self._min_distance, self._max_distance + 1), size=num_samples)

        for distance, sequence in zip(distances, starting_points):
            # sample a modification to the sequence
            proposal = sampling.sample_within_hamming_radius(
                sequence,
                num_samples=1,
                vocab_size=self._vocab_size,
                min_mutations=distance,
                max_mutations=distance,
                random_state=random_state)
            proposals.append(proposal)

        return np.vstack(proposals)


class MutationStackerSolver(Solver):
    """Solver which combines `top_k` sequences with the highest fitness.

    Given a set of sequences, the Mutation Stacker
      (1) ranks by fitness,
      (2) selects the top K unique sequences,
      (3) combines all (K choose 2) pairs to propose the new set.

    If there are more combinations than requested, a uniform subsample is taken
    from the combinations. If there are fewer combinations than requested, the
    maximum number of possible combinations (possibly fewer than `num_samples`)
    is returned.

    To combine a pair of sequences A and B, Mutation Stacker "stacks" their
    mutations relative to `reference_seq` so that the resulting variant includes
    all the mutations from A *and* B. See the docstring for `utils.add_seqs` for
    more details.
    """

    def __init__(self, top_k, reference_seq):
        self._top_k = top_k
        self._reference_seq = np.array(reference_seq)

    def propose(
            self,
            data,
            num_samples,
            random_state=None):
        if not random_state:
            random_state = np.random.RandomState()
        if num_samples <= 0:
            raise AssertionError('num_samples must be >0')
        pool = data.copy()
        # Create a hashable type for deduplication.
        pool['sequence_tuple'] = pool.sequence.apply(tuple)
        pool = pool.sort_values(by='fitness', ascending=False)
        pool = pool.drop_duplicates('sequence_tuple')
        pool = pool.head(self._top_k).sequence.values

        if len(pool) < 2:
            return np.vstack(pool * num_samples)

        proposals = []
        for pair in itertools.combinations(pool, 2):
            proposal = utils.add_seqs(pair[0], pair[1], self._reference_seq)
            # Create a hashable type for deduplication.
            proposal_tuples = [tuple(p) for p in proposal]
            proposals.extend(proposal_tuples)

        # Deduplicate proposals.
        proposals = list(set(proposals))
        random_state.shuffle(proposals)
        return np.vstack(proposals[:num_samples])
