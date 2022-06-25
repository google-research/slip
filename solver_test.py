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

"""Tests for synthetic_protein_landscapes.solver."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm

import solver
import utils


class RecombinationSolverTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='basic',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            num_samples=2,
            top_k=2,
            seed=0,
            expected_set=[[1, 0, 0], [0, 0, 0]],
        ),
        dict(
            testcase_name='top2',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            num_samples=10,
            top_k=2,
            seed=0,
            expected_set=[[1, 0, 0], [0, 0, 0]],
        ),
        dict(
            testcase_name='more combos',
            sequences=[[0, 0, 1], [0, 1, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            num_samples=10,
            top_k=2,
            seed=0,
            expected_set=[[0, 1, 0], [0, 0, 1], [0, 1, 1]],
        ),
    )
    def test_propose(self, sequences, fitnesses, num_samples, top_k, seed,
                     expected_set):

        def get_proposals():
            random_state = np.random.RandomState(seed)
            data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
            comb_solver = solver.RecombinationSolver(top_k)
            proposals = comb_solver.propose(
                data, num_samples, random_state=random_state)
            return proposals

        proposals = get_proposals()
        for proposal in proposals:
            self.assertIn(list(proposal), expected_set)

        # test deterministic.
        np.testing.assert_array_equal(proposals, get_proposals())

    @parameterized.named_parameters(
        dict(
            testcase_name='wrong lengths',
            sequences=[[0, 0, 0], [4, 4, 4, 4]],
            fitnesses=[1, 1],
            num_samples=2,
            top_k=2,
            seed=0,
        ),)
    def test_wrong_lengths(self, sequences, fitnesses, num_samples, top_k, seed):
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        comb_solver = solver.RecombinationSolver(top_k)
        with self.assertRaisesRegex(AssertionError, 'Input lengths'):
            comb_solver.propose(data, num_samples, random_state=random_state)

    @parameterized.named_parameters(
        dict(
            testcase_name='no samples',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            num_samples=0,
            top_k=2,
            seed=0,
        ),
        dict(
            testcase_name='negative samples',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            num_samples=-100,
            top_k=2,
            seed=0,
        ),
    )
    def test_wrong_num_samples(self, sequences, fitnesses, num_samples, top_k,
                               seed):
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        comb_solver = solver.RecombinationSolver(top_k)
        with self.assertRaisesRegex(AssertionError, 'num_samples'):
            comb_solver.propose(data, num_samples, random_state=random_state)


class RandomMutationSolverTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='basic',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            num_samples=2,
            top_k=2,
            seed=0,
        ),
        dict(
            testcase_name='top2',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            num_samples=10,
            top_k=2,
            seed=0),
    )
    def test_deterministic_propose(self, sequences, fitnesses, num_samples, top_k,
                                   seed):

        def get_proposals():
            random_state = np.random.RandomState(seed)
            data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
            random_mutation_solver = solver.RandomMutationSolver(
                1, 1, top_k=top_k)
            proposals = random_mutation_solver.propose(
                data, num_samples, random_state=random_state)
            return proposals

        np.testing.assert_array_equal(get_proposals(), get_proposals())

    @parameterized.named_parameters(
        dict(
            testcase_name='1 step',
            starting_sequence=[0, 0, 0],
            min_distance=1,
            max_distance=1,
            num_samples=10,
            seed=0,
        ),
        dict(
            testcase_name='2 steps',
            starting_sequence=[0, 0, 0],
            min_distance=1,
            max_distance=2,
            num_samples=10,
            seed=0,
        ),
    )
    def test_single_step_distances(self, starting_sequence, min_distance,
                                   max_distance, num_samples, seed):
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=[starting_sequence], fitness=[1]))
        random_mutation_solver = solver.RandomMutationSolver(
            min_distance, max_distance, top_k=1)
        proposals = random_mutation_solver.propose(
            data, num_samples, random_state=random_state)

        for proposal in proposals:
            distance = utils.hamming_distance(proposal, starting_sequence)
            self.assertLessEqual(distance, max_distance)
            self.assertGreaterEqual(distance, min_distance)

    @parameterized.named_parameters(
        dict(
            testcase_name='top 2',
            sequences=[[0, 0], [1, 1], [2, 2]],
            fitnesses=[2, 1, 0],
            top_k=2,
            min_distance=0,
            max_distance=0,
            num_samples=10,
            seed=0,
            expected_set=[[0, 0], [1, 1]],
        ),)
    def test_top_k(self, sequences, fitnesses, min_distance, max_distance, top_k,
                   num_samples, seed, expected_set):
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        random_mutation_solver = solver.RandomMutationSolver(
            min_distance, max_distance, top_k=top_k)
        proposals = random_mutation_solver.propose(
            data, num_samples, random_state=random_state)

        for proposal in proposals:
            self.assertIn(list(proposal), expected_set)

    @parameterized.named_parameters(
        dict(
            testcase_name='all 2 step variants',
            sequences=[[0, 0]],
            fitnesses=[0],
            top_k=2,
            min_distance=0,
            max_distance=2,
            vocab_size=2,
            num_samples=20,
            seed=0,
            expected_set=[[0, 0], [1, 1], [1, 0], [0, 1]],
        ),)
    def test_propose(self, sequences, fitnesses, min_distance, max_distance,
                     vocab_size, top_k, num_samples, seed, expected_set):
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        random_mutation_solver = solver.RandomMutationSolver(
            min_distance, max_distance, top_k=top_k, vocab_size=vocab_size)
        proposals = random_mutation_solver.propose(
            data, num_samples, random_state=random_state)

        for proposal in proposals:
            self.assertIn(list(proposal), expected_set)

    @parameterized.named_parameters(
        dict(
            testcase_name='find 10, 10, 10',
            sequences=[[0, 0, 0], [1, 1, 0]],
            fitnesses=[0, 2],
            vocab_size=10,
            num_rounds=10,
            num_samples=5000,
            best_score=30,
            seed=1,
        ),
        dict(
            testcase_name='find 3, 3, 3',
            sequences=[[0, 0, 0], [1, 1, 0]],
            fitnesses=[0, 2],
            vocab_size=3,
            num_rounds=10,
            num_samples=1000,
            best_score=9,
            seed=1,
        ))
    def test_optimize_inner_loop(self, sequences, fitnesses, vocab_size,
                                 num_rounds, num_samples, best_score, seed):
        random_state = np.random.RandomState(seed)
        def score_fn(seqs): return np.sum(seqs, axis=1)
        random_mutation_solver = solver.RandomMutationSolver(
            min_distance=1, max_distance=1, top_k=1, vocab_size=vocab_size)
        df = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))

        candidates_df = solver._optimize_inner_loop(random_mutation_solver, df,
                                                    score_fn, num_rounds,
                                                    num_samples, random_state)
        best_score_tolerance = 3
        self.assertGreaterEqual(candidates_df.fitness.max(),
                                best_score - best_score_tolerance)


class ModelBasedSolverTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='2 samples linear',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            model=linear_model.LinearRegression(),
            num_samples=2,
            seed=0,
        ),
        dict(
            testcase_name='10 samples linear',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            model=linear_model.LinearRegression(),
            num_samples=10,
            seed=0),
        dict(
            testcase_name='10 samples SVR',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            model=svm.SVR(C=1.0, epsilon=0.2),
            num_samples=10,
            seed=0),
        dict(
            testcase_name='10 samples Random Forest',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            model=ensemble.RandomForestRegressor(max_depth=2, random_state=0),
            num_samples=10,
            seed=0),
    )
    def test_deterministic_propose(self, sequences, fitnesses, model, num_samples,
                                   seed):
        vocab_size = 3
        inner_loop_solver = solver.RandomMutationSolver(
            1, 1, top_k=3, vocab_size=vocab_size)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))

        def get_proposals():
            random_state = np.random.RandomState(seed)
            mbo_solver = solver.ModelBasedSolver(
                model,
                vocab_size=vocab_size,
                flatten_inputs=True,
                inner_loop_num_rounds=1,
                inner_loop_num_samples=10,
                inner_loop_solver=inner_loop_solver)
            proposals = mbo_solver.propose(
                data, num_samples, random_state=random_state)
            return proposals

        np.testing.assert_array_equal(get_proposals(), get_proposals())


class MutationStackerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name='basic',
            sequences=[[0, 0, 0], [1, 0, 0]],
            fitnesses=[1, 1],
            num_samples=2,
            top_k=2,
            seed=0,
            expected_set=[[1, 0, 0], [0, 0, 0]],
        ),
        dict(
            testcase_name='top2',
            sequences=[[0, 0, 0], [1, 0, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            num_samples=10,
            top_k=2,
            seed=0,
            expected_set=[[1, 0, 0], [0, 0, 0]],
        ),
        dict(
            testcase_name='more combos',
            sequences=[[0, 0, 1], [0, 1, 0], [2, 2, 2]],
            fitnesses=[1, 1, 0],
            num_samples=10,
            top_k=2,
            seed=0,
            expected_set=[[0, 1, 0], [0, 0, 1], [0, 1, 1]],
        ),
    )
    def test_propose(self, sequences, fitnesses, num_samples, top_k, seed,
                     expected_set):

        def get_proposals():
            parent_seq = [0, 0, 0]
            random_state = np.random.RandomState(seed)
            data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
            mutation_stacker_solver = solver.MutationStackerSolver(
                top_k, parent_seq)
            proposals = mutation_stacker_solver.propose(
                data, num_samples, random_state=random_state)
            return proposals

        proposals = get_proposals()
        for proposal in proposals:
            self.assertIn(list(proposal), expected_set)

        # test deterministic.
        np.testing.assert_array_equal(proposals, get_proposals())

    @parameterized.named_parameters(
        dict(
            testcase_name='wrong lengths',
            sequences=[[0, 0, 0], [4, 4, 4, 4]],
            fitnesses=[1, 1],
            num_samples=2,
            top_k=2,
            seed=0,
        ),)
    def test_wrong_lengths(self, sequences, fitnesses, num_samples, top_k, seed):
        parent_seq = [0, 0, 0]
        random_state = np.random.RandomState(seed)
        data = pd.DataFrame(dict(sequence=sequences, fitness=fitnesses))
        mutation_stacker_solver = solver.MutationStackerSolver(
            top_k, parent_seq)
        with self.assertRaisesRegex(AssertionError, 'equal length'):
            mutation_stacker_solver.propose(
                data, num_samples, random_state=random_state)


if __name__ == '__main__':
    absltest.main()
