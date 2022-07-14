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

"""Tests for experiment."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import experiment
import utils


class ExperimentTest(parameterized.TestCase):
    """Tests for experiment."""
    test_set_dir= 'test_data/test_sets'
    mogwai_filepath = 'test_data/fakepdb_model_state_dict.npz'


    @parameterized.parameters(
        ('cnn'),
        ('linear'),
    )
    def test_run_regression_experiment_deterministic(self, model_name):
        regression_kwargs = dict(
            mogwai_filepath=self.mogwai_filepath,
            fraction_adaptive_singles=0.9,
            fraction_reciprocal_adaptive_epistasis=None,
            epistatic_horizon=None,
            normalize_to_singles=False,
            training_set_min_num_mutations=0,
            training_set_max_num_mutations=3,
            training_set_num_samples=100,
            training_set_include_singles=True,
            training_set_random_seed=0,
            test_set_dir=self.test_set_dir,
            model_name=model_name,
            model_random_seed=0,
            model_kwargs={'cnn_adam_learning_rate': 0.1, 'cnn_num_filters': 2},
            )
        run_one = experiment.run_regression_experiment(**regression_kwargs)  # type: ignore
        run_two = experiment.run_regression_experiment(**regression_kwargs)  # type: ignore
        # Test deterministic runs.
        self.assertEqual(run_one, run_two)

        # test that expected metrics are there.
        expected_split_keys = [
            'train',
            'test_set_1',
        ]
        expected_metric_keys = [
            'mse', 'std_test', 'std_predicted', 'test_size'
        ]
        for split_key in expected_split_keys:
            self.assertIn(split_key, run_one.keys())
            for metric_key in expected_metric_keys:
                self.assertIn(metric_key, run_one[split_key].keys())

    @parameterized.parameters(
        ('cnn',),
        ('linear',),
    )
    def test_run_design_experiment_deterministic(self, model_name):
        design_kwargs = dict(
            mogwai_filepath=self.mogwai_filepath,
            fraction_adaptive_singles=0.9,
            fraction_reciprocal_adaptive_epistasis=None,
            epistatic_horizon=None,
            normalize_to_singles=False,
            training_set_min_num_mutations=0,
            training_set_max_num_mutations=3,
            training_set_num_samples=100,
            training_set_include_singles=True,
            training_set_random_seed=0,
            model_name=model_name,
            model_random_seed=0,
            model_kwargs={'cnn_num_filters': 8, 'cnn_hidden_size': 32},
            mbo_random_seed=0,
            mbo_num_designs=1000,
            inner_loop_solver_top_k=50,
            inner_loop_solver_min_mutations=1,
            inner_loop_solver_max_mutations=3,
            inner_loop_num_rounds=1,
            inner_loop_num_samples=10,
            design_metrics_hit_threshold=0,
            design_metrics_cluster_hamming_distance=3,
            design_metrics_fitness_percentiles=[0.5, 0.9],
        )
        run_one = experiment.run_design_experiment(**design_kwargs)  # type: ignore
        run_two = experiment.run_design_experiment(**design_kwargs)  # type: ignore
        # Test deterministic runs.
        self.assertEqual(run_one, run_two)

        expected_keys = [
            'diversity_normalize_hit_rate',
            '0.5_percentile_fitness',
            '0.9_percentile_fitness'
        ]
        for key in expected_keys:
            self.assertIn(key, run_one.keys())


if __name__ == '__main__':
    absltest.main()
