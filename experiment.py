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

"""Methods for running optimization trajectories."""

import random as python_random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm
import tensorflow as tf

import metrics
import models
import potts_model
import sampling
import solver
import utils


def get_fitness_df(sequences, fitness_fn,
                   ref_seq):
    """Get a DataFrame with the fitness of the requested sequences.

    Args:
      sequences: A 2D NxL numpy array of integer encoded sequences.
      fitness_fn: A function, that when given a single integer encoded sequence,
        returns a fitness value.
      ref_seq: An integer encoded sequence. `num_mutations` is measured with
        respect to this sequence.

    Returns:
      A pd.DataFrame with the fields `sequence`, `num_mutations`, `fitness`.
    """
    num_mutations = [utils.hamming_distance(ref_seq, s) for s in sequences]
    df = pd.DataFrame(
        dict(
            sequence=list(sequences),
            num_mutations=num_mutations,
            fitness=fitness_fn(sequences)))
    return df


def get_random_split_df(
        df, train_fraction,
        random_state):
    """Returns two dfs, randomly split into `train_fraction` and 1-`train_fraction`."""
    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)]
    return (train_df, test_df)


def get_distance_split_df(
        df, reference_seq,
        distance_threshold):
    """Returns DataFrames split based on distance from `reference_seq`.

    The first df includes all sequences within `distance_threshold` (inclusive) of
    `reference_seq`, and the second df contains the rest.

    Args:
      df: A pd.DataFrame with a `sequence` column.
      reference_seq: A 1D integer-encoded sequence.
      distance_threshold: An integer threshold, sequences in `df` with hamming
        distance within `distance_threshold` (inclusive) of `reference_sequence`
        are included in the first returned df, while the second df contains the
        rest.

    Returns:
      Two pd.DataFrames with the sequences split according to distance.
    """
    distance_from_reference = df.sequence.apply(
        utils.hamming_distance, y=reference_seq)
    train_df = df[distance_from_reference <= distance_threshold].copy()
    test_df = df[~df.index.isin(train_df.index)].copy()
    return (train_df, test_df)


def fit_model(model, df, vocab_size, flatten_inputs):
    """Fit `model` to training data given by `df`."""

    x_train, y_train = utils.get_x_y_from_df(
        df, vocab_size=vocab_size, flatten=flatten_inputs)
    model.fit(x_train, y_train)


def get_regression_metrics(y_pred,
                           y_true):
    """Returns a long-form dictionary of metrics."""
    metrics_dict = {}
    metrics_dict['mse'] = skm.mean_squared_error(y_pred, y_true)
    metrics_dict['std_predicted'] = np.std(y_pred)
    metrics_dict['std_test'] = np.std(y_true)
    if np.std(y_pred) != 0.0 and np.std(y_true) != 0.0:
        # Correlation coefficients are undefined if the deviation of either array is 0.
        coef, _ = stats.spearmanr(y_pred, y_true)
        metrics_dict['spearman_r'] = coef
        coef, _ = stats.kendalltau(y_pred, y_true)
        metrics_dict['kendalltau'] = coef
        coef, _ = stats.pearsonr(y_pred, y_true)
        metrics_dict['pearson_r'] = coef
        metrics_dict['r_squared'] = skm.r2_score(y_pred, y_true)
    return metrics_dict


def compute_regression_metrics_random_split(
        model, df, train_fraction, vocab_size,
        flatten_inputs,
        random_state):
    """Returns regression metrics for a random split of the data."""
    train_df, test_df = get_random_split_df(
        df, train_fraction, random_state=random_state)

    fit_model(model, train_df, vocab_size, flatten_inputs)

    x_test, y_true = utils.get_x_y_from_df(
        test_df, vocab_size=vocab_size, flatten=flatten_inputs)
    y_pred = model.predict(x_test)

    size_dict = {'train_size': len(train_df), 'test_size': len(test_df)}
    metrics_dict = get_regression_metrics(y_pred, y_true)
    metrics_dict.update(size_dict)
    return metrics_dict


def compute_regression_metrics_distance_split(
        model, df, reference_seq,
        distance_threshold, vocab_size,
        flatten_inputs):
    """Returns regression metrics for a distance-based split of the data."""
    train_df, test_df = get_distance_split_df(df, reference_seq,
                                              distance_threshold)

    fit_model(model, train_df, vocab_size, flatten_inputs)

    size_dict = {'train_size': len(train_df), 'test_size': len(test_df)}
    if len(test_df) == 0:
        return size_dict
    else:
        x_test, y_true = utils.get_x_y_from_df(
            test_df, vocab_size=vocab_size, flatten=flatten_inputs)
        y_pred = model.predict(x_test)
        metrics_dict = get_regression_metrics(y_pred, y_true)
        metrics_dict.update(size_dict)
        return metrics_dict


def get_samples_around_wildtype(
        landscape, num_samples, min_num_mutations,
        max_num_mutations, vocab_size, include_singles,
        random_state):
    """Return a DataFrame with a sample centered around the `landscape` wildtype.

    If `include_singles` is true, then L*A singles are added in addition to the
    `num_samples` random samples.

    Args:
      landscape: A landscape with a .evaluate() method.
      num_samples: The number of random samples to draw from the landscape.
      min_num_mutations: The minimum number of mutations to randomly sample.
      max_num_mutations: The maximum number of mutations to randomly sample.
      vocab_size: The number of amino acids in the vocabulary.
      include_singles: Whether to include all single mutants or not.
      random_state: np.random.RandomState which dictates the sampling.

    Returns:
      A DataFrame of samples with `sequence` and `fitness` keys.
    """
    sample = sampling.sample_within_hamming_radius(
        landscape.wildtype_sequence,
        num_samples,
        vocab_size,
        min_mutations=min_num_mutations,
        max_mutations=max_num_mutations,
        random_state=random_state)
    if include_singles:
        all_singles = sampling.get_all_single_mutants(
            landscape.wildtype_sequence, vocab_size=vocab_size)
        sample = np.vstack([sample, all_singles])
    random_state.shuffle(sample)

    sample_df = get_fitness_df(sample, landscape.evaluate,
                               landscape.wildtype_sequence)
    sample_df['sequence_tuple'] = sample_df.sequence.apply(tuple)
    sample_df = sample_df.drop_duplicates('sequence_tuple')
    sample_df = sample_df.drop(labels='sequence_tuple', axis='columns')
    return sample_df


def run_regression_experiment(
        mogwai_filepath, potts_coupling_scale, potts_field_scale,
        potts_single_mut_offset, potts_epi_offset, vocab_size,
        training_set_min_num_mutations, training_set_max_num_mutations,
        training_set_num_samples, training_set_include_singles,
        training_set_random_seed, model_name, model_random_seed,
        metrics_random_split_fraction, metrics_random_split_random_seed,
        metrics_distance_split_radii):
    """Returns metrics for a regression experiment."""
    # Load Potts model landscape
    landscape = potts_model.load_from_mogwai_npz(
        mogwai_filepath,
        coupling_scale=potts_coupling_scale,
        field_scale=potts_field_scale,
        single_mut_offset=potts_single_mut_offset,
        epi_offset=potts_epi_offset)

    # Sample a training dataset.
    training_random_state = np.random.RandomState(training_set_random_seed)
    sample_df = get_samples_around_wildtype(landscape, training_set_num_samples,
                                            training_set_min_num_mutations,
                                            training_set_max_num_mutations,
                                            vocab_size,
                                            training_set_include_singles,
                                            training_random_state)

    # Keras reproducibility
    np.random.seed(model_random_seed)
    python_random.seed(model_random_seed)
    tf.random.set_seed(model_random_seed)

    # Compute regression metrics.
    sequence_length = len(landscape.wildtype_sequence)
    model, flatten_inputs = models.get_model(model_name, sequence_length,
                                             vocab_size)
    metrics_random_state = np.random.RandomState(
        metrics_random_split_random_seed)

    run_metrics = {}

    random_split_metrics = compute_regression_metrics_random_split(
        model, sample_df, metrics_random_split_fraction, vocab_size,
        flatten_inputs, metrics_random_state)
    run_metrics['random_split'] = random_split_metrics

    for distance_threshold in metrics_distance_split_radii:
        distance_metrics = compute_regression_metrics_distance_split(
            model, sample_df, landscape.wildtype_sequence, distance_threshold,
            vocab_size, flatten_inputs)
        run_metrics['distance_split_{}'.format(
            distance_threshold)] = distance_metrics
    return run_metrics


def run_design_experiment(
    mogwai_filepath,
    potts_coupling_scale,
    potts_field_scale,
    potts_single_mut_offset,
    potts_epi_offset,
    vocab_size,
    training_set_min_num_mutations,
    training_set_max_num_mutations,
    training_set_num_samples,
    training_set_include_singles,
    training_set_random_seed,
    model_name,
    model_random_seed,
    mbo_num_designs,
    mbo_random_seed,
    inner_loop_solver_top_k,
    inner_loop_solver_min_mutations,
    inner_loop_solver_max_mutations,
    inner_loop_num_rounds,
    inner_loop_num_samples,
    design_metrics_hit_threshold,
    design_metrics_cluster_hamming_distance,
    design_metrics_fitness_percentiles,
    output_filepath=None,
):
    """Returns a tuple of (metrics, proposal DataFrame) for a design experiment."""
    # Load Potts model landscape
    landscape = potts_model.load_from_mogwai_npz(
        mogwai_filepath,
        coupling_scale=potts_coupling_scale,
        field_scale=potts_field_scale,
        single_mut_offset=potts_single_mut_offset,
        epi_offset=potts_epi_offset)

    # Sample a training dataset.
    training_random_state = np.random.RandomState(training_set_random_seed)
    sample_df = get_samples_around_wildtype(landscape, training_set_num_samples,
                                            training_set_min_num_mutations,
                                            training_set_max_num_mutations,
                                            vocab_size,
                                            training_set_include_singles,
                                            training_random_state)

    # Keras reproducibility
    np.random.seed(model_random_seed)
    python_random.seed(model_random_seed)
    tf.random.set_seed(model_random_seed)

    # MBO
    sequence_length = len(landscape.wildtype_sequence)
    model, flatten_inputs = models.get_model(model_name, sequence_length,
                                             vocab_size)
    inner_loop_solver = solver.RandomMutationSolver(
        inner_loop_solver_min_mutations,
        inner_loop_solver_max_mutations,
        top_k=inner_loop_solver_top_k,
        vocab_size=vocab_size)

    mbo_random_state = np.random.RandomState(mbo_random_seed)
    mbo_solver = solver.ModelBasedSolver(
        model,
        vocab_size=vocab_size,
        flatten_inputs=flatten_inputs,
        inner_loop_num_rounds=inner_loop_num_rounds,
        inner_loop_num_samples=inner_loop_num_samples,
        inner_loop_solver=inner_loop_solver)
    proposals = mbo_solver.propose(
        sample_df, num_samples=mbo_num_designs, random_state=mbo_random_state)
    proposals_df = get_fitness_df(proposals, landscape.evaluate,
                                  landscape.wildtype_sequence)
    if output_filepath:
        _write_seq_df_to_path(proposals_df, output_filepath)

    # Metrics
    run_metrics = {}

    normalized_hit_rate = metrics.diversity_normalized_hit_rate(
        proposals_df, design_metrics_hit_threshold,
        design_metrics_cluster_hamming_distance)
    run_metrics['diversity_normalize_hit_rate'] = normalized_hit_rate

    for percentile in design_metrics_fitness_percentiles:
        percentile_fitness = np.percentile(proposals_df.fitness, q=percentile)
        run_metrics['{}_percentile_fitness'.format(
            percentile)] = percentile_fitness
    return run_metrics


def _write_seq_df_to_path(df, output_filepath):
    with open(output_filepath) as f:
        df.to_csv(f)
