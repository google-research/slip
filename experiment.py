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
import functools
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm
import tensorflow as tf

import epistasis_selection
import metrics
import models
import potts_model
import sampling
import solver
import tuning
import utils


def get_fitness_df(sequences,
                   fitness_fn,
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
    sequences = np.array(sequences)
    num_mutations = [utils.hamming_distance(ref_seq, s) for s in sequences.tolist()]
    df = pd.DataFrame(
        dict(
            sequence=sequences.tolist(),
            num_mutations=num_mutations,
            fitness=fitness_fn(sequences)))
    return df


def get_random_split_df(df: pd.DataFrame,
                        train_fraction: float,
                        random_state: np.random.RandomState
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns two dfs, randomly split into `train_fraction` and 1-`train_fraction`."""
    train_df = df.sample(frac=train_fraction, random_state=random_state)
    test_df = df[~df.index.isin(train_df.index)].copy()
    return (train_df, test_df)


def get_distance_split_df(
        df: pd.DataFrame,
        reference_seq: Sequence[int],
        distance_threshold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def fit_model(model, df: pd.DataFrame, vocab_size: int, flatten_inputs: bool):
    """Fit `model` to training data given by `df`."""
    x_train, y_train = utils.get_x_y_from_df(
        df, vocab_size=vocab_size, flatten=flatten_inputs)
    model.fit(x_train, y_train)


def get_regression_metrics(y_pred: np.ndarray,
                           y_true: np.ndarray):
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


def compute_regression_metrics(model,  # trained
                               test_df: pd.DataFrame,
                               vocab_size: int,
                               flatten_inputs: bool):
    """Returns regression metrics for a trained model on a given test set."""
    x_test, y_true = utils.get_x_y_from_df(
        test_df, vocab_size=vocab_size, flatten=flatten_inputs)
    y_pred = model.predict(x_test)

    size_dict = {'test_size': len(test_df)}
    metrics_dict = get_regression_metrics(y_pred, y_true)
    metrics_dict.update(size_dict)
    return metrics_dict


def compute_regression_metrics_random_split(
        model,
        df: pd.DataFrame,
        train_fraction: float,
        vocab_size: int,
        flatten_inputs: bool,
        random_state: np.random.RandomState):
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
        model,
        df: pd.DataFrame,
        reference_seq: Sequence[int],
        distance_threshold: int,
        vocab_size: int,
        flatten_inputs: bool):
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
        landscape: potts_model.PottsModel,
        num_samples: int,
        min_num_mutations: int,
        max_num_mutations: int,
        include_singles: bool,
        random_state: np.random.RandomState):
    """Return a DataFrame with a sample centered around the `landscape` wildtype.

    If `include_singles` is true, then L*A singles are added in addition to the
    `num_samples` random samples.

    Args:
      landscape: A landscape with a .evaluate() method.
      num_samples: The number of random samples to draw from the landscape.
      min_num_mutations: The minimum number of mutations to randomly sample.
      max_num_mutations: The maximum number of mutations to randomly sample.
      include_singles: Whether to include all single mutants or not.
      random_state: np.random.RandomState which dictates the sampling.

    Returns:
      A DataFrame of samples with `sequence` and `fitness` keys.
    """
    sample = sampling.sample_within_hamming_radius(
        landscape.wildtype_sequence,
        num_samples,
        landscape.vocab_size,
        min_mutations=min_num_mutations,
        max_mutations=max_num_mutations,
        random_state=random_state)
    if include_singles:
        all_singles = sampling.get_all_single_mutants(
            landscape.wildtype_sequence, vocab_size=landscape.vocab_size)
        sample = np.vstack([sample, all_singles])
    random_state.shuffle(sample)

    sample_df = get_fitness_df(sample, landscape.evaluate,
                               landscape.wildtype_sequence)
    sample_df['sequence_tuple'] = sample_df.sequence.apply(tuple)
    sample_df = sample_df.drop_duplicates('sequence_tuple')
    sample_df = sample_df.drop(labels='sequence_tuple', axis='columns')
    return sample_df


def run_regression_experiment(
        mogwai_filepath: str,
        fraction_adaptive_singles: float,
        fraction_reciprocal_adaptive_epistasis: float,
        epistatic_horizon: float,
        normalize_to_singles: bool,
        training_set_min_num_mutations: int,
        training_set_max_num_mutations: int,
        training_set_num_samples: int,
        training_set_include_singles: bool,
        training_set_random_seed: int,
        model_name: str,
        model_random_seed: int,
        test_set_distances: Sequence[int],
        test_set_n: int,
        test_set_random_seed: int,
        test_set_max_reuse: int,
        test_set_singles_top_k: int,
        test_set_epistatic_top_k: int):
    """Returns metrics for a regression experiment."""
    # Load Potts model landscape
    print('Loading tuned landscape...')
    untuned_landscape = potts_model.load_from_mogwai_npz(mogwai_filepath)
    tuning_kwargs = tuning.get_tuning_kwargs(untuned_landscape,
                                             fraction_adaptive_singles,
                                             fraction_reciprocal_adaptive_epistasis,
                                             epistatic_horizon,
                                             normalize_to_singles=normalize_to_singles)
    landscape = potts_model.load_from_mogwai_npz(
        mogwai_filepath,
        **tuning_kwargs)

    # Sample a training dataset.
    print('Sampling training set...')
    training_random_state = np.random.RandomState(training_set_random_seed)
    train_df = get_samples_around_wildtype(landscape,
                                           training_set_num_samples,
                                           training_set_min_num_mutations,
                                           training_set_max_num_mutations,
                                           training_set_include_singles,
                                           training_random_state)

    # Keras reproducibility
    np.random.seed(model_random_seed)
    python_random.seed(model_random_seed)
    tf.random.set_seed(model_random_seed)

    # Train model.
    print('Training model...')
    sequence_length = len(landscape.wildtype_sequence)
    model, flatten_inputs = models.get_model(model_name,
                                             sequence_length,
                                             landscape.vocab_size)

    fit_model(model, train_df, landscape.vocab_size, flatten_inputs)
    run_metrics = {}

    # Compute regression metrics.
    print('Computing regression metrics on curated tests sets...')
    test_random_state = np.random.RandomState(test_set_random_seed)
    compute_regression_metrics_for_model = functools.partial(compute_regression_metrics,
                                                             vocab_size=landscape.vocab_size,
                                                             flatten_inputs=flatten_inputs)
    get_fitness_df_for_landscape = functools.partial(get_fitness_df,
                                                     fitness_fn=landscape.evaluate,
                                                     ref_seq=landscape.wildtype_sequence)
    train_metrics = compute_regression_metrics_for_model(model, train_df)
    run_metrics['train'] = train_metrics
    for distance in test_set_distances:
        # epistatic test set
        print('Constructing adaptive epistatic test set...')
        epistatic_seqs = epistasis_selection.get_epistatic_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=test_set_n,
            adaptive=True,
            max_reuse=test_set_max_reuse,
            top_k=test_set_epistatic_top_k,
            random_state=test_random_state)
        epistatic_test_df = get_fitness_df_for_landscape(epistatic_seqs)
        epistatic_metrics = compute_regression_metrics_for_model(model, epistatic_test_df)
        run_metrics[f'adaptive_epistatic_seqs_distance_{distance}'] = epistatic_metrics

        print('Constructing deleterious epistatic test set...')
        epistatic_seqs = epistasis_selection.get_epistatic_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=test_set_n,
            adaptive=False,
            max_reuse=test_set_max_reuse,
            top_k=test_set_epistatic_top_k,
            random_state=test_random_state)
        epistatic_test_df = get_fitness_df_for_landscape(epistatic_seqs)
        epistatic_metrics = compute_regression_metrics_for_model(model, epistatic_test_df)
        run_metrics[f'deleterious_epistatic_seqs_distance_{distance}'] = epistatic_metrics

        # adaptive test set
        print('Constructing adaptive singles test set...')
        adaptive_seqs = epistasis_selection.get_adaptive_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=test_set_n,
            adaptive=True,
            max_reuse=test_set_max_reuse,
            top_k=test_set_singles_top_k,
            random_state=test_random_state)
        adaptive_test_df = get_fitness_df_for_landscape(adaptive_seqs)
        adaptive_metrics = compute_regression_metrics_for_model(model, adaptive_test_df)
        run_metrics[f'adaptive_singles_seqs_distance_{distance}'] = adaptive_metrics
    return run_metrics


def run_design_experiment(
    mogwai_filepath: str,
    fraction_adaptive_singles: float,
    fraction_reciprocal_adaptive_epistasis: float,
    epistatic_horizon: float,
    normalize_to_singles: bool,
    training_set_min_num_mutations: int,
    training_set_max_num_mutations: int,
    training_set_num_samples: int,
    training_set_include_singles: bool,
    training_set_random_seed: int,
    model_name: str,
    model_random_seed: int,
    mbo_num_designs: int,
    mbo_random_seed: int,
    inner_loop_solver_top_k: int,
    inner_loop_solver_min_mutations: int,
    inner_loop_solver_max_mutations: int,
    inner_loop_num_rounds: int,
    inner_loop_num_samples: int,
    design_metrics_hit_threshold: float,
    design_metrics_cluster_hamming_distance: int,
    design_metrics_fitness_percentiles: Sequence[float],
    output_filepath: Optional[str] = None,
):
    """Returns a tuple of (metrics, proposal DataFrame) for a design experiment."""
    # Load Potts model landscape
    untuned_landscape = potts_model.load_from_mogwai_npz(mogwai_filepath)
    tuning_kwargs = tuning.get_tuning_kwargs(untuned_landscape,
                                             fraction_adaptive_singles,
                                             fraction_reciprocal_adaptive_epistasis,
                                             epistatic_horizon,
                                             normalize_to_singles)
    landscape = potts_model.load_from_mogwai_npz(
        mogwai_filepath,
        **tuning_kwargs)

    # Sample a training dataset.
    training_random_state = np.random.RandomState(training_set_random_seed)
    sample_df = get_samples_around_wildtype(landscape, training_set_num_samples,
                                            training_set_min_num_mutations,
                                            training_set_max_num_mutations,
                                            training_set_include_singles,
                                            training_random_state)

    # Keras reproducibility
    np.random.seed(model_random_seed)
    python_random.seed(model_random_seed)
    tf.random.set_seed(model_random_seed)

    # MBO
    sequence_length = len(landscape.wildtype_sequence)
    model, flatten_inputs = models.get_model(model_name, sequence_length,
                                             landscape.vocab_size)
    inner_loop_solver = solver.RandomMutationSolver(
        inner_loop_solver_min_mutations,
        inner_loop_solver_max_mutations,
        top_k=inner_loop_solver_top_k,
        vocab_size=landscape.vocab_size)

    mbo_random_state = np.random.RandomState(mbo_random_seed)
    mbo_solver = solver.ModelBasedSolver(
        model,
        vocab_size=landscape.vocab_size,
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
        run_metrics['{}_percentile_fitness'.format(percentile)] = percentile_fitness
    return run_metrics


def _write_seq_df_to_path(df, output_filepath):
    with open(output_filepath) as f:
        df.to_csv(f)
