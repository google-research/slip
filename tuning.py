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

"""Landscape tuning."""

import itertools
from pprint import pprint
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
from scipy.special import comb

import potts_model
import sampling
import experiment
import utils


def get_adaptive_single_fraction(landscape: potts_model.PottsModel) -> float:
    """Returns the fraction of single mutants with fitness >= to the wildtype fitness."""
    all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence, landscape.vocab_size)
    wt_fitness = landscape.evaluate(landscape.wildtype_sequence).item()
    fraction_adaptive = (landscape.evaluate(all_singles) >= wt_fitness).mean()
    return fraction_adaptive


def get_doubles_df(landscape: potts_model.PottsModel, threshold: float, adaptive: bool) -> pd.DataFrame:
    """Returns a DataFrame of all pairwise combinations of singles, where the singles are above/below a given fitness threshold.

    For each double mutant, the DataFrame includes the columns:
        'fitness': the fitness of the double.
        'a_fitness': the fitness of the first constituent single.
        'b_fitness': the fitness of the second constituent single.
        'residual': the epistatic term: fitness(ab) - fitness(a) - fitness(b) + fitness(wt).

    Args:
        landscape: The landscape on which to compute double fitness.
        threshold: The threshold fitness for filtering single mutants.
        adaptive: A boolean, indicating, if True (False), to construct pairwise combinations from single mutants
            that are above (below) the `threshold` fitness.

    Returns:
        A DataFrame of double mutants with keys 'fitness', 'a_fitness', 'b_fitness', and 'residual'.
    """
    wt_seq = landscape.wildtype_sequence
    all_singles = sampling.get_all_single_mutants(wt_seq, landscape.vocab_size)
    df = experiment.get_fitness_df(all_singles, landscape.evaluate, wt_seq)
    if adaptive:
        singles = df[df.fitness >= threshold].sequence
    else:
        singles = df[df.fitness < threshold].sequence

    if len(singles) < 2:
        raise ValueError(f'Invalid Landscape: fewer than 2 singles for threshold {threshold}')

    doubles = []
    single_a = []
    single_b = []
    for a, b in itertools.combinations(singles, 2):
        # check that this is a true double
        if utils.get_mutation_positions(a, wt_seq) == utils.get_mutation_positions(b, wt_seq):
            continue
        double = utils.add_seqs(a, b, wt_seq)[0]
        doubles.append(double)
        single_a.append(a)
        single_b.append(b)

    doubles_df = experiment.get_fitness_df(doubles, landscape.evaluate, wt_seq)

    single_a_fitness = landscape.evaluate(np.vstack(single_a))
    single_b_fitness = landscape.evaluate(np.vstack(single_b))
    doubles_df['a_fitness'] = single_a_fitness
    doubles_df['b_fitness'] = single_b_fitness

    residual = doubles_df.fitness - doubles_df.a_fitness - doubles_df.b_fitness + landscape.evaluate(wt_seq)
    doubles_df['residual'] = residual
    return doubles_df


def get_epistasis_stats(landscape: potts_model.PottsModel, doubles_df: pd.DataFrame) -> Tuple[float, float]:
    """Returns statistics about epistasis for combinations of singles.

    Mean epistasis effect size is defined as the average effect of epistasis when two single mutants
    from the selected set are combined. The rate of reciprocal sign epistasis is defined as the fraction
    of epistatic interactions that have effects opposing the effects of the constituent singles. (e.g. the rate
    of negative epistasis for adaptive singles in combination).

    Args:
      landscape: The landscape.
      doubles_df: A DataFrame consisting of double mutants to compute epistasis statistic son.

    Returns:
        mean_epistasis_effect: The average effect of epistasis when two single mutants from the selected set are combined.
        rate_reciprocal_sign_epistasis: The fraction of epistatic interactions that have opposite effects to the constituent singles.
    """
    residual = doubles_df.residual
    threshold = landscape.evaluate(landscape.wildtype_sequence).item()

    mean_epistasis_effect = np.mean(residual)
    if (doubles_df.a_fitness >= threshold).all() and (doubles_df.b_fitness >= threshold).all():
        adaptive = True
    elif (doubles_df.a_fitness < threshold).all() and (doubles_df.b_fitness < threshold).all():
        adaptive = False
    else:
        raise ValueError('Reciprocal sign epistasis undefined for inconsistent doubles effects.')

    if adaptive:
        rate_reciprocal_epistasis = (residual < 0).sum() / residual.shape[0]
    else:
        rate_reciprocal_epistasis = (residual > 0).sum() / residual.shape[0]
    return mean_epistasis_effect, rate_reciprocal_epistasis


def get_mean_single_effect(landscape: potts_model.PottsModel, threshold: float, adaptive: bool) -> float:
    """Returns average effect size of singles for a given threshold.

    Args:
      landscape: The landscape.
      threshold: The threshold fitness for singles.
      adaptive: If True, thresholded singles will have fitness >= `threshold`. If False,
        selected singles will have fitness < `threshold`.

    Returns:
      The average effect size of the selected singles.
    """
    all_singles = sampling.get_all_single_mutants(
        landscape.wildtype_sequence, landscape.vocab_size)
    df = experiment.get_fitness_df(
        all_singles, landscape.evaluate, landscape.wildtype_sequence)
    if adaptive:
        mean_single_effect = df[df.fitness >= threshold].fitness.mean()
    else:
        mean_single_effect = df[df.fitness < threshold].fitness.mean()
    return mean_single_effect


def get_epistatic_horizon(landscape: potts_model.PottsModel) -> float:
    r"""Returns the epistatic horizon for the given landscape.

    The "epistatic horizon" is defined as the distance K from the wildtype at which, on average,
    epistatic contributions outweigh linear contributions from adaptive singles. This is interpreted as
    the average distance we can expect a greedy algorithm to perform well on the given landscape.

    Let s_+ be the average adaptive single mutant effect. let e_{+,+} be the average
    epistatic effect for a pair of adaptive singles. Then K is defined as :

    $$
    K * s_+ + (K \\choose 2) e_{+,+} = 0
    $$

    Solving for K:

    $$
    K = \\dfrac{e_{+,+} - 2 s_+}
               {e_{+,+}}
    $$

    Returns:
      The epistatic horizon.
    """
    doubles_df = get_doubles_df(landscape, threshold=0.0, adaptive=True)
    mean_adaptive_epistasis, _ = get_epistasis_stats(landscape, doubles_df)
    mean_adaptive_single = get_mean_single_effect(
        landscape, threshold=0, adaptive=True)
    if mean_adaptive_epistasis == 0.0:
        epistatic_horizon = np.inf
    else:
        epistatic_horizon = (mean_adaptive_epistasis - 2 * mean_adaptive_single) / mean_adaptive_epistasis
    return epistatic_horizon


def get_single_std(landscape: potts_model.PottsModel) -> float:
    """Returns the standard deviation of single mutant effects."""
    all_singles = sampling.get_all_single_mutants(
        landscape.wildtype_sequence, landscape.vocab_size)
    return np.std(landscape.evaluate(all_singles))


def get_landscape_stats(landscape: potts_model.PottsModel) -> dict:
    """Returns a dictionary of landscape statistics."""
    adaptive_doubles_df = get_doubles_df(
        landscape, threshold=0.0, adaptive=True)
    reciprocal_adaptive_epistasis_effect, fraction_reciprocal_adaptive_epistasis = get_epistasis_stats(
        landscape, adaptive_doubles_df)
    stats_dict = {'fraction_adaptive_singles': get_adaptive_single_fraction(landscape),
                  'reciprocal_adaptive_epistasis_effect': reciprocal_adaptive_epistasis_effect,
                  'fraction_reciprocal_adaptive_epistasis': fraction_reciprocal_adaptive_epistasis,
                  'epistatic_horizon': get_epistatic_horizon(landscape),
                  'std_singles': get_single_std(landscape)}
    return stats_dict


def get_normalizing_field_scale(landscape: potts_model.PottsModel) -> float:
    """Returns the field scale tuning parameter to normalize the spread of single mutant fitness."""
    std_singles = get_single_std(landscape)
    return 1.0 / std_singles


def get_single_mut_offset(landscape: potts_model.PottsModel, fraction_adaptive_singles: float) -> float:
    """Returns the single mutant offset tuning parameter to achieve `fraction_adaptive_singles`."""
    all_singles = sampling.get_all_single_mutants(
        landscape.wildtype_sequence, landscape.vocab_size)
    single_fitness = landscape.evaluate(all_singles)
    single_mut_offset = -1 * np.quantile(single_fitness, q=1 - fraction_adaptive_singles)
    # np.quantile returns float64 by default
    return single_mut_offset.astype(np.float32)


def get_epi_offset(landscape: potts_model.PottsModel,
                   fraction_reciprocal_adaptive_epistasis: float,
                   single_mut_offset: float = 0.0) -> float:
    """Returns the epistatic offset tuning parameter to achieve `fraction_reciprocal_adaptive_epistasis`.

    # TODO(nthomas) add note about pretuning the landscape
    """
    # single mutant offset changes the set of adaptive singles
    pretuned_landscape = potts_model.PottsModel(landscape.weight_matrix,
                                                landscape.field_vec,
                                                landscape.wildtype_sequence,
                                                single_mut_offset=single_mut_offset)
    doubles_df = get_doubles_df(
        pretuned_landscape, threshold=0.0, adaptive=True)
    # we want fraction to remain negative
    epi_offset = -1 * np.quantile(doubles_df.residual,
                                  q=fraction_reciprocal_adaptive_epistasis)
    # np.quantile returns float64 by default
    return epi_offset.astype(np.float32)


def get_coupling_scale(landscape: potts_model.PottsModel,
                       epistatic_horizon: float,
                       field_scale: float,
                       single_mut_offset: float,
                       epi_offset: float) -> float:
    """Returns the coupling scale tuning parameter to achieve `epistatic_horizon`.

    The epistatic horizon depends on other landscape statistics.
    Let $s_+$ be the average adaptive single mutant effect.
    Let $e_{+,+}$ be the average epistatic effect for a pair of adaptive singles.
    Then, given K, we solve the following equation to determine the coupling-scale:

    $$K * field_scale (s_+ + field-scale) + (K choose 2) * coupling-scale (e_{+,+} + epi_offset) = 0$$
    """
    if epistatic_horizon < 0:
        raise ValueError('Epistatic horizon must be positive.')
    adaptive_threshold = 0.0
    is_adaptive = True

    pretuned_landscape = potts_model.PottsModel(landscape.weight_matrix,
                                                landscape.field_vec,
                                                wt_seq=landscape.wildtype_sequence,
                                                field_scale=field_scale,
                                                single_mut_offset=single_mut_offset,
                                                epi_offset=epi_offset)
    mean_adaptive_single_effect = get_mean_single_effect(
        pretuned_landscape, threshold=adaptive_threshold, adaptive=is_adaptive)
    mean_epistasis_effect = get_doubles_df(
        pretuned_landscape, threshold=adaptive_threshold, adaptive=is_adaptive).residual.mean()

    K = epistatic_horizon
    numerator = - K * mean_adaptive_single_effect
    denominator = comb(K, 2) * mean_epistasis_effect
    coupling_scale = numerator / denominator
    return coupling_scale


def get_tuning_kwargs(landscape: potts_model.PottsModel,
                      fraction_adaptive_singles: Optional[float] = None,
                      fraction_reciprocal_adaptive_epistasis: Optional[float] = None,
                      epistatic_horizon: Optional[float] = None,
                      normalize_to_singles: bool = False) -> Dict[str, float]:
    """Returns the landscape tuning parameters.

    Each tuning argument is optional. If `None` is passed to a floating point argument,
    or `False` is passed to a boolean argument, the returned tuning parameter dict will not tune that property.

    Args:
      landscape: A landscape.
      fraction_adaptive_singles: The fraction of singles that achieve a fitness > wildtype. If unset,
        this fraction is preserved in the initial landscape.
      fraction_reciprocal_adaptive_epistasis: The fraction of adaptive (+, +) doubles that exhibit negative
        epistasis.
      epistatic_horizon: The average distance at which epistatic effects outweigh single effects.
      normalize_to_singles: A boolean, when True, ensures that the standard deviation of single mutant effects
        is 1.0.

    Returns:
      A dict of named tuning arguments to the potts_model.PottsModel constructor.
    """
    # default tuning params

    print('Untuned landscape stats:')
    pprint(get_landscape_stats(landscape))

    # compute tuning parameters
    if normalize_to_singles:
        field_scale = get_normalizing_field_scale(landscape)
    else:
        field_scale = 1.0

    if fraction_adaptive_singles is not None:
        if not (fraction_adaptive_singles >= 0 and fraction_adaptive_singles <= 1.0):
            raise ValueError(
                f'Invalid fraction: {fraction_adaptive_singles} must be between 0 and 1.')
        single_mut_offset = get_single_mut_offset(
            landscape, fraction_adaptive_singles)
    else:
        single_mut_offset = 0.0

    if fraction_reciprocal_adaptive_epistasis is not None:
        if not (fraction_reciprocal_adaptive_epistasis >= 0 and fraction_reciprocal_adaptive_epistasis <= 1.0):
            raise ValueError(
                f'Invalid fraction: {fraction_reciprocal_adaptive_epistasis} must be between 0 and 1.')
        epi_offset = get_epi_offset(landscape,
                                    fraction_reciprocal_adaptive_epistasis,
                                    single_mut_offset=single_mut_offset)
    else:
        epi_offset = 0.0

    if epistatic_horizon:
        if not (epistatic_horizon >= 0):
            raise ValueError(
                f'Invalid fraction: {epistatic_horizon} must be greater than 0.')
        coupling_scale = get_coupling_scale(
            landscape, epistatic_horizon, field_scale, single_mut_offset, epi_offset)
    else:
        coupling_scale = 1.0

    tuning_kwargs = {'coupling_scale': coupling_scale,
                     'field_scale': field_scale,
                     'single_mut_offset': single_mut_offset,
                     'epi_offset': epi_offset}
    return tuning_kwargs
