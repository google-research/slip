"""Utilities for tuning Potts Model landscapes."""

import itertools
import math
from typing import Tuple, Dict

import numpy as np

from synthetic_protein_landscapes import potts_model
from synthetic_protein_landscapes import sampling
from synthetic_protein_landscapes import utils


def n_choose_r(n: int, r: int) -> int:
  f = math.factorial
  return f(n) // f(r) // f(n - r)


def get_singles_for_double(
    double_sequence: np.ndarray,
    ref_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Get the constituent single mutants for a double mutant."""
  double = tuple(utils.get_mutations(double_sequence, ref_seq))
  mutation1, mutation2 = double
  mutation1 = [
      mutation1,
  ]
  mutation2 = [
      mutation2,
  ]
  single1 = utils.apply_mutations(ref_seq, mutation1)
  single2 = utils.apply_mutations(ref_seq, mutation2)

  return single1, single2


def get_effect_sizes(sample: np.ndarray, landscape: potts_model.PottsModel,
                     ref_seq: np.ndarray) -> np.ndarray:
  """Get the fitness differences between `sample` and `ref_seq`."""
  fitness = landscape.evaluate(sample)
  ref_fitness = landscape.evaluate(ref_seq)

  return fitness - ref_fitness


def get_epistasis_terms_for_doubles(doubles_sample: np.ndarray,
                                    landscape: potts_model.PottsModel,
                                    ref_seq: np.ndarray) -> np.ndarray:
  """Get the epistasis terms for a sample of double mutants."""
  singles = [get_singles_for_double(s, ref_seq) for s in doubles_sample]
  singles1, singles2 = zip(*singles)

  double_fitness = landscape.evaluate(doubles_sample)
  single1_fitness = landscape.evaluate(np.vstack(singles1))
  single2_fitness = landscape.evaluate(np.vstack(singles2))
  ref_fitness = landscape.evaluate(ref_seq)

  return double_fitness - single1_fitness - single2_fitness + ref_fitness


def get_adaptive_singles(landscape: potts_model.PottsModel) -> np.ndarray:
  """Get single mutants with fitness greater than WT."""
  all_singles = sampling.get_all_single_mutants(landscape.wildtype_sequence,
                                                landscape.vocab_size)
  all_single_effects = landscape.evaluate(all_singles) - landscape.evaluate(
      landscape.wildtype_sequence)

  adaptive_singles = all_singles[all_single_effects >= 0]
  return adaptive_singles


def get_mean_std_adaptive_singles(
    landscape: potts_model.PottsModel) -> Tuple[float, float]:
  """Get mean and standard deviation of the fitness of adaptive single mutants."""
  adaptive_singles = get_adaptive_singles(landscape)
  adaptive_single_effects = landscape.evaluate(
      adaptive_singles) - landscape.evaluate(landscape.wildtype_sequence)
  return np.mean(adaptive_single_effects), np.std(adaptive_single_effects)


def get_adaptive_doubles(landscape: potts_model.PottsModel) -> np.ndarray:
  """Get double mutants with fitness greater than WT."""
  adaptive_singles = get_adaptive_singles(landscape)

  adaptive_doubles = []
  for pair in itertools.combinations(adaptive_singles, 2):
    doubles = utils.add_seqs(pair[0], pair[1], landscape.wildtype_sequence)
    for double in doubles:
      if len(utils.get_mutation_positions(double,
                                          landscape.wildtype_sequence)) == 2:
        adaptive_doubles.append(double)
  adaptive_doubles = np.vstack(adaptive_doubles)
  return adaptive_doubles


def get_mean_std_epi_adaptive(
    landscape: potts_model.PottsModel) -> Tuple[float, float]:
  """Get statistcs of epistatic interactions between adaptive single mutants."""
  adaptive_doubles = get_adaptive_doubles(landscape)

  adaptive_double_epistasis = get_epistasis_terms_for_doubles(
      adaptive_doubles, landscape, landscape.wildtype_sequence)
  return np.mean(adaptive_double_epistasis), np.std(adaptive_double_epistasis)


def get_shift_and_scale_kwargs_for_k(landscape: potts_model.PottsModel,
                                     k: int) -> Dict[str, float]:
  """Return tuning kwarg dict for given distance `k`.

  To choose parameters for a negative epistasis landscape, one must first choose
  a scale k over which randomly recombined adaptive single mutants may not be
  adaptive in combination. This is accomplished by making the epistasis both
  more negative and stronger.
  The first condition is that after taking k random adaptive mutants, the
  typical epistasis is negative the average fitness due to single mutants. This
  gives us:

  (k choose 2) * coupling_scale * (mean_epi_adaptive + epi_offset) + k *
  mean_single_adaptive = 0

  We need one more condition to completely determine coupling_scale and
  epi_offset. We choose to set the mean of the modified (adaptive) epistasis
  distribution equal to the standard deviation of said distribution - so that
  even though the typical amount of epistasis in negative, there still exist
  pairs of adaptive single mutants without strongly negative epistasis. This
  condition is:

  mean_epi_adaptive + epi_offset + std_epi_adaptive = 0

  Solving these equations for coupling_scale and epi_offset gives us the tuning
  knob settings for the Neg-epi landscape. We have:

  epi_offset = -1 * (mean_epi_adaptive + std_epi_adaptive)
  coupling_scale = k * mean_single_adaptive / ( (k choose 2) * std_epi_adaptive)

  Args:
    landscape: The input landscape to be tuned.
    k: The desired number of adaptive singles mutations where negatively
      epistatic terms cancel contributions due to adaptive singles.

  Returns:
    A dictionary of shift/scale kwargs for tuning the landscape.
  """
  mean_epi_adaptive, std_epi_adaptive = get_mean_std_epi_adaptive(landscape)
  mean_single_adaptive, _ = get_mean_std_adaptive_singles(landscape)

  coupling_scale = k * mean_single_adaptive / (
      n_choose_r(k, 2) * std_epi_adaptive)
  epi_offset = -1 * (mean_epi_adaptive + std_epi_adaptive)
  return {'epi_offset': epi_offset, 'coupling_scale': coupling_scale}

