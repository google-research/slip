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

"""Functions for sampling sequences."""

from typing import Optional, Sequence

import numpy as np


def _validate_min_max_mutations(min_mutations, max_mutations, seq_len):
    if min_mutations < 0 or min_mutations > seq_len:
        raise ValueError('min_mutations (%f) must be in (0;seq_len)!' %
                         min_mutations)
    if max_mutations < 0 or max_mutations > seq_len:
        raise ValueError('max_mutations (%f) must be in (0;seq_len)!' %
                         max_mutations)
    if min_mutations > max_mutations:
        raise ValueError(
            'min_mutations (%f) must be smaller or equal than max_mutations (%f)!' %
            (min_mutations, max_mutations))


def sample_within_hamming_radius(
        sequence,
        num_samples,
        vocab_size,
        min_mutations,
        max_mutations,
        random_state=None):
    """Returns samples that have a constrained distance to `sequence`.

    Args:
      sequence: The reference sequence. Must be a 1d vector of ints.
      num_samples: The number of samples to draw.
      vocab_size: The vocabulary size.
      min_mutations: The minimum (inclusive) hamming distance of samples to
        `sequence`.
      max_mutations: The maximum (inclusive) hamming distance of samples to
        `sequence`.
      random_state: An optional instance of np.random.RandomState.

    Returns:
      A [num_samples, len(sequence)] numpy array with integer encoded sequences
      that a have a hamming distance between `min_mutations` and `max_mutations`
      to `sequence`.
    """
    _validate_min_max_mutations(min_mutations, max_mutations, len(sequence))
    if not random_state:
        random_state = np.random.RandomState()

    sequence = np.array(sequence)
    samples = np.tile(sequence, (num_samples, 1))

    num_mutations = random_state.choice(
        range(min_mutations, max_mutations + 1), num_samples)
    for sample, num_mutation in zip(samples, num_mutations):
        pos = random_state.choice(len(sequence), num_mutation, replace=False)
        delta = random_state.choice(range(1, vocab_size), num_mutation)
        sample[pos] = (sample[pos] + delta) % vocab_size
    return samples


def get_all_single_mutants(sequence,
                           vocab_size):
    """Returns all single mutants of given `sequence`.

    For a given sequence, at each position there are `vocab_size` - 1 possible
    mutations.

    Args:
      sequence: A 1d vector of ints.
      vocab_size: The vocabulary size.

    Returns:
      A [(V-1)*L, L] numpy array of integer encoded sequences, where L is the
      length of the sequence and V is the vocab size.
    """
    sequence = np.array(sequence)
    seq_length = len(sequence)

    all_singles = []
    for pos in range(seq_length):
        num_singles = vocab_size - 1
        singles_at_pos = np.tile(sequence, (num_singles, 1))
        delta = np.arange(1, vocab_size, 1)
        singles_at_pos[:, pos] = (singles_at_pos[:, pos] + delta) % vocab_size
        all_singles.append(singles_at_pos)
    return np.vstack(all_singles)
