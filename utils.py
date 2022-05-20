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

"""Utility functions for transforming sequence data."""

import collections
import itertools
import functools

from typing import Iterable, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd


def onehot(labels, num_classes):
  """Convert integer encoded sequences to onehot format.

  Args:
    labels: 2D ndarray of integer-encoded sequences, num_seqs x seq_len
    num_classes: Number of classes

  Returns:
    3D onehot encoded array, num_seqs x seq_len x num_classes
  """
  if len(np.shape(labels)) == 3:  # if already one-hot, return labels
    return labels
  elif isinstance(labels, list):
    labels = np.asarray(labels)  # ndarray view
  x = (labels[Ellipsis, None] == np.arange(num_classes)[None])
  return x.astype(np.float32)


def recombine_seqs(
    seq_a,
    seq_b,
    random_state = None):
  """Recombine `seq_a` and `seq_b` into a new variant.

  `seq_a` and `seq_b` must be the same length.

  Args:
   seq_a: A 1D np.ndarray of integers.
   seq_b: A 1D np.ndarray of integers.
   random_state: An optional instance of np.random.RandomState

  Returns:
   An integer encoded sequence.
  """
  if len(seq_a) != len(seq_b):
    raise AssertionError('Input lengths {}, {} do not match.'.format(
        len(seq_a), len(seq_b)))

  if not random_state:
    random_state = np.random.RandomState()
  sequence_order_idx = random_state.randint(0, 2)
  crossover_idx = random_state.randint(len(seq_a))
  sequence_pair = [seq_a, seq_b]
  return _crossover_at_index(sequence_pair[sequence_order_idx],
                             sequence_pair[1 - sequence_order_idx],
                             crossover_idx)


def _crossover_at_index(seq_start, seq_end,
                        crossover_idx):
  """Return the result of crossover of `seq_start` and `seq_end`.

  Args:
   seq_start: A 1D integer encoded sequence.
   seq_end: A 1D integer encoded sequence.
   crossover_idx: an integer specifying the crossover point from `seq_start` to
     `seq_end`.

  Returns:
    A 1D np.ndarray with integer encoded recombination of `seq_start` and
    `seq_end`.
  """
  seq_start = np.array(seq_start)
  seq_end = np.array(seq_end)
  return np.append(seq_start[:crossover_idx], seq_end[crossover_idx:])


def hamming_distance(x, y):
  """Return the Hamming distance between `x` and `y`."""
  assert len(x) == len(y)
  return sum(xi != yi for xi, yi in zip(x, y))


def merge_mutation_sets(
    mutations_a,
    mutations_b):
  """Constructs all possible merges of mutations sets `mutations_a` and `mutations_b`.

  Merging two mutation sets involves combining all the mutations together into
  a combined mutation set. A valid mutation set has mutations to unique
  positions. For a given position P, if there are mutations in the both of the
  two sets, then a valid merge can include either the mutation to position P
  from set A or B. Thus, for M positions with overlap, this function returns
  2^{M} merged mutation sets.

  >>> m1 = [(0, 2), (1, 3)]
  >>> m2 = [(0, 4), (3, 5)]
  >>> merge_mutation_sets(m1, m2)
  [((0, 2), (1, 3), (3, 5)),
   ((0, 4), (1, 3), (3, 5))]

  Args:
    mutations_a: An iterable of tuples encoding mutations (position, mutation).
    mutations_b: See `mutations_a`.

  Returns:
    All possible merges of mutation sets `mutations_a` and `mutations_b`.
  """
  assert all(len(m) == 2 for m in mutations_a)
  assert all(len(m) == 2 for m in mutations_b)
  position_idx = 0

  grouped_mutations = collections.defaultdict(list)
  for m in list(mutations_a) + list(mutations_b):
    position = m[position_idx]
    grouped_mutations[position].append(m)
  singletons = []
  collisions = []
  for position, muts in grouped_mutations.items():
    if len(muts) == 1:
      singletons.append(muts[0])
    else:
      collisions.append(muts)

  to_return = []
  for c in itertools.product(*collisions):
    mutations = c + tuple(singletons)
    sorted_mutations = tuple(sorted(mutations, key=lambda m: m[position_idx]))
    to_return.append(sorted_mutations)
  to_return = sorted(to_return)
  return to_return


def merge_mutation_set_into_multiple(mutation_sets: Sequence[Tuple[Tuple[int, int], ...]], mutation_set: Tuple[Tuple[int, int], ...]) -> List[Tuple[Tuple[int, int], ...]]:
  """Returns the merge of `mutation_set` into multiple `mutation_sets`.

  Returns:
    A list of mutation set tuples.
  """
  if len(mutation_sets) == 0:
    return [mutation_set,]

  all_merges = []
  for mutation_set_a in mutation_sets:
    all_merges.extend(merge_mutation_sets(mutation_set_a, mutation_set))
  return all_merges

def merge_multiple_mutation_sets(mutation_sets: Sequence[Tuple[Tuple[int, int], ...]]) -> List[Tuple[Tuple[int, int], ...]]:
  """Returns the merge of all `mutation_sets`.

  Returns:
    A list of mutation set tuples.
  """
  return functools.reduce(merge_mutation_set_into_multiple, mutation_sets, [])

def get_mutation_positions(sequence, parent):
  """Returns positions where sequence and parent disagree.

  Args:
    sequence: An int-encoded sequence.
    parent: An int-encoded sequence of the same length as `sequence`.

  Returns:
    np.ndarray int array of positions where `sequence` and `parent` disagree.
  """
  sequence = np.array(sequence)
  parent = np.array(parent)
  if len(sequence) != len(parent):
    raise AssertionError('Input sequences must have equal length '
                         '(%d vs. %d).' % (len(sequence), len(parent)))
  return np.where(sequence != parent)[0]


def get_mutations(sequence, parent):
  """Returns locations and values where sequence and parent disagree.

  Args:
    sequence: An int-encoded sequence.
    parent: An int-encoded sequence of the same length as `sequence`.

  Returns:
    List of `(i, sequence[i])` pairs, for each position `i` where
    `sequence[i] != parent[i]`.
  """
  if len(sequence) != len(parent):
    raise AssertionError('Input sequences must have equal length '
                         '(%d vs. %d).' % (len(sequence), len(parent)))

  mutation_positions = get_mutation_positions(sequence, parent)
  return tuple(zip(mutation_positions, sequence[mutation_positions]))


def get_num_mutations(seqs, parent_seq):
  """Returns array of hamming distances from each of `seqs` to `parent_seq`."""
  return np.sum(seqs != np.expand_dims(parent_seq, 0), axis=1)


def apply_mutations(parent,
                    mutations,
                    allow_same = False):
  """Returns a copy of `parent` with `mutations` applied.

  Args:
    parent: 1D integer encoded sequence.
    mutations: A sequence of (position, mutation) tuples. A ValueError is raised
      if these are overlapping.
    allow_same: By default, a ValueError is raised if `mutations` mutate to values
      that are the same as the parent. If this flag is True, this check is ignored.
  """
  parent = np.array(parent)
  mutations = np.array(mutations)
  if not mutations.size:
    return parent

  locations = mutations[:, 0]
  values = mutations[:, 1]

  # assert all mutation locations are unique
  if not len(set(locations)) == len(locations):
    raise ValueError('Invalid mutation: not all mutation locations are unique.')


  if (parent[locations] == values).any() and not allow_same:
    raise ValueError('Invalid mutation: attempting to mutate the parent '
                     'to a value it already has.')

  parent[locations] = values
  return parent


def add_seqs(seq_a, seq_b,
             ref_seq):
  """Returns the additive combination of mutations in `seq_a` and `seq_b`.

  Example usage:
  >>> seq_a = [1, 0, 0]
  >>> seq_b = [0, 2, 0]
  >>> ref_seq = [0, 0, 0]
  >>> combine_seqs(seq_a, seq_b,  ref_seq)
  [[1, 2, 0]]

  >>> seq_a = [1, 1, 0]
  >>> seq_b = [0, 2, 2]
  >>> ref_seq = [0, 0, 0]
  >>> combine_seqs(seq_a, seq_b,  ref_seq)
  [[1, 2, 2], [1, 1, 2]]

  Args:
    seq_a: An integer encoded sequence.
    seq_b: An integer encoded sequence.
    ref_seq: An integer encoded sequence, relative to which mutations are
      defined.

  Returns:
    A set of integer encoded sequences. Each returned sequence is a way to
    add all the mutations in `seq_a` and `seq_b`.
  """
  if len(seq_a) != len(seq_b) != len(ref_seq):
    raise AssertionError('Input sequences and reference must have equal length '
                         '(%d vs %d vs %d).' %
                         (len(seq_a), len(seq_b), len(ref_seq)))

  seq_a = np.array(seq_a)
  seq_b = np.array(seq_b)
  ref_seq = np.array(ref_seq)

  mutations_a = get_mutations(seq_a, ref_seq)
  mutations_b = get_mutations(seq_b, ref_seq)

  merged_mutation_sets = merge_mutation_sets(mutations_a, mutations_b)
  combined_seqs = []
  for mutation_set in merged_mutation_sets:
    combined_seqs.append(apply_mutations(ref_seq, mutation_set))
  return combined_seqs


def get_x_y_from_df(df, vocab_size, flatten = True):
  """Returns one-hot encoded x and y targets."""
  x = onehot(np.vstack(df.sequence.values), num_classes=vocab_size)
  if flatten:
    x = np.reshape(x, [x.shape[0], -1])  # flatten
  y = df.fitness.values
  return x, y


def one_hot_and_flatten(array_2d_sequence, num_classes):
  x = onehot(array_2d_sequence, num_classes=num_classes)
  x = np.reshape(x, [x.shape[0], -1])  # flatten
  return x


def get_top_n_mutation_pairs(interaction_tensor: np.ndarray, top_n: int, lowest: bool = False) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
  """Returns the mutation-pairs for the `top_n` indexes of mutation-pairs with high effect.

  Args:
    interaction_tensor: LxLxAxA 4D tensor.
    top_n: the number of interactions to return.
    lowest: if True, return the coordinates with lowest effect.

  Returns:
    A list of 2-tuple of mutations ((i, a), (j, b)), where i, j are positions and a, b are amino acids.
  """
  if len(interaction_tensor.shape) != 4:
      raise ValueError('Input tensor must be 4D')

  if lowest == True:
    sorted_flat_indexes = np.argsort(interaction_tensor, axis=None)
  else:
    sorted_flat_indexes = np.argsort(-1 * interaction_tensor, axis=None)

  top_n_flat_indexes = sorted_flat_indexes[:top_n]
  top_indexes = np.unravel_index(top_n_flat_indexes, shape=interaction_tensor.shape)
  index_list = np.vstack(top_indexes).T.tolist()

  def _convert_to_mutation_pair(tensor_index):
    position_0 = tensor_index[0]
    aa_0 = tensor_index[2]
    position_1 = tensor_index[1]
    aa_1 = tensor_index[3]
    return ((position_0, aa_0), (position_1, aa_1))

  return [_convert_to_mutation_pair(tuple(idx)) for idx in index_list]
