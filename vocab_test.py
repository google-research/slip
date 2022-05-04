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

"""Tests for synthetic_protein_landscapes.vocab."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from vocab import Vocab


class VocabTest(parameterized.TestCase):
  def test_encode_decode(self):
    sequence = "LFKLGAENIFLGRKAATKEEAIRFAGEQLVKGGYVEPEYVQAMLDREKLTPTYLGESIAVPHGTVEAK"
    indices = Vocab.tokenize(sequence)

    self.assertEquals(sequence, Vocab.convert_indices_to_tokens(indices))

  def test_encode_decode(self):
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    indices = Vocab.tokenize(alphabet)
    self.assertEquals(indices, list(range(20)))


if __name__ == '__main__':
  absltest.main()
