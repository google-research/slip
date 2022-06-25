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

"""Tests for synthetic_protein_landscapes.potts_model."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import potts_model


class PottsModelTest(parameterized.TestCase):

    def _basic_params(self):
        """Weight matrix and field vector for many test cases."""
        weight_matrix = np.zeros((3, 3, 20, 20))

        weight_matrix[0, 0, 19, 19] = -3.0
        weight_matrix[0, 0, 0, 1] = 3.0 / 2
        weight_matrix[0, 0, 1, 0] = 3.0 / 2
        weight_matrix[0, 1, 0, 0] = 2.5
        weight_matrix[1, 0, 0, 0] = 2.5
        weight_matrix[0, 1, 19, 18] = -2.5
        weight_matrix[1, 0, 18, 19] = -2.5
        weight_matrix[0, 2, 0, 0] = 4.0
        weight_matrix[2, 0, 0, 0] = 4.0

        field_vec = np.zeros((3, 20))
        field_vec[0, 3] = 7.0
        field_vec[1, 0] = 11.0

        return weight_matrix, field_vec

    def _build_problem(self,
                       coupling_scale=1.0,
                       field_scale=1.0,
                       distance_threshold_for_nearby_residues=0,
                       wt_seq=None,
                       **kwargs):
        """Builds a small PottsModel."""
        weight_matrix, field_vec = self._basic_params()
        if wt_seq is None:
            wt_seq = [0, 0, 0]

        return potts_model.PottsModel(
            weight_matrix,
            field_vec,
            distance_threshold_for_nearby_residues=distance_threshold_for_nearby_residues,
            coupling_scale=coupling_scale,
            field_scale=field_scale,
            wt_seq=wt_seq,
            **kwargs)

    def test_get_couplings(self):
        weight_matrix, _ = self._basic_params()
        potts_problem = self._build_problem()
        np.testing.assert_allclose(potts_problem.weight_matrix, weight_matrix)

    def test_get_field_vec(self):
        _, field_vec = self._basic_params()
        potts_problem = self._build_problem()
        np.testing.assert_allclose(potts_problem.field_vec, field_vec)

    @parameterized.named_parameters(
        dict(
            testcase_name='default_idx',
            start_idx=0,
            end_idx=None,
        ),
        dict(
            testcase_name='all_elements',
            start_idx=0,
            end_idx=3,
        ),
        dict(
            testcase_name='two_elements',
            start_idx=1,
            end_idx=2,
        ),
    )
    def test_wildtype_sequence_from_iseq(self, start_idx, end_idx):
        problem = self._build_problem(start_idx=start_idx, end_idx=end_idx)
        wt_seq = [0, 0, 0]
        np.testing.assert_equal(problem.wildtype_sequence,
                                wt_seq[start_idx:end_idx])

    @parameterized.named_parameters(
        dict(
            testcase_name='no_mod',
            dist_thresh=0,
        ),
        dict(
            testcase_name='diag_only',
            dist_thresh=1,
        ),
        dict(
            testcase_name='all_zero',
            dist_thresh=3,
        ),
    )
    def test_diagonal_filtering(self, dist_thresh):
        weight_matrix, _ = self._basic_params()
        seq_len = np.shape(weight_matrix)[0]
        # set close-to-diagonal elements to 0
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) < dist_thresh:
                    weight_matrix[i, j, :, :] = 0.

        potts_problem = self._build_problem(
            distance_threshold_for_nearby_residues=dist_thresh)
        np.testing.assert_allclose(potts_problem.weight_matrix, weight_matrix)

    @parameterized.named_parameters(
        dict(
            testcase_name='default_idx',
            start_idx=0,
            end_idx=None,
        ),
        dict(
            testcase_name='all_elements',
            start_idx=0,
            end_idx=3,
        ),
        dict(
            testcase_name='two_elements',
            start_idx=1,
            end_idx=3,
        ),
    )
    def test_subsequence_parameters(self, start_idx, end_idx):
        weight_matrix, field_vec = self._basic_params()
        potts_problem = self._build_problem(
            start_idx=start_idx, end_idx=end_idx)
        if end_idx is None:
            end_idx = field_vec.shape[0]
        np.testing.assert_allclose(potts_problem.field_vec,
                                   field_vec[start_idx:end_idx, :])
        np.testing.assert_allclose(
            potts_problem.weight_matrix, weight_matrix[np.ix_(
                range(start_idx, end_idx), range(
                    start_idx, end_idx), range(20),
                range(20))])

    @parameterized.named_parameters(
        dict(
            testcase_name='single_sequence',
            start_idx=0,
            end_idx=None,
            test_seqs=[0, 0, 0],
            expected_energy=np.array([17.5]),
        ),
        dict(
            testcase_name='full_sequences',
            start_idx=0,
            end_idx=None,
            test_seqs=[[0, 0, 0], [0, 1, 0]],
            expected_energy=np.array([17.5, 4.0]),
        ),
        dict(
            testcase_name='short_sequence',
            start_idx=0,
            end_idx=2,
            test_seqs=[[0, 0], [0, 1]],
            expected_energy=np.array([13.5, 0.0]),
        ),
    )
    def test_potts_energy(self, start_idx, end_idx, test_seqs, expected_energy):
        potts_problem = self._build_problem(
            start_idx=start_idx, end_idx=end_idx, center_fitness_to_wildtype=False)
        np.testing.assert_allclose(
            potts_problem.evaluate(test_seqs), -expected_energy)

    def test_single_mut_shift(self):
        offset = 1.0
        field_scale = 2.0
        coupling_scale = 5.3
        epi_offset = -0.8
        wt_seq = [0, 0, 0]
        base_problem = self._build_problem(
            wt_seq=wt_seq,
            distance_threshold_for_nearby_residues=1,
        )
        shifted_problem = self._build_problem(
            wt_seq=wt_seq,
            distance_threshold_for_nearby_residues=1,
            field_scale=field_scale,
            single_mut_offset=offset,
            coupling_scale=coupling_scale,
            epi_offset=epi_offset)

        single_mutants = []
        for k in range(3):
            single_mutants += [k * [0] + [i] +
                               (2 - k) * [0] for i in range(1, 20)]

        base_wt_fit = base_problem.evaluate([wt_seq])[0]
        shifted_wt_fit = shifted_problem.evaluate([wt_seq])[0]

        base_single_fits = base_problem.evaluate(single_mutants) - base_wt_fit
        shifted_single_fits = shifted_problem.evaluate(
            single_mutants) - shifted_wt_fit

        rescaled_single_fits = field_scale * (base_single_fits + offset)

        np.testing.assert_allclose(rescaled_single_fits, shifted_single_fits)

    def test_epistasis_shift(self):
        offset = 1.0
        coupling_scale = 2.0
        mut_offset = -4.5
        field_scale = 3.1
        wt_seq = [0, 0, 0]
        base_problem = self._build_problem(
            wt_seq=wt_seq, distance_threshold_for_nearby_residues=1)
        shifted_problem = self._build_problem(
            wt_seq=wt_seq,
            distance_threshold_for_nearby_residues=1,
            coupling_scale=coupling_scale,
            epi_offset=offset,
            field_scale=field_scale,
            single_mut_offset=mut_offset)

        base_wt_fit = base_problem.evaluate([wt_seq])[0]
        shifted_wt_fit = shifted_problem.evaluate([wt_seq])[0]

        single_muts = [[19, 0, 0], [0, 18, 0]]
        double_muts = [[19, 18, 0]]

        # Single mutant fitness gains
        base_single_fits = base_problem.evaluate(single_muts) - base_wt_fit
        shifted_single_fits = shifted_problem.evaluate(
            single_muts) - shifted_wt_fit

        # double mutant fitness gains
        base_double_fits = base_problem.evaluate(double_muts) - base_wt_fit
        shifted_double_fits = shifted_problem.evaluate(
            double_muts) - shifted_wt_fit

        base_epi = base_double_fits[0] - np.sum(base_single_fits)
        shifted_epi = shifted_double_fits[0] - np.sum(shifted_single_fits)

        rescaled_epi = coupling_scale * (base_epi + offset)

        np.testing.assert_allclose(rescaled_epi, shifted_epi)

    def test_center_fitness_to_wildtype(self):
        uncentered_problem = self._build_problem()
        centered_problem = self._build_problem(center_fitness_to_wildtype=True)
        wt_seq = uncentered_problem.wildtype_sequence
        np.testing.assert_equal(wt_seq, centered_problem.wildtype_sequence)
        np.testing.assert_allclose(centered_problem.evaluate([wt_seq]), [0])

    def test_init_asymmetric(self):
        weight_matrix = np.zeros((3, 3, 20, 20))
        weight_matrix[0, 1, 18, 18] = -2.5
        weight_matrix[1, 0, 18, 18] = 2.5

        field_vec = np.ones((3, 20))

        wt_seq = [0, 0, 0]

        with self.assertRaisesRegex(ValueError, 'symmetric'):
            potts_model.PottsModel(weight_matrix, field_vec, wt_seq=wt_seq)


class LoadMogwaiTest(parameterized.TestCase):

    def _write_mock_mogwai_state_dict(self, symmetric):
        l = 10
        v = 5

        weight = np.random.normal(size=(l, v, l, v))
        if symmetric:
            weight = weight + weight.transpose(2, 3, 0, 1)
        bias = np.random.normal(size=(l, v))
        query_seq = np.zeros(l)

        state_dict = {
            'bias': bias,
            'weight': weight,
            'query_seq': query_seq,
        }

        _, filepath = tempfile.mkstemp(suffix='.npz')

        np.savez(filepath, **state_dict)

        self._vocab_size = v
        if symmetric:
            self._mock_mogwai_filepath_symmetric = filepath
        else:
            self._mock_mogwai_filepath_asymmetric = filepath

    def setUp(self):
        super().setUp()
        self._write_mock_mogwai_state_dict(symmetric=True)
        self._write_mock_mogwai_state_dict(symmetric=False)

    def tearDown(self):
        super().tearDown()
        os.remove(self._mock_mogwai_filepath_asymmetric)
        os.remove(self._mock_mogwai_filepath_symmetric)

    def test_asymmetric_load_raises(self):
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            potts_model.load_from_mogwai_npz(
                self._mock_mogwai_filepath_asymmetric)

    def test_symmetric_load(self):
        landscape = potts_model.load_from_mogwai_npz(
            self._mock_mogwai_filepath_symmetric)
        np.testing.assert_allclose(landscape.weight_matrix,
                                   landscape.weight_matrix.transpose(1, 0, 3, 2))


if __name__ == '__main__':
    absltest.main()
