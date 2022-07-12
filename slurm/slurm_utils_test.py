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
"""Tests for slurm utils."""


from absl.testing import absltest
from absl.testing import parameterized

import json

import slurm_utils


class UtilsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name='3 options, 2 locals',
            global_defaults={'g_default': 'GD1', },
            global_options={'g_option': ["GO1", "GO2"]},
            local_defaults_1={'l1_default': 'L1D1', },
            local_options_1={'l1_option': ["L1O1", "L1O2"]},
            local_defaults_2={'l2_default': 'L2D2', },
            local_options_2={'l2_option': ["J2O1", "J2O2"]},
            expected_json=[
                '{"g_option": "GO1", "l1_option": "L1O1", "l1_default": "L1D1", "g_default": "GD1"}',
                '{"g_option": "GO1", "l1_option": "L1O2", "l1_default": "L1D1", "g_default": "GD1"}',
                '{"g_option": "GO2", "l1_option": "L1O1", "l1_default": "L1D1", "g_default": "GD1"}',
                '{"g_option": "GO2", "l1_option": "L1O2", "l1_default": "L1D1", "g_default": "GD1"}',
                '{"g_option": "GO1", "l2_option": "J2O1", "l2_default": "L2D2", "g_default": "GD1"}',
                '{"g_option": "GO1", "l2_option": "J2O2", "l2_default": "L2D2", "g_default": "GD1"}',
                '{"g_option": "GO2", "l2_option": "J2O1", "l2_default": "L2D2", "g_default": "GD1"}',
                '{"g_option": "GO2", "l2_option": "J2O2", "l2_default": "L2D2", "g_default": "GD1"}']

        ),
    )
    def test_params_json(self, expected_json, global_defaults, global_options, local_defaults_1, local_defaults_2, local_options_1, local_options_2):
        local_options_list = [local_options_1, local_options_2]
        local_defaults_list = [local_defaults_1, local_defaults_2]
        actual_json = slurm_utils.get_params_json(
            global_defaults, global_options, local_defaults_list, local_options_list)
        self.assertListEqual(actual_json, expected_json)

    def test_product(self):
        actual = list(slurm_utils.product_dict(num=[1, 2], alpha=['a', 'b']))
        actual_set = set([json.dumps(d) for d in actual])
        expected = [{'num': 1, 'alpha': 'a'},
                    {'num': 2, 'alpha': 'a'},
                    {'num': 1, 'alpha': 'b'},
                    {'num': 2, 'alpha': 'b'}, ]
        expected_set = set([json.dumps(d) for d in expected])
        self.assertSetEqual(actual_set, expected_set)


if __name__ == '__main__':
    absltest.main()
