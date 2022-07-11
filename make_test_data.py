"""Utilities for writing mock test files."""

import numpy as np
import utils
from pathlib import Path

test_set_dir = 'test_data/test_sets'
mock_test_set_filepath = Path(test_set_dir) / 'fakepdb' / 'test_set_1.npz'
mock_mogwai_filepath = 'test_data/fakepdb_model_state_dict.npz'
L = 3
A = 5


def write_mock_mogwai_state_dict():
    # linear landscape, all singles are adaptive
    query_seq = np.zeros(L, dtype=np.int32)
    bias = np.zeros((L, A)) - utils.onehot(query_seq, num_classes=A)
    weight = np.zeros((L, A, L, A))

    state_dict = {
        'bias': bias,
        'weight': weight,
        'query_seq': query_seq,
    }

    with open(mock_mogwai_filepath, 'wb') as f:
        np.savez(f, **state_dict)


def write_mock_test_set():
    test_seq = np.arange(L, dtype=np.int32)
    with open(mock_test_set_filepath, 'wb') as f:
        np.savez(f, sequences=np.array([test_seq,]))
