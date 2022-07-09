import random as python_random
import functools
from typing import Sequence, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm

import epistasis_selection
import metrics
import models
import potts_model
import sampling
import solver
import tuning
import utils


def write_test_sets(directory, landscape, distances, n, max_reuse, epistatic_top_k, singles_top_k, random_state):
    print(f'Constructing test sets')
    test_set_name_to_seqs = get_test_sets(landscape, distances, n, max_reuse, epistatic_top_k, singles_top_k, random_state)
    print(test_set_name_to_seqs.keys())
    for test_set_name, test_set_seqs in test_set_name_to_seqs.items():
        filename = f'{test_set_name}.npz'
        filepath = Path(directory) / Path(filename)
        print(f'Writing to {filepath}')
        with open(filepath, 'wb') as f:
            np.savez(f, sequences=np.array(test_set_seqs))


def get_test_sets(landscape, distances, n, max_reuse, epistatic_top_k, singles_top_k, random_state):
    test_set_name_to_seqs = {}
    for distance in distances:
        # epistatic test set
        print(distance)
        print('Constructing adaptive epistatic test set...')
        adaptive_epistatic_seqs = epistasis_selection.get_epistatic_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=n,
            adaptive=True,
            max_reuse=max_reuse,
            top_k=epistatic_top_k,
            random_state=random_state)
        test_set_name_to_seqs[f'adaptive_epistatic_seqs_distance_{distance}'] = adaptive_epistatic_seqs

        print('Constructing deleterious epistatic test set...')
        deleterious_epistatic_seqs = epistasis_selection.get_epistatic_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=n,
            adaptive=False,
            max_reuse=max_reuse,
            top_k=epistatic_top_k,
            random_state=random_state)
        test_set_name_to_seqs[f'deleterious_epistatic_seqs_distance_{distance}'] = deleterious_epistatic_seqs

        # adaptive test set
        print('Constructing adaptive singles test set...')
        adaptive_seqs = epistasis_selection.get_adaptive_seqs_for_landscape(
            landscape=landscape,
            distance=distance,
            n=n,
            adaptive=True,
            max_reuse=max_reuse,
            top_k=singles_top_k,
            random_state=random_state)
        test_set_name_to_seqs[f'adaptive_singles_seqs_distance_{distance}'] = adaptive_seqs

    return test_set_name_to_seqs


def get_mogwai_filepath(pdb):
    return f'/global/home/users/nthomas/git/slip/data/{pdb}_model_state_dict.npz'


def main(pdbs,
         test_set_dir,
         test_set_distances,
         test_set_epistatic_top_k,
         test_set_max_reuse,
         test_set_n,
         test_set_random_seed,
         test_set_singles_top_k):
    test_random_state = np.random.RandomState(test_set_random_seed)
    for pdb in pdbs:
        print(pdb)
        #  make the directory
        test_set_dir = test_set_dir
        pdb_test_set_dir = Path(test_set_dir) / Path(pdb)
        pdb_test_set_dir.mkdir(exist_ok=True)

        # load the landscape
        mogwai_filepath = get_mogwai_filepath(pdb)
        landscape = potts_model.load_from_mogwai_npz(mogwai_filepath)
        write_test_sets(pdb_test_set_dir,
                        landscape,
                        test_set_distances,
                        test_set_n,
                        test_set_max_reuse,
                        test_set_epistatic_top_k,
                        test_set_singles_top_k,
                        test_random_state)


pdbs = ['3gfb_1_A', '3my2_1_A', '5hu4_1_A', '3er7_1_A', '3bfo_1_A']
test_set_kwargs = {
    'test_set_dir': '/global/scratch/projects/fc_songlab/nthomas/slip/data/',
    'test_set_n': 200,
    'test_set_random_seed': 0,
    'test_set_max_reuse': 10,
    'test_set_singles_top_k': 20,
    'test_set_distances': [4, 6],
    'test_set_epistatic_top_k': 1000,}


main(pdbs, **test_set_kwargs)
