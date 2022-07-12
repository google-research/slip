"""Script for generating and persisting curated test sets."""
import os
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Sequence, Union

import numpy as np

import epistasis_selection
import potts_model





def get_test_sets(landscape: potts_model.PottsModel,
                  distances: Sequence[int],
                  n: int,
                  max_reuse: int,
                  epistatic_top_k: int,
                  singles_top_k: int,
                  random_state: np.random.RandomState) -> Dict[str, np.ndarray]:
    """Returns a dictionary mapping string labels to arrays of sequences.

    Note that because we generate test sequences using the rank ordering of epistatic effects and
    linear effects, they are not sensitive to landscape tuning, and can be reused for all
    tuned versions of the same landscape.

    See the docstrings for epistasis_selection.get_epistatic_seqs_for_landscape and
    epistasis_selection.get_adaptive_seqs_for_landscape for details on the arguments."""
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

def write_sequence_set_files(directory: Union[os.PathLike, str],
                             name_to_seqs: Dict[str, np.ndarray]) -> Path:
    """Write .npz files containing sequences to input directory.

    Args:
        directory: The target directory.
        """
    print(name_to_seqs.keys())
    for name, seqs in name_to_seqs.items():
        filename = f'{name}.npz'
        filepath = Path(directory) / Path(filename)
        print(f'Writing to {filepath}')
        with open(filepath, 'wb') as f:
            np.savez(f, sequences=np.array(seqs))
    return Path(directory)


def get_mogwai_filepath(pdb: str) -> str:
    return f'/global/home/users/nthomas/git/slip/data/{pdb}_model_state_dict.npz'


def main(pdbs: Sequence,
         test_set_dir: Union(os.pathlike, str),
         test_set_distances: Sequence[int],
         test_set_singles_top_k: int,
         test_set_epistatic_top_k: int,
         test_set_max_reuse: int,
         test_set_n: int,
         test_set_random_seed: int):
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
        test_set_names_to_seqs = get_test_sets(
                        landscape,
                        test_set_distances,
                        test_set_n,
                        test_set_max_reuse,
                        test_set_epistatic_top_k,
                        test_set_singles_top_k,
                        test_random_state)
        write_sequence_set_files(pdb_test_set_dir, test_set_names_to_seqs)

pdbs = ('3gfb_1_A', '3my2_1_A', '5hu4_1_A', '3er7_1_A', '3bfo_1_A')
test_set_config = {
    'test_set_dir': '/global/scratch/projects/fc_songlab/nthomas/slip/data/',
    'test_set_n': 200,
    'test_set_random_seed': 0,
    'test_set_max_reuse': 10,
    'test_set_singles_top_k': 20,
    'test_set_distances': [4, 6],
    'test_set_epistatic_top_k': 1000, }
default_kwargs = MappingProxyType(test_set_config)

main(pdbs, **default_kwargs)
