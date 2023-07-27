"""Script for generating a SLURM gridsearch."""
from datetime import datetime
import json
from os import PathLike
from pathlib import Path
from typing import Iterable, Dict

import numpy as np

import slurm_utils

LOG_DIRECTORY = '/global/scratch/projects/fc_songlab/nthomas/slip/log/'

global_defaults = {
    'fraction_adaptive_singles': None,
    'fraction_reciprocal_adaptive_epistasis': None,
    'normalize_to_singles': True,
    'test_set_dir': '/global/scratch/projects/fc_songlab/nthomas/slip/data/',
    'training_set_min_num_mutations': 1,
    'training_set_max_num_mutations': 3,
}

global_options = {
    'training_set_random_seed': list(range(20)),
    'epistatic_horizon': [2.0, 1024.0],
    'mogwai_filepath': ["/global/home/users/nthomas/git/slip/data/3er7_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/3my2_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/5hu4_1_A_model_state_dict.npz",],
    'training_set_num_samples': [100, 500, 1000, 5000, 10000],
    'training_set_include_singles': [True, False],

}

linear_defaults = {
    'model_name': 'linear',
    'model_random_seed': 0,
    'ridge_fit_intercept': False
}

linear_options = {
    'ridge_alpha': list(10**np.linspace(-3, 2, 11)),
}

cnn_defaults = {
    'model_name': 'cnn',
    'cnn_kernel_size': 5,
    'cnn_hidden_size': 64,
    'model_random_seed': 0,
}

cnn_options = {
    'cnn_adam_learning_rate': list(10**np.linspace(-3, -2, 11)),
    'cnn_batch_size': [64, 128],
    'cnn_num_epochs': [100, 500, 1000],
    'cnn_num_filters': [16, 32, 64],
}

local_defaults_list = [linear_defaults, cnn_defaults]
local_options_list = [linear_options, cnn_options]

SBATCH_TEMPLATE_FILEPATH = Path('run_experiment_template.txt')


def write_regression_params(outfile: PathLike,
                            global_defaults: Dict,
                            global_options: Dict,
                            local_defaults_list: Iterable[Dict],
                            local_options_list: Iterable[Dict]) -> None:
    """Writes a json of regression parameters.

    Args:
        outfile: A filepath to write the job parameters.
        defaults: A dictionary with atomic values.
        options: A dictionary with iterable values.


    The intention is for the outfile to be in the same directory as the
    run logs, so that logs can be compared to the parameters that generated them.
    """
    json_lines = slurm_utils.get_params_json(global_defaults, global_options, local_defaults_list, local_options_list)
    with open(outfile, 'w') as f:
        f.write('\n'.join(json_lines))


def get_batch_id() -> str:
    """Returns a batch id based on the time."""
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def write_sbatch_script(batch_id: str, template_filepath: PathLike, out_filepath: PathLike) -> None:
    """Reads in a `template` and fills in the batch_id where necessary."""
    with open(template_filepath, 'r') as f:
        text = f.read()
    text = text.format(batch_id=batch_id)
    with open(out_filepath, 'w') as f:
        f.write(text)


def write_options_and_defaults(directory: PathLike,
                               global_defaults: Dict,
                               global_options: Dict,
                               local_defaults_list: Iterable[Dict],
                               local_options_list: Iterable[Dict]) -> None:
    """Write the experiment options to the log directory.

    These files are more readable than the gridsearch params file, and can be parsed to
    reproduce the full gridsearch."""
    with open(directory / Path('defaults.json'), 'w') as f:
        for defaults in [global_defaults] + local_defaults_list:
            json.dump(defaults, f)
            f.write('\n')
    with open(directory / Path('options.json'), 'w') as f:
        for options in [global_options] + local_options_list:
            json.dump(options, f)
            f.write('\n')


def get_command_string(job_directory: PathLike) -> str:
    """Return the command string to submit parallel sbatch jobs."""
    command_string = "while read i ; "
    command_string += "do sbatch {job_directory}/run_experiment.sh \"$i\"; "
    command_string += "done < {job_directory}/regression_params.json"
    return command_string.format(job_directory=job_directory)


def main():
    """Main function for creating a gridsearch. Prints the submission command for ease of use."""
    # create batch ID
    batch_id = get_batch_id()
    # create directory for the job
    job_directory = Path(LOG_DIRECTORY) / Path(batch_id)
    job_directory.mkdir()

    # write regression_params to the job directory
    outfile = job_directory / Path('regression_params.json')
    write_regression_params(outfile, global_defaults, global_options, local_defaults_list, local_options_list)

    # read in the experiment template and add the batch ID
    outfile = job_directory / Path('run_experiment.sh')
    write_sbatch_script(batch_id, SBATCH_TEMPLATE_FILEPATH, outfile)

    # write the options into a text file for human readability
    write_options_and_defaults(job_directory, global_defaults, global_options, local_defaults_list, local_options_list)

    print(get_command_string(job_directory))
    return job_directory


if __name__ == '__main__':
    main()
