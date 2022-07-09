"""Script for generating a SLURM gridsearch."""
from datetime import datetime
import itertools
import json
from os import PathLike
from typing import Dict

from pathlib import Path


LOG_DIRECTORY = '/global/scratch/projects/fc_songlab/nthomas/slip/log/'

defaults = {
    'fraction_adaptive_singles': None,
    'fraction_reciprocal_adaptive_epistasis': None,
    'normalize_to_singles': True,
    'model_random_seed': 0,
    'test_set_dir': '/global/scratch/projects/fc_songlab/nthomas/slip/data/',
    'training_set_min_num_mutations': 0,
    'training_set_max_num_mutations': 3,
    'training_set_num_samples': 5000,

}


options = {
    'training_set_random_seed': list(range(20)),
    'epistatic_horizon': [2.0, 4.0, 6.0, 8.0, 16.0, 32.0],
    'training_set_include_singles': [True, False],
    'model_name': ['linear', 'cnn'],
    'mogwai_filepath': ["/global/home/users/nthomas/git/slip/data/3er7_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/3bfo_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/3gfb_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/3my2_1_A_model_state_dict.npz",
                        "/global/home/users/nthomas/git/slip/data/5hu4_1_A_model_state_dict.npz"]
}

SBATCH_TEMPLATE_FILEPATH = Path('run_experiment_template.txt')


def product_dict(**kwargs):
    """Returns the dictionaries representing the cartesian product over all keyword arguments.

    Example usage:
    >>> list(product_dict(num=[1, 2], alpha=['a', 'b']))
    [{'num': 1, 'alpha': 'a'},
     {'num': 2, 'alpha': 'a'},
     {'num': 1, 'alpha': 'b'},
     {'num': 2, 'alpha': 'b'},]
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def update_dict(input_dict: Dict, update_dict: Dict) -> Dict:
    """Returns a dict including (key: value) pairs in both `input_dict` and `update_dict`."""
    return dict(input_dict, **update_dict)


def write_regression_params(options: Dict, defaults: Dict, outfile: PathLike) -> None:
    """Writes a json of regression parameters.

    Args:
        options: A dictionary with iterable values.
        defaults: A dictionary with atomic values.
        outfile: A filepath to write the job parameters.

    The intention is for the outfile to be in the same directory as the
    run logs.
    """
    json_lines = [json.dumps(update_dict(d, defaults)) for d in product_dict(**options)]

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


def write_options_and_defaults(options: Dict, defaults: Dict, directory: PathLike) -> None:
    """Write the experiment options to the log directory."""
    with open(directory / Path('defaults.json'), 'w') as f:
        json.dump(defaults, f)
    with open(directory / Path('options.json'), 'w') as f:
        json.dump(options, f)


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
    write_regression_params(options, defaults, outfile)

    # read in the experiment template and add the batch ID
    outfile = job_directory / Path('run_experiment.sh')
    write_sbatch_script(batch_id, SBATCH_TEMPLATE_FILEPATH, outfile)

    # write the options into a text file for human readability
    write_options_and_defaults(options, defaults, job_directory)

    print(get_command_string(job_directory))
    return job_directory


if __name__ == '__main__':
    main()
