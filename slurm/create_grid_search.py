"""Script for generating a SLURM gridsearch."""
import argparse
from asyncio import create_task
from datetime import datetime
import json
from os import PathLike
from pathlib import Path
from typing import Iterable, Dict

import numpy as np

import slurm_utils

LOG_DIRECTORY = '/global/scratch/projects/fc_songlab/nthomas/slip/log/'

SBATCH_TEMPLATE_FILEPATH = Path('run_experiment_template.txt')
HT_HELPER_TEMPLATE_FILEPATH = Path('ht_helper_template.txt')


def get_batch_id() -> str:
    """Returns a batch id based on the time."""
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def write_sbatch_script(template_filepath: PathLike, batch_id: str, out_filepath: PathLike) -> None:
    """Reads in a `template` and fills in the batch_id where necessary."""
    with open(template_filepath, 'r') as f:
        text = f.read()
    text = text.format(batch_id=batch_id)
    with open(out_filepath, 'w') as f:
        f.write(text)

def write_ht_helper_sbatch_script(template_filepath: PathLike, batch_id: str, taskfile_path: PathLike, out_filepath: PathLike) -> None:
    """Reads in a ht_helper submission `template` and formats it with batch id and taskfile."""
    with open(template_filepath, 'r') as f:
        text = f.read()
    text = text.format(batch_id=batch_id, taskfile=str(taskfile_path))
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

def get_ht_helper_command_string(job_directory: PathLike) -> str:
    """Return the command string for submitting ht_helper job."""
    return 'sbatch {job_directory}/run_ht_helper.sh'


###############
## ht_helper_utilities
taskfile_line_template = 'source activate slip; ./run_regression_main.py --kwargs_json="{kwargs_json}" --job_id=$HT_TASK_ID --output_dir={output_dir}'

def create_taskfile_from_params(param_dicts, taskfile_path, output_dir):
    """Create a taskfile for ht_helper from a json file of configuration keyword arguments.
    """
    with open(taskfile_path, 'w') as taskfile:
        for param_dict in param_dicts:
            line = taskfile_line_template.format(kwargs_json=json.dumps(param_dict), output_dir=output_dir)
            taskfile.write(line)
            taskfile.write('\n')


def exponentiate_log10_param(params):
    return [10**x for x in params]

def process_raw_param_dict(raw_param_dict):
    """Programatically update log10 parameters. Delete intermediate parameters."""
    param_dict = {}
    param_dict['global_defaults'] = raw_param_dict['global_defaults']
    param_dict['global_options'] = raw_param_dict['global_options']
    param_dict['linear_defaults'] = raw_param_dict['linear_defaults']
    param_dict['cnn_defaults'] = raw_param_dict['cnn_defaults']

    linear_log10_param = 'ridge_alpha_log10'
    linear_options = raw_param_dict['linear_options'].copy()
    linear_options['ridge_alpha'] = exponentiate_log10_param(raw_param_dict['linear_options'][linear_log10_param])
    del linear_options[linear_log10_param]
    param_dict['linear_options'] = linear_options

    cnn_log10_param = 'cnn_adam_learning_rate_log10'
    cnn_options = raw_param_dict['cnn_options'].copy()
    cnn_options['cnn_adam_learning_rate'] = exponentiate_log10_param(raw_param_dict['cnn_options'][cnn_log10_param])
    del cnn_options[cnn_log10_param]
    param_dict['cnn_options'] = cnn_options
    return param_dict

### HT helper
############################################################

def main(jsonfile):
    """Main function for creating a gridsearch. Prints the submission command for ease of use."""
    # create batch ID
    batch_id = get_batch_id()
    # create directory for the job
    job_directory = Path(LOG_DIRECTORY) / Path(batch_id)
    job_directory.mkdir()


    # read parameter grid from a json file...
    with open(jsonfile, 'r') as f:
        raw_param_dict = json.load(f)
    param_dict = process_raw_param_dict(raw_param_dict)

    local_defaults_list = [param_dict['linear_defaults'], param_dict['cnn_defaults']]
    local_options_list = [param_dict['linear_options'], param_dict['cnn_options']]

    # expand full grid
    param_dicts = slurm_utils.expand_grid_params(param_dict['global_defaults'],
                                                param_dict['global_options'],
                                                local_defaults_list,
                                                local_options_list)


    # write regression_params to the job directory
    outfile = job_directory / Path('regression_params.json')
    with open(outfile, 'w') as f:
        f.write('\n'.join([json.dumps(d) for d in param_dicts]))

    # read in the experiment template and add the batch ID
    outfile = job_directory / Path('run_experiment.sh')
    write_sbatch_script(SBATCH_TEMPLATE_FILEPATH, batch_id, outfile)


    # write ht_helper taskfile to the job directory https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/hthelper-script/
    taskfile_path = job_directory / Path('taskfile.json')
    create_taskfile_from_params(param_dicts, taskfile_path, job_directory)
    # read ht helper template.txt, add batch ID, add taskfile, write the sbatch script
    outfile = job_directory / Path('run_ht_helper.sh')
    write_ht_helper_sbatch_script(HT_HELPER_TEMPLATE_FILEPATH, batch_id, taskfile_path, outfile)




    # write the options into a text file for human readability
    write_options_and_defaults(job_directory,
                               param_dict['global_defaults'],
                               param_dict['global_options'],
                               local_defaults_list,
                               local_options_list)

    print('sbatch')
    print(get_command_string(job_directory))
    print('ht_helper')
    print(get_ht_helper_command_string(job_directory))
    return job_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a gridsearch directory for the grid job parameterized by the input json')
    parser.add_argument('--jsonfile', type=str, help='A json file with gridsearch configuration')
    args = parser.parse_args()
    main(args.jsonfile)
