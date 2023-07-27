# Submitting a gridsearch job to Savio
## Go to savio login node
```
ssh savio
```

Then start or attach to a tmux session
```
tmux a -t slip
```

## Load the correct environment
```
module unload python
module load ml/tensorflow/2.5.0-py37
source activate slip
```

## Restart ssh agent to get access to deploy key
```
eval `ssh-agent`
ssh-add ~/.ssh/id_ed25519  # enter password
```

## Create gridsearch parameters
Specify the desired options in create_gridsearch.py

```
vi create_gridsearch.py
```

## Create the gridsearch directory for job logging

```
$ python create_gridsearch.py
while read i ; do sbatch $DIR/run_experiment.sh "$i"; done < $DIR/regression_params.json
```
This function creates a unique directory for the logfiles, and prints the sbatch command for running the gridsearch.

## Submit parallel jobs for each parameter set
Copy the sbatch command from `create_gridsearch`. This command iterates over the
lines in the `regression_params.json` file to create one job per task.

```
while read i ; do sbatch $DIR/run_experiment.sh "$i"; done < $DIR/regression_params.json
```

## View job standard out
```
$DIR/*.out
```

Or stderr
```
$DIR/*.err
```

# Look at results
## Log in to a jupyter notebook on savio

Navigate to:
`https://ood.brc.berkeley.edu/`.

Start an interactive jupyter session. Navigate to
```
~/git/slip/savio/notebooks
```
