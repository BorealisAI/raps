# Randomized Parent Search in Causal Bandits

This codebase contains implementation of experiments for the ["Causal
Bandits without Graph Learning"](https://arxiv.org/abs/2301.11401) paper.

### Installation

The `raps` package was developed using Python 3.9 and compatibility with other
Python versions is not guaranteed. First, clone the repository with the
submodules using `git clone --recurse-submodules` command. After that, to
install simply run
```{bash}
pip install -e raps
```

### Running Instructions

Installing the package automatically adds the `raps` script to the $PATH
variable. This script could be run to obtain the results of experiments
measuring the regret, for example:
```{bash}
raps --logdir logdir/tree-p3d2n20.01 --num-parents 3 --domain-size 2 \
    --num-nodes 20 --nruns 10
```
This adds 10 tasks to the task spooler. Task spooler can be installed using
`brew install task-spooler` or `sudo apt install task-spooler`.  If you're
using a Linux system, then be sure to pass `--laucher tsp` argument to the
script as on Linux the command to launch task spooler is different. You can
control the number of tasks run at the same, for example, to launch all 10 runs
at the same time use `ts -S 10` or `tsp -S 10` if on Linux. The progress bars
of the runs could be watched by running `./watch-runs first last` where `first`
and `last` are the first and last indices of the tasks in task spooler.
Alternatively, you can use SLURM to manage the runs, for this pass `--launcher
slurm` argument to the script.

Running the script generates pickle files with matplotlib figure and
experiment objects. Later figure objects are used in the corresponding
jupyter notebook in the directory notebooks to obtain the final figure
aggregating the result from multiple runs of the same experiment.

For the experiments that test our theoretical findings regarding
the number of interventions performed by RAPS see
`notebooks/num-interventions.ipynb` jupyter notebook.
