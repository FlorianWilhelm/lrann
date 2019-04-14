# lrann

Experiments on the effectiveness of **l**ow-**r**ank **a**pproximations in collaborative filtering compared to **n**eural **n**etworks.


## Installation

In order to reproduce our experiments: 

1. create an environment `lrann` with the help of [Miniconda][],
   ```bash
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```bash
    conda activate lrann
    ```
3. install `lrann` with:
   ```bash
    python setup.py install # or `develop`
    ```
4. optionally run the unit tests by executing `pytest`

Then take a look into the `experiments` folder.

## Run Experiments

In order to reproduce our research results, we provide an easy way to run different experiments on your own having the provided package installed. Each command requires three command line arguments:

* `-e`: denotes the name of the experiment, see below
* `-c`: config_file: relative path to the config file as already provided in `experiments`
* `-o`: results_file: path where the results .csv-file should be saved

In addition, by adding `-v` you may enable verbose mode.

### Best Neural Network Search
Run the comparative experiments between MF and DNN invoking the following command:

```
run_dnn_experiment -e nn_search -c <config_file> -o <results_file> -v
```

For example:

```
run_dnn_experiment -c experiments/experiment_config.yml -o test_result.csv -v
```

### Matrix Factorization Hyperparameter Optimization
Run the matrix factorization hyperparameter search:

```
run_dnn_experiment -e mf_hyperopt -c <config_file> -o <results_file> -v
```

### Covariance Analysis
In order to retrieve results for the covariance analysis, perform the following command

```
run_dnn_experiment -e covariance -c <config_file> -o <results_file> -v
```

## Note

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

The basic structure and some code was taken from the [Spotlight][] recommender library, which is also MIT licensed.

## Todo

* Convert Numpy docstring style to Google style
* Change command name from `run_dnn_experiment` to `run_lrann_experiment`

[Miniconda]: https://conda.io/en/latest/miniconda.html
[Spotlight]: https://github.com/maciejkula/spotlight
