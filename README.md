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

## Note

This project has been set up using PyScaffold 3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

The basic structure and some code was taken from the [Spotlight][] recommender library, which is also MIT licensed.

## Todo

* Convert Numpy docstring style to Google style

[Miniconda]: https://conda.io/en/latest/miniconda.html
[Spotlight]: https://github.com/maciejkula/spotlight
