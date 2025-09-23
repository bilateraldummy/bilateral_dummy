This repository provides a python package to construct bilaterally compressed sets. The code is based on that found [here](https://github.com/gchq/coreax).

# Installation
To install with pip, download the repository and run `pip install .` in the repository's root folder. It is recommended to do so in a fresh virtual environment to ensure correct package versions are installed.

Coreax defaults to installing CPU-only JAX, if one has access to a GPU, after installation of Coreax run `pip install -U "jax[cuda12]"`.

# Instructions
In order to run the experiments, first download the supplemental material, from a terminal in the relevant folder, run `chmod +x /run_experiments.sh` followed by `./run_experiments.sh`.

Note that these experiments took around 48 hours to run on GPU. To increase speed, consider reducing the size of the compressed set generated, or the size of the target dataset.

Figures can be generated using the provided notebook, and the `.npy` files produced by the experiment scripts.
