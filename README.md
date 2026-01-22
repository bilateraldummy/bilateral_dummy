This repository provides a python package to construct bilaterally compressed sets. The code is based on that found [here](https://github.com/gchq/coreax).

# Installation
To install with pip, download the repository and run `pip install .` in the repository's root folder. It is recommended to do so in a fresh virtual environment to ensure correct package versions are installed.

Coreax defaults to installing CPU-only JAX, if one has access to a GPU, after installation of Coreax run `pip install -U "jax[cuda12]"`.

# Instructions
In order to run the experiments, first download the (supplemental material)[https://openreview.net/forum?id=B1pHQZS5LO&referrer=%5Bthe%20profile%20of%20Dominic%20Broadbent%5D(%2Fprofile%3Fid%3D~Dominic_Broadbent1], then from a terminal in the relevant folder, run `chmod +x /run_experiments.sh` followed by `./run_experiments.sh`.

Note that these experiments took around 48 hours to run on GPU. To increase speed, consider reducing the size of the compressed set generated, or the size of the target dataset.

Figures can be generated using the provided notebook, and the `.npy` files produced by the experiment scripts.

# Quick Start

```
  import jax.random as jr
  import optax
  from coreax.data import Data
  from coreax.kernels import SquaredExponentialKernel, median_heuristic
  from coreax.solvers import LinearBilateralDistributionCompression

  # Set random seed and data hyperparams
  SEED = 42
  n_data_points = 1000
  intrinsic_dimension = 2
  ambient_dimension = 100
  
  # Generate intrinsic data from a standard normal
  X_intrinsic = jr.normal(jr.key(SEED), shape=(n_data_points, intrinsic_dimension))
  
  # Randomly generate a matrix with standard normal entries and use this to project our intrinsic data to high dimension
  V = jr.normal(jr.key(SEED + 1), shape=(intrinsic_dimension, ambient_dimension))
  X_ambient = X_intrinsic @ V
  
  # Now we have our target data lying in a  high dimensional space, but we know it has low-dimensional manifold structure
  # as we constructed it as such.
  
  # Define a kernel on the ambient space using the median heuristic
  reconstruction_kernel = SquaredExponentialKernel(median_heuristic(X_ambient))
  
  # Define the solver
  solver = LinearBilateralDistributionCompression(
      coreset_size=10,  # Reduce to 10 observations
      intrinsic_dimension=2,  # Reduce to 2 dimensions
      random_key=jr.key(SEED),
      reconstruction_kernel=reconstruction_kernel,
      compression_kernel="median_heuristic",  # Use a SquaredExponentialKernel defined on the embedding space using the median heuristic
      orthonormal=True,  # Ensure the optimised projection matrix is orthonormal
      num_projection_seeds=1,  # Initialise the projection matrix randomly
      num_projection_epochs=100,  # Optimise for 100 epochs
      projection_optimiser=optax.adam(
          optax.constant_schedule(1e-3)
      ),  # Optimise projection matrix with ADAM
      num_coreset_seeds=1,  # Initialise the compressed set randomly
      max_coreset_iterations=500,  # Optimise the compressed set for a maximum of 500 iterations,
      coreset_optimiser=optax.adam(
          optax.constant_schedule(1e-3)
      ),  # Optimise compressed set with ADAM
      track_info=True,  # Output optimisation information
  )
  
  # Construct the coreset
  compressed_set, state = solver.reduce(Data(X_ambient))
  
  # To access the bilaterally compressed set you can then do
  compressed_set.coreset.data
```
