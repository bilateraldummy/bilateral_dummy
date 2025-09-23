# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Solvers for constructing coresets."""

from typing import Callable, Literal, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from jax import grad, vmap
from jaxtyping import Array, Shaped
from tqdm import tqdm as LoudTQDM  # noqa: N812

from coreax.coreset import Coreset
from coreax.data import SupervisedData
from coreax.kernels import (
    ScalarValuedKernel,
    SquaredExponentialKernel,
    median_heuristic,
)
from coreax.solvers.autoencoders import BaseCoder
from coreax.solvers.base import CoresetSolver
from coreax.util import KeyArrayLike, SilentTQDM

# pylint: disable=too-many-locals


def split_classes(y, classes):
    """Split classes array, outputting a dictionary with class key and index values."""
    # Order the classes
    order = jnp.argsort(y)
    ys = y[order]

    # Split the classes into separate vectors
    cut = jnp.flatnonzero(ys[1:] != ys[:-1]) + 1
    splits = jnp.split(order, cut.tolist())  # tolist() for Python split points

    # Organise into dictionary and return
    return {int(c): s for c, s in zip(classes.tolist(), splits)}


class M3DState(eqx.Module):
    """Optimisation results for :class:`M3D`."""


class M3D(CoresetSolver[SupervisedData, M3DState]):
    r"""
    Bilateral distribution compression two stage solver with non-linear encoder.

    :param coreset_size: The desired size of the solved coreset
    :param intrinsic_dimension: The desired dimension of the intrinsic space
    :param random_key: Key for random number generation
    :param encoder_generator: A function which takes a key and outputs a
        :class:`~coreax.solvers.encoders.BaseCoder`encoding into a latent space of
        desired size.
    :param compression_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` defined on
        the intrinsic space :math:`\mathbb{R}^p`. Defaults to :data:`None`, indicating
        a :class:`~coreax.kernels.base.PullBackKernel` is used.
    :param initial_coreset: Initial coreset, must be of size `coreset_size`.
    :param max_coreset_iterations: An integer representing the maximum permitted number
        of gradient steps on coreset. Defaults to :math:`100`.
    :param iterations_per_model: An integer representing the maximum permitted
        number of gradient steps per random latent space. Defaults to :math:`5`.
    :param batch_size: An integer representing the number of data points to use in
        estimating the MMD.
    :param coreset_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the ADAM optimiser with a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    encoder_generator: Callable[[KeyArrayLike], BaseCoder]
    compression_kernel: Union[ScalarValuedKernel, Literal["median_heuristic"], int] = (
        "median_heuristic"
    )
    max_coreset_iterations: int = 100
    iterations_per_model: int = 5
    batch_size: int = 256
    coreset_optimiser: ox.GradientTransformation = ox.sgd(ox.constant_schedule(1e-1))
    track_info: bool = False

    @eqx.filter_jit
    def _compression_maximum_mean_discrepancy(
        self,
        data: Shaped[Array, " n p"],
        coreset: Shaped[Array, " m d"],
        compression_kernel: ScalarValuedKernel,
        encoder: BaseCoder,
    ) -> Shaped[Array, ""]:
        r"""
        Compute the MMD between the compressed set and the projected dataset.

        .. math::

            \Vert \mu_{\mathbb{P}_{XV} - \mu_{\mathbb{P}_Z} \Vert_{\mathcal{H}_k}
        """
        # Encode coreset
        coreset = encoder(coreset)

        # term_1 = compression_kernel.compute(data, data).mean()
        term_2 = compression_kernel.compute(data, coreset).mean()
        term_3 = compression_kernel.compute(coreset, coreset).mean()

        return -2 * term_2 + term_3

    @eqx.filter_jit
    def _coreset_step(
        self,
        encoded_data: Shaped[Array, " n p"],
        coreset: Shaped[Array, " m d"],
        compression_kernel: ScalarValuedKernel,
        coreset_state: ox.OptState,
        encoder: BaseCoder,
    ) -> tuple[Shaped[Array, " m p"], ox.OptState, Shaped[Array, " m p"]]:
        """Do one gradient step on the coreset."""
        # Rename for better formatting and encode coreset
        x, z_x = encoded_data, coreset

        # Evaluate gradient of compressed set
        coreset_gradient = grad(self._compression_maximum_mean_discrepancy, argnums=1)(
            x, z_x, compression_kernel, encoder
        )

        # Do gradient step
        z_x_update, coreset_state = self.coreset_optimiser.update(
            updates=coreset_gradient, state=coreset_state, params=z_x
        )
        z_x_ = jnp.asarray(ox.apply_updates(z_x, z_x_update))

        return z_x_, coreset_state, coreset_gradient

    def reduce(  # noqa: C901, PLR0912, PLR0915, PLR0914
        self,
        dataset: SupervisedData,
        solver_state: Optional[M3DState] = None,
    ) -> tuple[Coreset[SupervisedData], M3DState]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The data to generate the coreset from.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information.
        :return: a tuple of the solved coreset and intermediate solver state information
        """
        # Delete unused solver state
        del solver_state

        # Extract data
        x, y, n, encoder_generator = (
            dataset.data,
            dataset.supervision,
            len(dataset),
            self.encoder_generator,
        )

        # Get the indices of the data corresponding to the different classes
        classes = jnp.unique(y)
        data_indices = split_classes(y.flatten(), classes)

        # Initialise coreset
        if self.track_info:
            print("Initialising coreset with random subset of data...")
        coreset_indices = jr.choice(
            self.random_key, n, shape=(self.coreset_size,), replace=False
        )
        z_x, z_y = x[coreset_indices], y[coreset_indices]

        # Get the indices of the coreset corresponding to the different classes
        coreset_indices = split_classes(z_y.flatten(), classes)

        # Initialise tracking
        compression_errors = []
        progress_bar = LoudTQDM
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM

        # Optimise coreset in latent space of random encoders
        iteration_keys = jr.split(self.random_key, self.max_coreset_iterations)
        for i in progress_bar(range(self.max_coreset_iterations)):
            # Initialise random encoder and vmap it
            encoder = vmap(encoder_generator(iteration_keys[i]))

            for c in classes:
                # Extract indices of data and coreset for current class
                class_data_indices = data_indices[int(c)]
                class_coreset_indices = coreset_indices[int(c)]

                # Extract class from coreset
                z_x_c = z_x[class_coreset_indices]

                # Sample batch_size from full data, for current class, encode it
                batch_indices = jr.choice(
                    iteration_keys[i],
                    class_data_indices,
                    shape=(self.batch_size,),
                    replace=False,
                )
                x_c_batch_encoded = encoder(x[batch_indices])

                # Initialise kernel as median heuristic kernel if one is not given
                compression_kernel = self.compression_kernel
                if not isinstance(compression_kernel, ScalarValuedKernel):
                    if compression_kernel == "median_heuristic":
                        compression_kernel = SquaredExponentialKernel(
                            median_heuristic(x_c_batch_encoded)
                        )
                    elif isinstance(compression_kernel, int):
                        compression_kernel = SquaredExponentialKernel(
                            median_heuristic(x_c_batch_encoded) / compression_kernel
                        )
                    else:
                        raise ValueError(
                            "Kernel must be 'median_heuristic' if not provided."
                        )

                # Initialise optimiser with coreset, for current class
                coreset_state = self.coreset_optimiser.init(z_x_c)

                # Take steps using latent space of random encoder
                for _ in range(self.iterations_per_model):
                    z_x_c, coreset_state, _ = self._coreset_step(
                        x_c_batch_encoded,
                        z_x_c,
                        compression_kernel,
                        coreset_state,
                        encoder,
                    )

                # Update the overall coreset with the optimised class
                z_x = z_x.at[class_coreset_indices].set(z_x_c)

                if self.track_info:
                    # Store loss value
                    compression_error = self._compression_maximum_mean_discrepancy(
                        x_c_batch_encoded, z_x_c, compression_kernel, encoder
                    )
                    compression_errors.append(compression_error)

                    # Print out optimisation information
                    itr_string = f"{i + 1}/{self.max_coreset_iterations}"
                    statement = (
                        f"Coreset Iteration = {itr_string:<12} | "
                        + f"Class = {c} | "
                        + "Batch Compression Error "
                        + f"= {compression_errors[-1].item():<12.10f} | "
                    )
                    print(statement)

        return Coreset(SupervisedData(z_x, z_y), dataset), M3DState()
