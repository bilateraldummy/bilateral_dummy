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

from typing import Literal, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from jax import grad, vmap
from jaxtyping import Array, Shaped
from tqdm import tqdm as LoudTQDM  # noqa: N812

from coreax.coreset import Coreset
from coreax.data import Data
from coreax.kernels import (
    PullBackKernel,
    ScalarValuedKernel,
    SquaredExponentialKernel,
    median_heuristic,
)
from coreax.solvers.autoencoders import Autoencoder
from coreax.solvers.base import CoresetSolver
from coreax.util import KeyArrayLike, SilentTQDM

# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements


@eqx.filter_jit
def _reconstruction_maximum_mean_discrepancy(
    batch: Shaped[Array, " B d"],
    projection: Shaped[Array, " d p"],
    reconstruction_kernel: ScalarValuedKernel,
    full_rmmd: bool = False,
) -> Shaped[Array, ""]:
    r"""
    Compute the MMD between the original data set and the reconstructed dataset.

    .. math::

        \Vert \mu_{\mathbb{P}_{X} - \mu_{\mathbb{P}_{XVV^T}} \Vert_{\mathcal{H}_k}
    """
    # Rename for better formatting
    b, v = batch, projection

    # Project and reconstruct the batch
    b_reconstructed = b @ v @ v.T

    # Estimate reconstruction MMD
    term_2 = reconstruction_kernel.compute(b_reconstructed, b).mean()
    term_3 = reconstruction_kernel.compute(b_reconstructed, b_reconstructed).mean()
    if full_rmmd:
        term_1 = reconstruction_kernel.compute(b, b).mean()
        return term_1 - 2 * term_2 + term_3

    return -2 * term_2 + term_3


@eqx.filter_jit
def _compression_maximum_mean_discrepancy(
    data: Shaped[Array, " n p"],
    coreset: Shaped[Array, " m p"],
    compression_kernel: ScalarValuedKernel,
) -> Shaped[Array, ""]:
    r"""
    Compute the MMD between the compressed set and the projected dataset.

    .. math::

        \Vert \mu_{\mathbb{P}_{XV} - \mu_{\mathbb{P}_Z} \Vert_{\mathcal{H}_k}
    """
    # Estimate MMD
    # term_1 = compression_kernel.compute(data, data).mean()
    term_2 = compression_kernel.compute(data, coreset).mean()
    term_3 = compression_kernel.compute(coreset, coreset).mean()

    return -2 * term_2 + term_3


def _project_to_tangent_space(
    projection: Shaped[Array, " d p"],
    array: Shaped[Array, " d p"],
) -> Shaped[Array, " d p"]:
    """Project `array` onto the tangent of the Stiefel manifold at `projection`."""
    product = projection.T @ array
    return array - 1 / 2 * projection @ (product + product.T)


def _retract_to_manifold(
    projection: Shaped[Array, " d p"],
) -> Shaped[Array, " d p"]:
    """Retract `projection` to Stiefel manifold using Q from QR decomposition."""
    return jnp.linalg.qr(projection)[0]


class BilateralDistributionCompressionState(eqx.Module):
    """
    Optimisation results for :class:`_BilateralSolver`.

    :param vs: list of projection matrices after each gradient step
    :param zs: list of coresets after each gradient step
    :param v_grads: list of gradients with respect to projection matrices
    :param z_grads: list of gradients with respect to coreset
    :param compression_errors: list of compression errors after each gradient step
    :param reconstruction_errors: list of reconstruction errors after each gradient step
    :param compression_kernel: kernel defined on intrinsic dimension
    :param reconstruction_kernel: kernel defined on ambient dimension
    :param autoencoder: optimised autoencoder
    """

    vs: list
    zs: list
    v_grads: list
    z_grads: list
    compression_errors: list
    reconstruction_errors: list
    compression_kernel: Union[ScalarValuedKernel, list]
    reconstruction_kernel: ScalarValuedKernel
    autoencoder: Union[Autoencoder, Shaped[Array, "d p"]]


class LinearBilateralDistributionCompression(
    CoresetSolver[Data, BilateralDistributionCompressionState]
):
    r"""
    Bilateral distribution compression two stage solver with linear autoencoder.

    :param coreset_size: The desired size of the solved coreset
    :param intrinsic_dimension: The desired dimension of the intrinsic space
    :param random_key: Key for random number generation
    :param reconstruction_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` defined on
        the ambient space :math:`\mathbb{R}^d`. Defaults to :data:`None`, indicating
        a :class:`~coreax.kernels.SquaredExponentialKernel` initialised using the
        median heuristic on data given to reduce method.
    :param compression_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` defined on
        the intrinsic space :math:`\mathbb{R}^p`. Defaults to 'pull_back' indicating
        a :class:`~coreax.kernels.base.PullBackKernel` is used. If 'median_heuristic'
        a :class:`~coreax.kernels.SquaredExponentialKernel` is used on the intrinsic
        space with length scale given by the median heuristic.
    :param orthonormal: A boolean representing whether to keep the projection on the
        Stiefel manifold or not. Defaults to :data:`False`.
    :param num_projection_seeds: Number of initial seeds to check for optimisation.
        Defaults to :data:`None`, indicating PCA is used.
    :param num_projection_epochs: An integer representing the maximum permitted
        number of gradient steps on projection loss. Defaults to :math:`100`.
    :param projection_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the ADAM optimiser with a constant step schedule of 1e-3.
    :param num_coreset_seeds: Number of initial seeds to check for optimisation.
        Defaults to :data:`None`, indicating  a single random sample is used.
    :param max_coreset_iterations: An integer representing the maximum permitted number
        of gradient steps on coreset. Defaults to :math:`100`.
    :param projection_convergence_parameter: Parameter to decide when gradient descent
        has converged. Defaults to :math:`1e-3`.
    :param coreset_convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param coreset_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the ADAM optimiser with a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    intrinsic_dimension: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    reconstruction_kernel: ScalarValuedKernel
    compression_kernel: Union[
        ScalarValuedKernel, Literal["pull_back"], Literal["median_heuristic"], int
    ] = "median_heuristic"
    orthonormal: bool = False
    num_projection_seeds: Optional[int] = None
    num_projection_epochs: int = 100
    projection_optimiser: ox.GradientTransformation = ox.adam(
        ox.constant_schedule(1e-3)
    )
    projection_batch_size: int = 128
    num_coreset_seeds: Optional[int] = None
    max_coreset_iterations: int = 100
    projection_convergence_parameter: float = 1e-5
    coreset_convergence_parameter: float = 1e-5
    coreset_optimiser: ox.GradientTransformation = ox.adam(ox.constant_schedule(1e-3))
    validation_data: Optional[Shaped[Array, " n d"]] = None
    track_info: bool = False

    def _initialise_projection_with_gaussian_sketch(
        self,
        random_key: KeyArrayLike,
        num_seeds: int,
        ambient_dimension: int,
        batch: Shaped[Array, " B d_x"],
    ) -> Shaped[Array, " d p"]:
        """Initialise the projection with Gaussian sketch."""
        # Sample Gaussian sketches according to size of ambient dimension
        if num_seeds == 1:
            return jr.normal(
                random_key,
                shape=(ambient_dimension, self.intrinsic_dimension),
            ) / jnp.sqrt(ambient_dimension)

        initial_vs = jr.normal(
            random_key,
            shape=(num_seeds, ambient_dimension, self.intrinsic_dimension),
        ) / jnp.sqrt(ambient_dimension)

        # If orthonormal, retract to Stiefel manifold
        if self.orthonormal:
            initial_vs = vmap(_retract_to_manifold)(initial_vs)

        # Find the optimal initialisation according to reconstruction MMD
        reconstruction_errors = vmap(
            _reconstruction_maximum_mean_discrepancy, in_axes=(None, 0, None)
        )(batch, initial_vs, self.reconstruction_kernel)
        return initial_vs[jnp.argmin(reconstruction_errors)]

    def _initialise_projection_with_pca(
        self, data: Shaped[Array, " n d"]
    ) -> Shaped[Array, " d p"]:
        """Initialise the projection with PCA."""
        # Ensure zero mean
        data -= data.mean(axis=0)

        # Compute the empirical covariance matrix
        sigma = 1 / (data.shape[0] - 1) * data.T @ data

        # Do SVD and return projection matrix
        sig_svd = jnp.linalg.svd(sigma)
        return sig_svd.U[:, : self.intrinsic_dimension]

    @eqx.filter_jit
    def _projection_step(
        self,
        batch: Shaped[Array, " B d"],
        projection: Shaped[Array, " d p"],
        projection_state: ox.OptState,
    ) -> tuple[Shaped[Array, " d p"], ox.OptState, Shaped[Array, " d p"]]:
        """Do one gradient step on the projection."""
        # Rename for better formatting
        b, v = batch, projection

        # Evaluate gradient wrt V
        v_gradient = grad(_reconstruction_maximum_mean_discrepancy, argnums=1)(
            b, v, self.reconstruction_kernel
        )

        # Do gradient step on v
        v_update, projection_state = self.projection_optimiser.update(
            updates=v_gradient, state=projection_state, params=v
        )
        v_ = jnp.asarray(ox.apply_updates(v, v_update))

        return v_, projection_state, v_gradient

    @eqx.filter_jit
    def _stiefel_projection_step(
        self,
        batch: Shaped[Array, " B d"],
        projection: Shaped[Array, " d p"],
        projection_state: ox.OptState,
    ) -> tuple[Shaped[Array, " d p"], ox.OptState, Shaped[Array, " d p"]]:
        """Do one gradient step on the projection, restricted to Stiefel manifold."""
        # Rename for better formatting
        b, v = batch, projection

        # Evaluate gradient wrt V
        v_gradient = grad(_reconstruction_maximum_mean_discrepancy, argnums=1)(
            b, v, self.reconstruction_kernel
        )

        # Project the gradient to the tangent space defined by the current iterate v
        projected_v_gradient = _project_to_tangent_space(v, v_gradient)

        # Do gradient step on v
        v_update, projection_state = self.projection_optimiser.update(
            updates=projected_v_gradient, state=projection_state, params=v
        )
        v_ = jnp.asarray(ox.apply_updates(v, v_update))

        # Retract V_ to Stiefel manifold
        v_ = _retract_to_manifold(v_)

        return v_, projection_state, v_gradient

    @eqx.filter_jit
    def _coreset_step(
        self,
        data: Shaped[Array, " n p"],
        coreset: Shaped[Array, " m p"],
        compression_kernel: ScalarValuedKernel,
        coreset_state: ox.OptState,
    ) -> tuple[Shaped[Array, " m p"], ox.OptState, Shaped[Array, " m p"]]:
        """Do one gradient step on the coreset."""
        # Rename for better formatting
        x, z = data, coreset

        # Evaluate gradient of compressed set
        coreset_gradient = grad(_compression_maximum_mean_discrepancy, argnums=1)(
            x, z, compression_kernel
        )

        # Do gradient step
        z_update, coreset_state = self.coreset_optimiser.update(
            updates=coreset_gradient, state=coreset_state, params=z
        )
        z_ = jnp.asarray(ox.apply_updates(z, z_update))

        return z_, coreset_state, coreset_gradient

    def reduce(  # noqa: C901, PLR0912, PLR0915, PLR0914
        self,
        dataset: Data,
        solver_state: Optional[BilateralDistributionCompressionState] = None,
    ) -> tuple[Coreset[Data], BilateralDistributionCompressionState]:
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
        x, n = dataset.data, len(dataset)

        # Initialise the projection
        if self.num_projection_seeds is None:
            if self.track_info:
                print("Initialising projection matrix with PCA...")
            v = self._initialise_projection_with_pca(x)
        else:
            if self.track_info:
                print("Initialising projection matrix with Gaussian sketch...")

            # Get a batch to compute an initial projection
            batch_key, seed_key = jr.split(self.random_key)
            batch_indices = jr.choice(
                batch_key, n, shape=(self.projection_batch_size,), replace=False
            )
            batch = x[batch_indices]

            v = self._initialise_projection_with_gaussian_sketch(
                seed_key, self.num_projection_seeds, x.shape[1], batch
            )

        # Initialise optimiser
        projection_state = self.projection_optimiser.init(v)

        # Initialise tracking lists
        reconstruction_errors, vs, v_grads, v_grad = ([], [v], [], 0)

        progress_bar = LoudTQDM
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM

            # Store loss value
            if self.validation_data is not None:
                reconstruction_error = _reconstruction_maximum_mean_discrepancy(
                    self.validation_data, v, self.reconstruction_kernel, True
                )
            else:
                reconstruction_error = _reconstruction_maximum_mean_discrepancy(
                    x, v, self.reconstruction_kernel, True
                )
            reconstruction_errors.append(reconstruction_error)

            # Print out optimisation information
            itr_string = f"{0}/{self.num_projection_epochs}"
            statement = (
                f"Epoch = {itr_string:<12} | "
                + "Validation Error"
                + f" = {reconstruction_error:<12.10f} | "
            )
            print(statement)

        # Do gradient steps on projection
        permute_key, _ = jr.split(self.random_key)
        for i in progress_bar(range(self.num_projection_epochs)):
            # Shuffle the data
            permute_key, _ = jr.split(permute_key)
            permutation = jr.permutation(permute_key, n)

            # Learn on the epoch
            for j in range(0, n, self.projection_batch_size):
                batch = x[permutation[j : j + self.projection_batch_size]]
                if self.orthonormal:
                    v, projection_state, v_grad = self._stiefel_projection_step(
                        batch, v, projection_state
                    )
                else:
                    v, projection_state, v_grad = self._projection_step(
                        batch, v, projection_state
                    )

            # Store parameter info
            vs.append(v)

            if self.track_info:
                # Store gradient
                v_grads.append(v_grad)

                # Store loss value
                if self.validation_data is not None:
                    reconstruction_error = _reconstruction_maximum_mean_discrepancy(
                        self.validation_data, v, self.reconstruction_kernel, True
                    )
                else:
                    reconstruction_error = _reconstruction_maximum_mean_discrepancy(
                        x, v, self.reconstruction_kernel, True
                    )
                reconstruction_errors.append(reconstruction_error)

                # Print out optimisation information
                itr_string = f"{i + 1}/{self.num_projection_epochs}"
                statement = (
                    f"Epoch = {itr_string:<12} | "
                    + "Validation Error"
                    + f" = {reconstruction_errors[-1].item():<12.10f} | "
                    + f"V Gradient Norm = {jnp.linalg.norm(v_grad).item():<12.10f} | "
                )
                print(statement)

            # Check convergence
            gradient_norm = jnp.linalg.norm(v_grad).item()
            if gradient_norm < self.projection_convergence_parameter:
                print("\nConverged!")
                break

        # Project the data down with final projection matrix
        x_projected = x @ v

        # Initialise intrinsic kernel as pull-back kernel or median heuristic kernel
        compression_kernel = self.compression_kernel
        if not isinstance(compression_kernel, ScalarValuedKernel):
            if compression_kernel == "pull_back":
                compression_kernel = PullBackKernel(
                    self.reconstruction_kernel, lambda x: x @ v.T
                )
            elif compression_kernel == "median_heuristic":
                heuristic_indices = jr.choice(
                    jr.split(self.random_key)[1],
                    n,
                    shape=(1000,),
                    replace=False,
                )
                compression_kernel = SquaredExponentialKernel(
                    median_heuristic(x_projected[heuristic_indices])
                )
            elif isinstance(compression_kernel, int):
                heuristic_indices = jr.choice(
                    jr.split(self.random_key)[1],
                    n,
                    shape=(1000,),
                    replace=False,
                )
                compression_kernel = SquaredExponentialKernel(
                    median_heuristic(x_projected[heuristic_indices])
                    / compression_kernel
                )
            else:
                raise ValueError(
                    "Compression kernel must be one of"
                    + " 'pull_back' or 'median_heuristic'",
                )

        if self.track_info:
            print("Initialising coreset with random subset of projected data...")
        coreset_indices = jr.choice(
            self.random_key, n, shape=(self.coreset_size,), replace=False
        )
        if self.num_coreset_seeds is not None and self.num_coreset_seeds != 1:
            # Sample sets of indices to check
            seed_keys = jr.split(self.random_key, num=(self.num_coreset_seeds,))
            seed_indices = vmap(
                lambda key: jr.choice(
                    key,
                    n,
                    shape=(self.coreset_size,),
                    replace=False,
                ),
                in_axes=0,
            )(seed_keys)

            # Compute the compression loss for each initial coreset
            seed_losses = vmap(
                _compression_maximum_mean_discrepancy,
                in_axes=(None, 0, None),
            )(
                x_projected,
                x_projected[seed_indices],  # Extract initial coresets as 3d array
                compression_kernel,
            )
            coreset_indices = seed_indices[jnp.argmin(seed_losses)]
        z = x_projected[coreset_indices]

        # Initialise optimiser
        coreset_state = self.coreset_optimiser.init(z)

        # Initialise tracking lists
        compression_errors, zs, z_grads = [], [z], []

        if self.track_info:
            # Store initial loss value
            compression_error = _compression_maximum_mean_discrepancy(
                x_projected,
                z,
                compression_kernel,
            )
            compression_errors.append(compression_error)

            # Print out optimisation information
            itr_string = f"{0}/{self.max_coreset_iterations}"
            statement = (
                f"Coreset Iteration = {itr_string:<12} | "
                + f"Compression Error = {compression_errors[-1].item():<12.10f} | "
            )
            print(statement)

        for j in progress_bar(range(self.max_coreset_iterations)):
            z, coreset_state, z_grad = self._coreset_step(
                x_projected, z, compression_kernel, coreset_state
            )

            # Store parameter info
            zs.append(z)

            if self.track_info:
                # Store gradient
                z_grads.append(z_grad)

                # Store initial loss values
                compression_error = _compression_maximum_mean_discrepancy(
                    x_projected,
                    z,
                    compression_kernel,
                )
                compression_errors.append(compression_error)

                # Print out optimisation information
                itr_string = f"{j + 1}/{self.max_coreset_iterations}"
                statement = (
                    f"Coreset Iteration = {itr_string:<12} | "
                    + "Compression Error "
                    + f"= {compression_errors[-1].item():<12.10f} | "
                    + f"Z Gradient Norm = {jnp.linalg.norm(z_grad).item():<12.10f} | "
                )
                print(statement)

            # Check convergence
            gradient_norm = jnp.linalg.norm(z_grad).item()
            if gradient_norm < self.coreset_convergence_parameter:
                print("\nConverged!")
                break

        return Coreset(Data(z), dataset), BilateralDistributionCompressionState(
            vs,
            zs,
            v_grads,
            z_grads,
            compression_errors,
            reconstruction_errors,
            compression_kernel,
            self.reconstruction_kernel,
            v,
        )


class NonLinearBilateralDistributionCompression(
    CoresetSolver[Data, BilateralDistributionCompressionState]
):
    r"""
    Bilateral distribution compression two stage solver with non-linear autoencoder.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param autoencoder: An instance of :class:`~coreax.solvers.autoencoders.Autoencoder`
        encoding into and decoding out of a latent space of desired size.
    :param reconstruction_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` defined on
        the ambient space :math:`\mathbb{R}^d`. Defaults to :data:`None`, indicating
        a :class:`~coreax.kernels.SquaredExponentialKernel` initialised using the
        median heuristic on data given to reduce method.
    :param compression_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` defined on
        the intrinsic space :math:`\mathbb{R}^p`. Defaults to :data:`None`, indicating
        a :class:`~coreax.kernels.base.PullBackKernel` is used.
    :param regularisation_parameter: A float indicating the strength of the MSRE term
        in the reconstruction loss. Defaults to 1.
    :param num_projection_epochs: An integer representing the maximum permitted
        number of gradient steps on projection loss. Defaults to :math:`100`.
    :param autoencoder_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the ADAM optimiser with a constant step schedule of 1e-3.
    :param num_coreset_seeds: Number of initial seeds to check for optimisation.
        Defaults to :data:`None`, indicating  a single random sample is used.
    :param max_coreset_iterations: An integer representing the maximum permitted number
        of gradient steps on coreset. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param coreset_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the ADAM optimiser with a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    autoencoder: Autoencoder
    reconstruction_kernel: ScalarValuedKernel
    compression_kernel: Union[
        ScalarValuedKernel, Literal["pull_back"], Literal["median_heuristic"], int
    ] = "median_heuristic"
    regularisation_parameter: float = 1.0
    num_autoencoder_epochs: int = 10
    autoencoder_optimiser: ox.GradientTransformation = ox.adam(
        ox.constant_schedule(1e-3)
    )
    autoencoder_batch_size: int = 128
    num_coreset_seeds: Optional[int] = None
    max_coreset_iterations: int = 100
    convergence_parameter: float = 1e-3
    coreset_optimiser: ox.GradientTransformation = ox.adam(ox.constant_schedule(1e-3))
    validation_data: Optional[Shaped[Array, " n d"]] = None
    track_info: bool = False

    @eqx.filter_jit
    def _reconstruction_loss(
        self, data: Shaped[Array, "n d"], autoencoder: Autoencoder
    ):
        """Convex combination of RMMD and MSRE."""
        reconstructed_data = vmap(autoencoder)(data)
        rmmd_term_2 = self.reconstruction_kernel.compute(
            data, reconstructed_data
        ).mean()
        rmmd_term_3 = self.reconstruction_kernel.compute(
            reconstructed_data, reconstructed_data
        ).mean()
        msre = ((data - reconstructed_data) ** 2).mean()

        return msre - 2 * rmmd_term_2 + rmmd_term_3

    @eqx.filter_jit
    def _reconstruction_mmd(
        self, features: Shaped[Array, "n d"], autoencoder: Autoencoder
    ):
        """Convex combination of RMMD and MSRE."""
        reconstructed_features = vmap(autoencoder)(features)
        rmmd_term_1 = (self.reconstruction_kernel.compute(features, features)).mean()
        rmmd_term_2 = (
            self.reconstruction_kernel.compute(features, reconstructed_features)
        ).mean()
        rmmd_term_3 = (
            self.reconstruction_kernel.compute(
                reconstructed_features, reconstructed_features
            )
        ).mean()

        return rmmd_term_1 - 2 * rmmd_term_2 + rmmd_term_3

    @eqx.filter_jit
    def _autoencoder_step(
        self,
        batch: Shaped[Array, " B d"],
        autoencoder: Autoencoder,
        autoencoder_state: ox.OptState,
    ) -> tuple[Autoencoder, ox.OptState]:
        """Do one gradient step on the parameters of the autoencoder."""

        def _reconstruction_loss(autoencoder: Autoencoder):
            """Convex combination of RMMD and MSRE."""
            return self._reconstruction_loss(batch, autoencoder)

        # Evaluate gradient wrt autoencoder parameters
        grads = eqx.filter_grad(_reconstruction_loss)(autoencoder)

        # Do gradient step on autoencoder parameters
        updates, autoencoder_state = self.autoencoder_optimiser.update(
            grads, autoencoder_state, autoencoder
        )
        autoencoder = eqx.apply_updates(autoencoder, updates)

        return autoencoder, autoencoder_state

    @eqx.filter_jit
    def _coreset_step(
        self,
        data: Shaped[Array, " n p"],
        coreset: Shaped[Array, " m p"],
        compression_kernel: ScalarValuedKernel,
        coreset_state: ox.OptState,
    ) -> tuple[Shaped[Array, " m p"], ox.OptState, Shaped[Array, " m p"]]:
        """Do one gradient step on the coreset."""
        # Rename for better formatting
        x, z = data, coreset

        # Evaluate gradient of compressed set
        coreset_gradient = grad(_compression_maximum_mean_discrepancy, argnums=1)(
            x, z, compression_kernel
        )

        # Do gradient step
        z_update, coreset_state = self.coreset_optimiser.update(
            updates=coreset_gradient, state=coreset_state, params=z
        )
        z_ = jnp.asarray(ox.apply_updates(z, z_update))

        return z_, coreset_state, coreset_gradient

    def reduce(  # noqa: C901, PLR0912, PLR0915, PLR0914
        self,
        dataset: Data,
        solver_state: Optional[BilateralDistributionCompressionState] = None,
    ) -> tuple[Coreset[Data], BilateralDistributionCompressionState]:
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
        x, n, autoencoder = dataset.data, len(dataset), self.autoencoder

        # Initialise optimiser
        autoencoder_state = self.autoencoder_optimiser.init(
            eqx.filter(autoencoder, eqx.is_array)
        )

        # Initialise tracking lists
        reconstruction_errors = []

        progress_bar = LoudTQDM
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM

            # Store initial loss values
            if self.validation_data is not None:
                reconstruction_error = self._reconstruction_mmd(
                    self.validation_data, autoencoder
                ).item()
            else:
                reconstruction_error = self._reconstruction_mmd(x, autoencoder).item()
            reconstruction_errors.append(reconstruction_error)

            # Print out optimisation information
            itr_string = f"{0}/{self.num_autoencoder_epochs}"
            statement = (
                f"Epoch = {itr_string:<12} | "
                + "Validation Error"
                + f" = {reconstruction_error:<12.10f} | "
            )
            print(statement)

        # Do gradient steps on autoencoder
        permute_key, _ = jr.split(self.random_key)
        for i in progress_bar(range(self.num_autoencoder_epochs)):
            # Shuffle the data
            permute_key, _ = jr.split(permute_key)
            permutation = jr.permutation(permute_key, n)

            # Learn on the epoch
            for j in range(0, n, self.autoencoder_batch_size):
                batch = x[permutation[j : j + self.autoencoder_batch_size]]
                autoencoder, autoencoder_state = self._autoencoder_step(
                    batch, autoencoder, autoencoder_state
                )

            if self.track_info:
                # Store loss value
                if self.validation_data is not None:
                    reconstruction_error = self._reconstruction_mmd(
                        self.validation_data, autoencoder
                    ).item()
                else:
                    reconstruction_error = self._reconstruction_mmd(
                        x, autoencoder
                    ).item()
                reconstruction_errors.append(reconstruction_error)

                # Print out optimisation information
                itr_string = f"{i + 1}/{self.num_autoencoder_epochs}"
                statement = (
                    f"Epoch = {itr_string:<12} | "
                    + "Validation Error"
                    + f" = {reconstruction_error:<12.10f} | "
                )
                print(statement)

        # Project the data down with final encoder
        # x_projected = vmap(autoencoder.encoder)(x)
        encodings = []
        for i in range(0, n, 1000):
            xb = x[i : i + 1000]
            zb = vmap(autoencoder.encoder)(xb)
            encodings.append(zb)
        x_projected = jnp.concatenate(encodings, axis=0)

        # Initialise intrinsic kernel as pull-back kernel or median heuristic kernel
        compression_kernel = self.compression_kernel
        if not isinstance(compression_kernel, ScalarValuedKernel):
            if compression_kernel == "pull_back":
                compression_kernel = PullBackKernel(
                    self.reconstruction_kernel, autoencoder.decoder
                )
            elif compression_kernel == "median_heuristic":
                heuristic_indices = jr.choice(
                    jr.split(self.random_key)[1],
                    n,
                    shape=(1000,),
                    replace=False,
                )
                compression_kernel = SquaredExponentialKernel(
                    median_heuristic(x_projected[heuristic_indices])
                )
            elif isinstance(compression_kernel, int):
                heuristic_indices = jr.choice(
                    jr.split(self.random_key)[1],
                    n,
                    shape=(1000,),
                    replace=False,
                )
                compression_kernel = SquaredExponentialKernel(
                    median_heuristic(x_projected[heuristic_indices])
                    / compression_kernel
                )
            else:
                raise ValueError(
                    "Compression kernel must be one of"
                    + " 'pull_back' or 'median_heuristic'",
                )

        if self.track_info:
            print("Initialising coreset with random subset of projected data...")
        coreset_indices = jr.choice(
            self.random_key, n, shape=(self.coreset_size,), replace=False
        )
        if self.num_coreset_seeds is not None and self.num_coreset_seeds != 1:
            # Sample sets of indices to check
            seed_keys = jr.split(self.random_key, num=(self.num_coreset_seeds,))
            seed_indices = vmap(
                lambda key: jr.choice(
                    key,
                    n,
                    shape=(self.coreset_size,),
                    replace=False,
                ),
                in_axes=0,
            )(seed_keys)

            # Compute the compression loss for each initial coreset
            seed_losses = vmap(
                _compression_maximum_mean_discrepancy,
                in_axes=(None, 0, None),
            )(
                x_projected,
                x_projected[seed_indices],  # Extract initial coresets as 3d array
                compression_kernel,
            )
            coreset_indices = seed_indices[jnp.argmin(seed_losses)]
        z = x_projected[coreset_indices]

        # Initialise optimiser
        coreset_state = self.coreset_optimiser.init(z)

        # Initialise tracking lists
        compression_errors, zs, z_grads = [], [z], []

        if self.track_info:
            # Store initial loss value
            compression_error = _compression_maximum_mean_discrepancy(
                x_projected,
                z,
                compression_kernel,
            )
            compression_errors.append(compression_error)

            # Print out optimisation information
            itr_string = f"{0}/{self.max_coreset_iterations}"
            statement = (
                f"Coreset Iteration = {itr_string:<12} | "
                + f"Compression Error = {compression_errors[-1].item():<12.10f} | "
            )
            print(statement)

        for j in progress_bar(range(self.max_coreset_iterations)):
            z, coreset_state, z_grad = self._coreset_step(
                x_projected, z, compression_kernel, coreset_state
            )

            # Store parameter info
            zs.append(z)

            if self.track_info:
                # Store gradient
                z_grads.append(z_grad)

                # Store initial loss values
                compression_error = _compression_maximum_mean_discrepancy(
                    x_projected,
                    z,
                    compression_kernel,
                )
                compression_errors.append(compression_error)

                # Print out optimisation information
                itr_string = f"{j + 1}/{self.max_coreset_iterations}"
                statement = (
                    f"Coreset Iteration = {itr_string:<12} | "
                    + "Compression Error "
                    + f"= {compression_errors[-1].item():<12.10f} | "
                    + f"Z Gradient Norm = {jnp.linalg.norm(z_grad).item():<12.10f} | "
                )
                print(statement)

            # Check convergence
            gradient_norm = jnp.linalg.norm(z_grad).item()
            if gradient_norm < self.convergence_parameter:
                print("\nConverged!")
                break

        return Coreset(Data(z), dataset), BilateralDistributionCompressionState(
            [],
            zs,
            [],
            z_grads,
            compression_errors,
            reconstruction_errors,
            compression_kernel,
            self.reconstruction_kernel,
            autoencoder,
        )
