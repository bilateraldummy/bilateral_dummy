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

from abc import abstractmethod
from time import time
from typing import Callable, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax
from jax import grad, lax, vmap
from jax.tree_util import Partial
from jaxtyping import Array, Shaped
from optax import OptState
from tqdm import tqdm as LoudTQDM  # noqa: N812

from coreax.coreset import Coreset
from coreax.data import Data, SupervisedData
from coreax.kernels import ScalarValuedKernel, SquaredExponentialKernel
from coreax.solvers.base import CoresetSolver
from coreax.util import KeyArrayLike, SilentTQDM

# pylint:disable=too-many-positional-arguments
# pylint:disable=too-many-locals
# pylint:disable=duplicate-code
# pylint:disable=too-many-arguments
# pylint:disable=too-many-statements
# pylint:disable=too-many-branches


class KIPState(eqx.Module):
    """Optimisation results for :class:`_KIPSolver`."""

    losses: Optional[Array] = None
    deltas: Optional[Array] = None
    relative_deltas: Optional[Array] = None
    gradient_norms: Optional[Array] = None
    average_gradient_norms: Optional[Array] = None
    x_coresets: Optional[Array] = None
    y_coresets: Optional[Array] = None
    times: Optional[Array] = None


class KIPExhaustiveSearch(eqx.Module):
    """
    Exhaustive search of responses in KIP-style classification problems.

    :param classes: Array of possible classes.
    """

    classes: Shaped[Array, "C 1"]

    @staticmethod
    def _carry_update(
        carry: tuple,
        i: int,
        vmapped_loss_function: Callable[
            [
                Shaped[Array, " J"],
                Shaped[Array, "C 1"],
                Shaped[Array, "M 1"],
                Shaped[Array, "B 1"],
                list,
            ],
            Shaped[Array, " C 1"],
        ],
    ):
        """Carry utility function to compute optimal class, passing result onwards."""
        (
            classes,
            indices,
            y_coreset,
            y_target,
            invariant_terms,
        ) = carry

        # Compute the loss for each potential class at the ith index
        losses = vmapped_loss_function(
            indices[i],
            classes,
            y_coreset,
            y_target,
            invariant_terms,
        )

        # Choose the best class
        optimal_y = jnp.argmin(losses).astype(jnp.float64)

        # Replace the correct index in the subset of the coreset we are optimising,
        # and the overall coreset with the optimal class choice.
        y_coreset = y_coreset.at[indices[i]].set(optimal_y)

        return (
            classes,
            indices,
            y_coreset,
            y_target,
            invariant_terms,
        ), None

    def update(
        self,
        loss: Callable[
            [
                Shaped[Array, ""],
                Shaped[Array, "1 1"],
                Shaped[Array, "M 1"],
                Shaped[Array, "B 1"],
                list,
            ],
            Shaped[Array, ""],
        ],
        indices: Shaped[Array, " J"],
        y_coreset: Shaped[Array, "M p"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "J 1"]:
        """
        Optimise the responses point-by-point by exhaustive search.

        :param loss: Callable with type signature of
            :method:`_KIPSolver._classification_loss_function`
        :param indices: Array containing the indices of the coreset we are optimising.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :param invariant_terms: Terms which are constant with respect to the coreset,
            and can be precomputed and stored.
        """
        # Vmap the loss function across the choice of class
        vmapped_loss_function = vmap(loss, in_axes=(None, 0, None, None, None))

        # Define the initial carry
        carry = (
            self.classes,
            indices,
            y_coreset,
            y_target,
            invariant_terms,
        )

        # Run `lax.scan` over all indices
        final_carry, _ = lax.scan(
            f=Partial(self._carry_update, vmapped_loss_function=vmapped_loss_function),
            init=carry,
            xs=indices,
        )

        # Return the final y_coreset
        return final_carry[2]


class _UnsupervisedKIPSolver(CoresetSolver[Data, KIPState]):
    r"""
    Generic class for solving unlabelled KIP-class problems via gradient descent.

    .. warning::

        This class is only suitable for use with unlabelled data.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3. Input of :data:`None` corresponds to no
        optimisation.
    :param target_size: An integer representing the number of data pairs to use for
        estimation of expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    kernel: ScalarValuedKernel
    target_sample_size: Optional[int] = None
    coreset_sample_size: Optional[int] = None
    num_seeds: Optional[int] = None
    max_iterations: int = 100
    convergence_parameter: float = 1e-3
    optimiser: Optional[optax.GradientTransformation] = optax.sgd(
        optax.constant_schedule(1e-3)
    )
    track_info: bool = False

    @abstractmethod
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        points: Shaped[Array, "J d"],
        coreset: Shaped[Array, "M d"],
        target: Shaped[Array, "B d"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """
        Loss function that the solver targets.

        :param indices: Array containing the indices of the coreset we are optimising.
        :param points: A two-dimensional array containing the coreset points under
            current optimisation.
        :param coreset: A two-dimensional array containing the current coreset.
        :param target: A two-dimensional array used to estimate the loss function.
        :invariant_terms: Terms which are constant with respect to the coreset, and
            can be precomputed and stored.
        """

    @eqx.filter_jit
    def _step(
        self,
        indices: Shaped[Array, " J"],
        points: Shaped[Array, "J d"],
        coreset: Shaped[Array, "M d"],
        target: Shaped[Array, "B d"],
        opt_state: Optional[OptState],
        invariant_terms: Optional[list],
    ) -> tuple[
        Shaped[Array, "M d"],
        Shaped[Array, "M d"],
        Optional[OptState],
    ]:
        """
        Do gradient descent step.

        :param indices: Array containing the indices of the coreset we are optimising.
        :param points: A two-dimensional array containing the coreset points under
            current optimisation.
        :param coreset: A two-dimensional array containing the current coreset.
        :param target: A two-dimensional array of features used to estimate the loss
            function.
        :opt_state: Current state of the :class:`~optax.GradientTransformation`
            optimiser.
        :invariant_terms: Terms which are constant with respect to the coreset, and
            can be precomputed and stored.
        """
        # Do feature step according to chosen solver
        if (
            isinstance(self.optimiser, optax.GradientTransformation)
            and opt_state is not None
        ):
            points_grad = grad(self._loss_function, argnums=1)(
                indices,
                points,
                coreset,
                target,
                invariant_terms,
            )
            update, opt_state = self.optimiser.update(
                updates=points_grad, state=opt_state, params=points
            )
            coreset_ = coreset.at[indices].set(optax.apply_updates(points, update))
        else:
            # Default is to do nothing
            coreset_, points_grad = (
                coreset,
                jnp.zeros((indices.shape[0], coreset.shape[1])),
            )

        return coreset_, points_grad, opt_state

    def reduce(  # noqa:PLR0914,PLR0915, C901, PLR0912
        self, dataset: Data, solver_state: Optional[KIPState] = None
    ) -> tuple[Coreset[Data], KIPState]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The data to generate the coreset from.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information.
        :return: a tuple of the solved coreset and intermediate solver state information
        """
        # Extract data from containers
        data, dataset_size = dataset.data, len(dataset)

        # If we are using the entire dataset to estimate the loss function then we
        # can estimate certain terms of the loss function once using the entire training
        # data. This will be not be used if we are subsampling.
        target_sample_indices = None
        if self.target_sample_size is not None:
            target_sample_indices = jr.choice(
                self.random_key,
                dataset_size,
                shape=(self.target_sample_size,),
                replace=False,
            )

        print(
            "Initialising with random subset... \n"
            if self.track_info and self.num_seeds is None
            else "Choosing best initial random subset... \n"
            if self.track_info and self.num_seeds is not None
            else "",
            end="",
        )
        initial_loss = None
        seed_timer = time()
        if self.num_seeds is None:
            # Initialise the coreset with a random subset
            initialisation_indices = jr.choice(
                self.random_key, dataset_size, shape=(self.coreset_size,), replace=False
            )
            # Estimate the loss of the randomly chosen initial coreset
            if self.track_info:
                initial_loss = self._loss_function(
                    indices=jnp.arange(self.coreset_size),
                    points=data[initialisation_indices],
                    coreset=data[initialisation_indices],
                    target=data[target_sample_indices]
                    if target_sample_indices is not None
                    else data,
                    invariant_terms=None,
                )
        else:
            # Get keys to choose seeds for optimisation
            seed_keys = jr.split(self.random_key, num=(self.num_seeds,))

            # Sample sets of indices to check
            seed_indices = vmap(
                lambda key: jr.choice(
                    key, dataset_size, shape=(self.coreset_size,), replace=False
                ),
                in_axes=0,
            )(seed_keys)

            # Extract all possible initial coresets as a 3d array
            seed_coresets = data[seed_indices]

            # Compute the loss for each initial coreset
            seed_losses = vmap(
                self._loss_function,
                in_axes=(None, 0, None, None, None),
            )(
                jnp.arange(self.coreset_size),
                seed_coresets,
                jnp.zeros((self.coreset_size, data.shape[1])),
                data[target_sample_indices]
                if target_sample_indices is not None
                else data,
                None,
            )

            # Choose the best coreset and store the loss
            initialisation_indices = seed_indices[jnp.argmin(seed_losses)]
            initial_loss = jnp.nanmin(seed_losses)

        seed_timer = time() - seed_timer

        # Initialise coreset
        coreset = data[initialisation_indices]

        (
            losses,
            deltas,
            relative_deltas,
            gradient_norms,
            average_gradient_norms,
            coreset_delta_norms,
            coresets,
            times,
        ) = [jnp.array([])] * 8
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM

            # Initialise arrays to store optimisation info
            (
                losses,
                deltas,
                relative_deltas,
                gradient_norms,
                average_gradient_norms,
                coreset_delta_norms,
                coresets,
                times,
            ) = (
                jnp.zeros(self.max_iterations + 1),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros((self.max_iterations + 1, self.coreset_size, data.shape[1])),
                jnp.zeros(self.max_iterations + 1),
            )

            # Store the time it took to generate the initial coreset
            times = times.at[0].set(seed_timer)

            # Store the initial coreset
            coresets = coresets.at[0, :].set(coreset)

            # Store the initial loss
            losses = losses.at[0].set(initial_loss)
        else:
            progress_bar = LoudTQDM

        # Generate some keys for sampling subsets as we optimise
        subset_keys = jr.split(self.random_key, num=(self.max_iterations, 2))

        # If we are optimising every coreset point simultaneously then we only
        # need to initialise our optax optimisers, and our coreset subset once.
        # This will be overwritten if we are subsampling.
        coreset_sample_indices, opt_state = jnp.arange(self.coreset_size), None
        if isinstance(self.optimiser, optax.GradientTransformation):
            opt_state = self.optimiser.init(coreset)

        i = 0
        for i in progress_bar(range(self.max_iterations)):
            if self.coreset_sample_size is not None:
                # Sample subset of the coreset we wish to optimise for this iteration
                coreset_sample_indices = jr.choice(
                    subset_keys[i, 0],
                    self.coreset_size,
                    shape=(self.coreset_sample_size,),
                    replace=False,
                )
                # Reinitialise the optimiser as the subset has changed
                if isinstance(self.optimiser, optax.GradientTransformation):
                    opt_state = self.optimiser.init(coreset[coreset_sample_indices])

            if self.target_sample_size is not None:
                # Sample the subset of the data we wish to use to estimate the loss
                target_sample_indices = jr.choice(
                    subset_keys[i, 1],
                    dataset_size,
                    shape=(self.target_sample_size,),
                    replace=False,
                )

            # Copy current coreset for tracking purposes
            old_coreset = None
            if self.track_info:
                old_coreset = jnp.copy(coreset)

            # Do a gradient step
            step_timer = time()
            coreset, grads, opt_state = self._step(
                indices=coreset_sample_indices,
                points=coreset[coreset_sample_indices],
                coreset=coreset,
                target=data[target_sample_indices]
                if target_sample_indices is not None
                else data,
                opt_state=opt_state,
                invariant_terms=None,
            )
            step_timer = time() - step_timer

            # Compute norm of the gradient for convergence purposes
            gradient_norm = jnp.linalg.norm(grads)

            if self.track_info:
                # Store current coreset
                coresets = coresets.at[i + 1, :].set(coreset)

                # Store timings
                times = times.at[i + 1].set(step_timer)

                # Estimate the loss of the current coreset
                losses = losses.at[i + 1].set(
                    self._loss_function(
                        indices=jnp.arange(self.coreset_size),
                        points=coreset,
                        coreset=coreset,
                        target=data[target_sample_indices]
                        if target_sample_indices is not None
                        else data,
                        invariant_terms=None,
                    )
                )

                # Compute change in estimated loss
                delta = losses[i + 1] - losses[i]
                deltas = deltas.at[i].set(delta)

                # Compute an absolute relative measure of change in estimated loss
                relative_delta = jnp.abs(delta) / jnp.abs(losses[i])
                relative_deltas = relative_deltas.at[i].set(relative_delta)

                # Compute the norm of the gradient wrt the entire subset of the coreset
                # we  have optimised.

                gradient_norms = gradient_norms.at[i].set(gradient_norm)

                # Compute the average norm of the gradient wrt each individual coreset
                # pair.
                average_gradient_norm = jnp.mean(jnp.linalg.norm(grads, axis=1))
                average_gradient_norms = average_gradient_norms.at[i].set(
                    average_gradient_norm
                )

                # Compute the change in position of the coreset
                coreset_delta_norm = jnp.linalg.norm(coreset - old_coreset)
                coreset_delta_norms = coreset_delta_norms.at[i].set(coreset_delta_norm)

                # Print out optimisation information
                itr_string = f"{i + 1}/{self.max_iterations}"
                statement = (
                    f"Iteration = {itr_string:<12} | "
                    + f"Loss = {losses[i + 1].item():<12.10f} | "
                    + f"Delta = {delta.item():<12.10f} | "
                    + f"Relative Delta = {relative_delta.item():<12.10f} | "
                    + f"Gradient Norm = {gradient_norm.item():<12.10f} | "
                    + f"Avg Gradient Norm = {average_gradient_norm.item():<12.10f} | "
                    + f"Coreset Delta Norm = {coreset_delta_norms[i].item():<12.10f}"
                    + "                                                  "
                )
                print(statement)

            # Check convergence
            if gradient_norm < self.convergence_parameter:
                print("\nConverged!")
                break

        if self.track_info:
            state = KIPState(
                losses[: i + 2],
                deltas[: i + 1],
                relative_deltas[: i + 1],
                gradient_norms[: i + 1],
                average_gradient_norms[: i + 1],
                coresets[: i + 2],
                None,
                times[: i + 2],
            )
        else:
            state = KIPState()

        return (Coreset(Data(coreset), dataset), state)


class KernelInducingPoints(_UnsupervisedKIPSolver):
    r"""
    Kernel Inducing Points - a batch stochastic gradient descent coreset solver.

    Kernel Inducing Points is a batch stochastic gradient descent algorithm which
    learns a coreset by targeting the Maximum Mean Discrepancy (MMD)
    between the true distribution, and the distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        points: Shaped[Array, "J d"],
        coreset: Shaped[Array, "M d"],
        target: Shaped[Array, "B d"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Joint Kernel Inducing Points loss function."""
        del invariant_terms

        # Update the current coreset with the xs and ys under consideration
        coreset = coreset.at[indices].set(points)

        # Compute feature gramians
        coreset_gramian = self.kernel.compute(coreset, coreset)
        cross_gramian = self.kernel.compute(coreset, target)

        # Compute the non-invariant terms of the JMMD
        term_1, term_2 = coreset_gramian.mean(), cross_gramian.mean()

        return term_1 - 2 * term_2


class _KIPSolver(CoresetSolver[SupervisedData, KIPState]):
    r"""
    Generic class for solving KIP-class problems via gradient descent.

    .. warning::

        This class is only suitable for use with supervised data.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3. Input of :data:`None` corresponds to no
        optimisation.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3. Input of :data:`None` corresponds to no
        optimisation.
    :param target_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    response_kernel: ScalarValuedKernel
    target_sample_size: Optional[int] = None
    coreset_sample_size: Optional[int] = None
    num_seeds: Optional[int] = None
    max_iterations: int = 100
    convergence_parameter: float = 1e-3
    feature_optimiser: Optional[optax.GradientTransformation] = optax.sgd(
        optax.constant_schedule(1e-3)
    )
    response_optimiser: Optional[
        Union[optax.GradientTransformation, KIPExhaustiveSearch]
    ] = optax.sgd(optax.constant_schedule(1e-3))
    track_info: bool = False

    @abstractmethod
    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    @abstractmethod
    def _set_up_classification_loss_function(
        self,
        x_coreset: Shaped[Array, "M d"],
        x_target: Shaped[Array, "B d"],
    ) -> Optional[list]:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        """

    @abstractmethod
    def _classification_loss_function(
        self,
        index: Shaped[Array, ""],
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_target: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`KIPExhaustiveSearch` solver targets.

        :param index: Scalar array with the index of the coreset we are optimising.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """

    @abstractmethod
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """
        Loss function that the solver targets.

        :param indices: Array containing the indices of the coreset we are optimising.
        :param xs: A two-dimensional array containing the coreset features under
            consideration.
        :param ys: A two-dimensional array containing the coreset supervisions under
            consideration.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the coreset, and
            can be precomputed and stored.
        """

    @eqx.filter_jit
    def _step(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        feature_opt_state: Optional[OptState],
        response_opt_state: Optional[OptState],
        invariant_terms: Optional[list],
    ) -> tuple[
        Shaped[Array, "M d"],
        Shaped[Array, "M p"],
        Shaped[Array, "M d+p"],
        Optional[OptState],
        Optional[OptState],
    ]:
        """
        Do gradient descent step.

        :param indices: Array containing the indices of the coreset we are optimising.
        :param xs: A two-dimensional array containing the coreset features under
            consideration.
        :param ys: A two-dimensional array containing the coreset supervisions under
            consideration.
        :response_opt_state: Current state of the :class:`~optax.GradientTransformation`
            feature optimiser.
        :feature_opt_state: Current state of the :class:`~optax.GradientTransformation`
            supervision optimiser.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        """
        # Do feature step according to chosen solver
        if (
            isinstance(self.feature_optimiser, optax.GradientTransformation)
            and feature_opt_state is not None
        ):
            xs_grad = grad(self._loss_function, argnums=1)(
                indices,
                xs,
                ys,
                x_coreset,
                y_coreset,
                x_target,
                y_target,
                invariant_terms,
            )
            xs_update, feature_opt_state = self.feature_optimiser.update(
                updates=xs_grad, state=feature_opt_state, params=xs
            )
            x_coreset_ = x_coreset.at[indices].set(optax.apply_updates(xs, xs_update))
        else:
            # Default is to do nothing
            x_coreset_, xs_grad = (
                x_coreset,
                jnp.zeros((indices.shape[0], x_coreset.shape[1])),
            )

        # Do response step according to chosen solver
        if (
            isinstance(self.response_optimiser, optax.GradientTransformation)
            and response_opt_state is not None
        ):
            ys_grad = grad(self._loss_function, argnums=2)(
                indices,
                xs,
                ys,
                x_coreset,
                y_coreset,
                x_target,
                y_target,
                invariant_terms,
            )
            ys_update, response_opt_state = self.response_optimiser.update(
                updates=ys_grad, state=response_opt_state, params=ys
            )
            y_coreset_ = y_coreset.at[indices].set(optax.apply_updates(ys, ys_update))
        elif isinstance(self.response_optimiser, KIPExhaustiveSearch):
            # Compute the terms that are invariant when doing exhaustive search
            classification_invariant_terms = self._set_up_classification_loss_function(
                x_coreset_,  # Make sure we use the updated coreset
                x_target,
            )

            # Update the responses by exhaustive search
            y_coreset_ = self.response_optimiser.update(
                loss=self._classification_loss_function,
                indices=indices,
                y_coreset=y_coreset,
                y_target=y_target,
                invariant_terms=classification_invariant_terms,
            )
            ys_grad = jnp.zeros((indices.shape[0], 1))
        else:
            # Default is to do nothing
            y_coreset_, ys_grad = (
                y_coreset,
                jnp.zeros((indices.shape[0], y_coreset.shape[1])),
            )

        return (
            x_coreset_,
            y_coreset_,
            jnp.hstack((xs_grad, ys_grad)),
            feature_opt_state,
            response_opt_state,
        )

    def reduce(  # noqa:PLR0914,PLR0915, C901, PLR0912
        self, dataset: SupervisedData, solver_state: Optional[KIPState] = None
    ) -> tuple[Coreset[SupervisedData], KIPState]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The (potentially weighted and supervised) data to generate the
            coreset from
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information
        :return: a tuple of the solved coreset and intermediate solver state information
        """
        # Extract data from containers
        data, supervision = dataset.data, dataset.supervision
        dataset_size = len(dataset)

        # If we are using the entire dataset to estimate the loss function then we
        # can estimate certain terms of the loss function once using the entire training
        # data. This will be not be used if we are subsampling.
        target_sample_indices = None
        if self.target_sample_size is None:
            invariant_timer = time()
            invariant_terms = self._set_up_loss_function(data, supervision)
            invariant_timer = time() - invariant_timer
        else:
            invariant_timer, invariant_terms = 0, None
            target_sample_indices = jr.choice(
                self.random_key,
                dataset_size,
                shape=(self.target_sample_size,),
                replace=False,
            )

        print(
            "Initialising with random subset... \n"
            if self.track_info and self.num_seeds is None
            else "Choosing best initial random subset... \n"
            if self.track_info and self.num_seeds is not None
            else "",
            end="",
        )
        initial_loss = None
        seed_timer = time()
        if self.num_seeds is None:
            # Initialise the coreset with a random subset
            initialisation_indices = jr.choice(
                self.random_key, dataset_size, shape=(self.coreset_size,), replace=False
            )
            # Estimate the loss of the randomly chosen initial coreset
            if self.track_info:
                initial_loss = self._loss_function(
                    indices=jnp.arange(self.coreset_size),
                    xs=data[initialisation_indices],
                    ys=supervision[initialisation_indices],
                    x_coreset=data[initialisation_indices],
                    y_coreset=supervision[initialisation_indices],
                    x_target=data[target_sample_indices]
                    if target_sample_indices is not None
                    else data,
                    y_target=supervision[target_sample_indices]
                    if target_sample_indices is not None
                    else supervision,
                    invariant_terms=invariant_terms,
                )
        else:
            # Get keys to choose seeds for optimisation
            seed_keys = jr.split(self.random_key, num=(self.num_seeds,))

            # Sample sets of indices to check
            seed_indices = vmap(
                lambda key: jr.choice(
                    key, dataset_size, shape=(self.coreset_size,), replace=False
                ),
                in_axes=0,
            )(seed_keys)

            # Extract all possible initial coresets as a 3d array
            seed_x_coresets = data[seed_indices]
            seed_y_coresets = supervision[seed_indices]

            # Compute the loss for each initial coreset
            seed_losses = vmap(
                self._loss_function,
                in_axes=(None, 0, 0, None, None, None, None, None),
            )(
                jnp.arange(self.coreset_size),
                seed_x_coresets,
                seed_y_coresets,
                jnp.zeros((self.coreset_size, data.shape[1])),
                jnp.zeros((self.coreset_size, supervision.shape[1])),
                data[target_sample_indices]
                if target_sample_indices is not None
                else data,
                supervision[target_sample_indices]
                if target_sample_indices is not None
                else supervision,
                invariant_terms,
            )

            # Choose the best coreset and store the loss
            initialisation_indices = seed_indices[jnp.argmin(seed_losses)]
            initial_loss = jnp.nanmin(seed_losses)

        seed_timer = time() - seed_timer

        # Initialise coreset
        x_coreset = data[initialisation_indices]
        y_coreset = supervision[initialisation_indices]

        (
            losses,
            deltas,
            relative_deltas,
            gradient_norms,
            average_gradient_norms,
            coreset_delta_norms,
            x_coresets,
            y_coresets,
            times,
        ) = [jnp.array([])] * 9
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM

            # Initialise arrays to store optimisation info
            (
                losses,
                deltas,
                relative_deltas,
                gradient_norms,
                average_gradient_norms,
                coreset_delta_norms,
                x_coresets,
                y_coresets,
                times,
            ) = (
                jnp.zeros(self.max_iterations + 1),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros(self.max_iterations),
                jnp.zeros((self.max_iterations + 1, self.coreset_size, data.shape[1])),
                jnp.zeros(
                    (self.max_iterations + 1, self.coreset_size, supervision.shape[1])
                ),
                jnp.zeros(self.max_iterations + 1),
            )

            # Store the time it took to generate the initial coreset
            times = times.at[0].set(seed_timer + invariant_timer)

            # Store the initial coreset
            x_coresets = x_coresets.at[0, :].set(x_coreset)
            y_coresets = y_coresets.at[0, :].set(y_coreset)

            # Store the initial loss
            losses = losses.at[0].set(initial_loss)
        else:
            progress_bar = LoudTQDM

        # Generate some keys for sampling subsets as we optimise
        subset_keys = jr.split(self.random_key, num=(self.max_iterations, 2))

        # If we are optimising every coreset point simultaneously then we only
        # need to initialise our optax optimisers, and our coreset subset once.
        # This will be overwritten if we are subsampling.
        coreset_sample_indices = jnp.arange(self.coreset_size)
        feature_opt_state, response_opt_state = None, None
        if isinstance(self.feature_optimiser, optax.GradientTransformation):
            feature_opt_state = self.feature_optimiser.init(x_coreset)
        if isinstance(self.response_optimiser, optax.GradientTransformation):
            response_opt_state = self.response_optimiser.init(y_coreset)

        i = 0
        for i in progress_bar(range(self.max_iterations)):
            if self.coreset_sample_size is not None:
                # Sample subset of the coreset we wish to optimise for this iteration
                coreset_sample_indices = jr.choice(
                    subset_keys[i, 0],
                    self.coreset_size,
                    shape=(self.coreset_sample_size,),
                    replace=False,
                )
                # Reinitialise the optimiser as the subset has changed
                if isinstance(self.feature_optimiser, optax.GradientTransformation):
                    feature_opt_state = self.feature_optimiser.init(
                        x_coreset[coreset_sample_indices]
                    )
                if isinstance(self.response_optimiser, optax.GradientTransformation):
                    response_opt_state = self.response_optimiser.init(
                        y_coreset[coreset_sample_indices]
                    )

            if self.target_sample_size is not None:
                # Sample the subset of the data we wish to use to estimate the loss
                target_sample_indices = jr.choice(
                    subset_keys[i, 1],
                    dataset_size,
                    shape=(self.target_sample_size,),
                    replace=False,
                )

            # Copy current coreset for tracking purposes
            old_x_coreset, old_y_coreset = None, None
            if self.track_info:
                old_x_coreset, old_y_coreset = jnp.copy(x_coreset), jnp.copy(y_coreset)

            # Do a gradient step
            step_timer = time()
            x_coreset, y_coreset, grads, feature_opt_state, response_opt_state = (
                self._step(
                    indices=coreset_sample_indices,
                    xs=x_coreset[coreset_sample_indices],
                    ys=y_coreset[coreset_sample_indices],
                    x_coreset=x_coreset,
                    y_coreset=y_coreset,
                    x_target=data[target_sample_indices]
                    if target_sample_indices is not None
                    else data,
                    y_target=supervision[target_sample_indices]
                    if target_sample_indices is not None
                    else supervision,
                    feature_opt_state=feature_opt_state,
                    response_opt_state=response_opt_state,
                    invariant_terms=invariant_terms,
                )
            )
            step_timer = time() - step_timer

            # Compute norm of the gradient for convergence purposes
            gradient_norm = jnp.linalg.norm(grads)

            if self.track_info:
                # Store current coreset
                x_coresets = x_coresets.at[i + 1, :].set(x_coreset)
                y_coresets = y_coresets.at[i + 1, :].set(y_coreset)

                # Store timings
                times = times.at[i + 1].set(step_timer)

                # Estimate the loss of the current coreset
                losses = losses.at[i + 1].set(
                    self._loss_function(
                        indices=jnp.arange(self.coreset_size),
                        xs=x_coreset,
                        ys=y_coreset,
                        x_coreset=x_coreset,
                        y_coreset=y_coreset,
                        x_target=data[target_sample_indices]
                        if target_sample_indices is not None
                        else data,
                        y_target=supervision[target_sample_indices]
                        if target_sample_indices is not None
                        else supervision,
                        invariant_terms=invariant_terms,
                    )
                )

                # Compute change in estimated loss
                delta = losses[i + 1] - losses[i]
                deltas = deltas.at[i].set(delta)

                # Compute an absolute relative measure of change in estimated loss
                relative_delta = jnp.abs(delta) / jnp.abs(losses[i])
                relative_deltas = relative_deltas.at[i].set(relative_delta)

                # Compute the norm of the gradient wrt the entire subset of the coreset
                # we  have optimised.

                gradient_norms = gradient_norms.at[i].set(gradient_norm)

                # Compute the average norm of the gradient wrt each individual coreset
                # pair.
                average_gradient_norm = jnp.mean(jnp.linalg.norm(grads, axis=1))
                average_gradient_norms = average_gradient_norms.at[i].set(
                    average_gradient_norm
                )

                # Compute the change in position of the coreset
                coreset_delta_norm = jnp.linalg.norm(
                    jnp.hstack((x_coreset, y_coreset))
                    - jnp.hstack((old_x_coreset, old_y_coreset))
                )
                coreset_delta_norms = coreset_delta_norms.at[i].set(coreset_delta_norm)

                # Print out optimisation information
                itr_string = f"{i + 1}/{self.max_iterations}"
                statement = (
                    f"Iteration = {itr_string:<12} | "
                    + f"Loss = {losses[i + 1].item():<12.10f} | "
                    + f"Delta = {delta.item():<12.10f} | "
                    + f"Relative Delta = {relative_delta.item():<12.10f} | "
                    + f"Gradient Norm = {gradient_norm.item():<12.10f} | "
                    + f"Avg Gradient Norm = {average_gradient_norm.item():<12.10f} | "
                    + f"Coreset Delta Norm = {coreset_delta_norms[i].item():<12.10f}"
                    + "                                                  "
                )
                print(statement)

            # Check convergence
            if gradient_norm < self.convergence_parameter:
                print("\nConverged!")
                break

        if self.track_info:
            state = KIPState(
                losses[: i + 2],
                deltas[: i + 1],
                relative_deltas[: i + 1],
                gradient_norms[: i + 1],
                average_gradient_norms[: i + 1],
                x_coresets[: i + 2],
                y_coresets[: i + 2],
                times[: i + 2],
            )
        else:
            state = KIPState()

        return (Coreset(SupervisedData(x_coreset, y_coreset), dataset), state)


class JointKIP(_KIPSolver):
    r"""
    Joint Kernel Herding - a batch stochastic gradient descent coreset solver.

    Joint Kernel Herding is a batch stochastic gradient descent algorithm which
    learns a coreset by targeting the Joint Maximum Mean Discrepancy (JMMD)
    between the true joint distribution, and the joint distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x_coreset: Shaped[Array, "M d"],
        x_target: Shaped[Array, "B d"],
    ) -> Optional[list]:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        """
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        cross_feature_gramian = self.feature_kernel.compute(x_target, x_coreset)
        return [coreset_feature_gramian, cross_feature_gramian]

    def _classification_loss_function(
        self,
        index: Shaped[Array, ""],
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_target: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`KIPExhaustiveSearch` solver targets.

        :param index: Scalar array with the index of the coreset we are optimising.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        # Ensure the index is a scalar array
        index = jnp.squeeze(index)

        # Update the current coreset with the y under consideration
        y_coreset = y_coreset.at[index].set(y)
        coreset_size = y_coreset.shape[0]
        target_size = y_target.shape[0]

        # Extract the invariant terms
        coreset_feature_gramian, cross_feature_gramian = invariant_terms

        # Compute the non-invariant terms
        coreset_response_vector = self.response_kernel.compute(
            y_coreset[index],
            y_coreset,
        )
        full_response_vector = self.response_kernel.compute(
            y_coreset[index],
            y_target,
        )

        # Compute loss function
        term_1 = coreset_response_vector.dot(coreset_feature_gramian[:, index])
        term_2 = full_response_vector.dot(cross_feature_gramian[:, index])

        return jnp.squeeze(
            (2 / coreset_size**2) * term_1 - (2 / (coreset_size * target_size)) * term_2
        )

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Joint Kernel Inducing Points loss function."""
        del invariant_terms

        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)

        # Compute feature gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        cross_feature_gramian = self.feature_kernel.compute(x_coreset, x_target)

        # Compute response gramians
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)
        cross_response_gramian = self.response_kernel.compute(y_coreset, y_target)

        # Compute the non-invariant terms of the JMMD
        term_1 = (coreset_feature_gramian * coreset_response_gramian).mean()
        term_2 = (cross_feature_gramian * cross_response_gramian).mean()

        return term_1 - 2 * term_2


class ExactJointKIP(_KIPSolver):
    r"""
    Exact Joint Kernel Herding - a batch stochastic gradient descent coreset solver.

    Joint Kernel Herding is a batch stochastic gradient descent algorithm which
    learns a coreset by targeting the Joint Maximum Mean Discrepancy (JMMD)
    between the true joint distribution, and the joint distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    bias: float = 0.0
    slope: float = 1.0
    feature_mean: float = 0.0
    feature_standard_deviation: float = 1
    response_standard_deviation: float = 0.25

    @eqx.filter_jit
    def _compute_joint_expectation(
        self, x: Shaped[Array, " 1 d"], y: Shaped[Array, " 1 p"]
    ):
        r"""Compute :math:`E[k(X, X)l(Y, y)]` for :math:`l, k` RBF kernels."""
        if not isinstance(
            self.response_kernel, SquaredExponentialKernel
        ) or not isinstance(self.feature_kernel, SquaredExponentialKernel):
            raise ValueError(
                "The feature and response kernels must be instances of the"
                + " SquaredExponentialKernel class."
            )
        # Only valid for 1-dimensional features and responses
        x = jnp.squeeze(x)
        y = jnp.squeeze(y)

        # Rename variables to improve formatting
        x_length_scale, y_length_scale = (
            self.feature_kernel.length_scale,
            self.response_kernel.length_scale,
        )
        x_sd, y_sd = self.feature_standard_deviation, self.response_standard_deviation
        x_mu, a_0, a_1 = self.feature_mean, self.bias, self.slope

        # Compute terms relating to k(x, X)l(y, Y)
        constant_matrix_1 = jnp.array(
            [
                [1 / x_length_scale**2, 0],
                [0, 1 / y_length_scale**2],
            ]
        )
        constant_vector_1 = jnp.array(
            [
                [x / x_length_scale**2],
                [y / y_length_scale**2],
            ]
        )
        constant_scalar_1 = -(x**2 / (2 * x_length_scale**2)) - (
            y**2 / (2 * y_length_scale**2)
        )

        # Compute terms relating to f_{X, Y}(x, y)
        constant_matrix_2 = jnp.array(
            [
                [
                    1 / x_sd**2 + (a_1**2 / y_sd**2),
                    -(a_1 / y_sd**2),
                ],
                [
                    -(a_1 / y_sd**2),
                    1 / y_sd**2,
                ],
            ]
        )
        constant_vector_2 = jnp.array(
            [
                [x_mu / x_sd**2 - (a_0 * a_1) / y_sd**2],
                [a_0 / y_sd**2],
            ]
        )
        constant_scalar_2 = -(a_0**2) / (2 * y_sd**2) - x_mu**2 / (2 * x_sd**2)

        # Compute combined terms
        constant_matrix, constant_vector, constant_scalar = (
            constant_matrix_1 + constant_matrix_2,
            constant_vector_1 + constant_vector_2,
            constant_scalar_1 + constant_scalar_2,
        )

        # Compute integral
        return jnp.squeeze(
            (1 / jnp.sqrt(x_sd**2 * y_sd**2 * jnp.linalg.det(constant_matrix)))
            * jnp.exp(
                constant_scalar
                + (1 / 2)
                * constant_vector.T.dot(jnp.linalg.inv(constant_matrix)).dot(
                    constant_vector
                )
            )
        )

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x_coreset: Shaped[Array, "M d"],
        x_target: Shaped[Array, "B d"],
    ) -> Optional[list]:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        """

    def _classification_loss_function(
        self,
        index: Shaped[Array, ""],
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_target: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`KIPExhaustiveSearch` solver targets.

        :param index: Scalar array with the index of the coreset we are optimising.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        return jnp.array([None])

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function."""
        # Delete unused invariant terms
        del invariant_terms, x_target, y_target

        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)

        # Compute the coreset kernel gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)

        # Compute the non-invariant terms of the JMMD
        term_1 = (coreset_feature_gramian * coreset_response_gramian).mean()
        term_2 = vmap(self._compute_joint_expectation, in_axes=(0, 0))(
            x_coreset, y_coreset
        ).mean()

        return term_1 - 2 * term_2


class ConditionalKIP(_KIPSolver):
    r"""
    Conditional Kernel Herding - a target stochastic gradient descent coreset solver.

    Conditional Kernel Herding is a target stochastic gradient descent algorithm which
    learns a coreset by targeting the Conditional Maximum Mean Discrepancy (CMMD)
    between the true conditional distribution, and the conditional
    distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param target_regularisation_parameter: Regularisation parameter for estimation
        of the KCME with the target data.
    :param coreset_regularisation_parameter: Regularisation parameter for estimation
        of the KCME with the coreset.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    target_regularisation_parameter: float = 1e-3
    coreset_regularisation_parameter: float = 1e-3

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """
        # Delete unused terms
        del supervision

        # Compute the target feature gramian
        target_feature_gramian = self.feature_kernel.compute(data, data)

        # Regularise the feature gramian ready for inversion
        identity = jnp.eye(data.shape[0])
        regularised_target_feature_gramian = (
            target_feature_gramian + self.target_regularisation_parameter * identity
        )

        # Compute the LU factorisation of the regularised feature gramian
        lu_factor = jsp.linalg.lu_factor(regularised_target_feature_gramian)
        return [lu_factor]

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function."""
        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)
        coreset_size = x_coreset.shape[0]
        target_size = x_target.shape[0]

        # Compute the coreset kernel gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)

        # Regularise the feature gramian ready for inversion
        regularised_coreset_feature_gramian = (
            coreset_feature_gramian
            + self.coreset_regularisation_parameter * jnp.eye(coreset_size)
        )

        # Compute the LU factorisation of the regularised coreset feature gramian
        coreset_lu_factor = jsp.linalg.lu_factor(regularised_coreset_feature_gramian)

        # Compute the cross-gramians between the coreset and the targets
        cross_feature_gramian = self.feature_kernel.compute(x_coreset, x_target)
        cross_response_gramian = self.response_kernel.compute(y_target, y_coreset)

        # First term is equivalent to Tr(K_tilde @ W_tilde @ L_tilde @ W_tilde)
        term_1 = (
            jsp.linalg.lu_solve(
                lu_and_piv=coreset_lu_factor,
                b=coreset_feature_gramian,
            )
            * jsp.linalg.lu_solve(
                lu_and_piv=coreset_lu_factor,
                b=coreset_response_gramian,
            )
        ).sum()

        # Check if we have precomputed the target feature gramian
        if invariant_terms is not None:
            target_lu_factor = invariant_terms[0]
        else:
            # Compute the target feature gramian
            target_feature_gramian = self.feature_kernel.compute(x_target, x_target)

            # Regularise the feature gramian ready for inversion
            regularised_target_feature_gramian = (
                target_feature_gramian
                + self.target_regularisation_parameter * jnp.eye(target_size)
            )

            target_lu_factor = jsp.linalg.lu_factor(regularised_target_feature_gramian)

        # Evaluate the target-trained KCME on the coreset
        target_kcme_evaluations = jsp.linalg.lu_solve(
            lu_and_piv=target_lu_factor,
            b=cross_response_gramian,
        )

        # Second term is equivalent to Tr(K_bar @ W_tilde @ L_bar^T @ W_target)
        term_2 = (
            jsp.linalg.lu_solve(
                lu_and_piv=coreset_lu_factor,
                b=cross_feature_gramian,
            )
            * target_kcme_evaluations.T  # m x b
        ).sum()
        return term_1 - 2 * term_2


class ExactConditionalKIP(_KIPSolver):
    r"""
    Conditional Kernel Herding - a target stochastic gradient descent coreset solver.

    Conditional Kernel Herding is a target stochastic gradient descent algorithm which
    learns a coreset by targeting the Conditional Maximum Mean Discrepancy (CMMD)
    between the true conditional distribution, and the conditional
    distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param coreset_regularisation_parameter: Regularisation parameter for estimation
        of the KCME with the coreset.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_regularisation_parameter: float = 1e-3
    generating_function: Callable = lambda x: x
    standard_deviation: float = 0.25

    @eqx.filter_jit
    def _compute_conditional_expectation(
        self, x: Shaped[Array, " 1 d"], y: Shaped[Array, " 1 p"]
    ):
        r"""Compute :math:`E[l(Y, y)|X = x]` for :math:`l` the RBF kernel."""
        if not isinstance(self.response_kernel, SquaredExponentialKernel):
            raise ValueError(
                "The response kernel must be an instance of the"
                + " SquaredExponentialKernel class."
            )

        # Rename variables for nicer formatting
        f_val = self.generating_function(x)
        sd, length_scale = self.standard_deviation, self.response_kernel.length_scale
        constant_1 = (1 / length_scale**2) + (1 / sd**2)
        constant_2 = (y / length_scale**2) + (f_val / sd**2)

        return jnp.sqrt(1 / (constant_1 * sd**2)) * jnp.exp(
            (constant_1 / 2) * ((constant_2 / constant_1) ** 2)
            - y**2 / (2 * length_scale**2)
            - f_val**2 / (2 * sd**2)
        )

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """
        # Delete unused terms
        del data, supervision

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function."""
        # Delete unused terms
        del x_target, y_target, invariant_terms

        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)
        coreset_size = x_coreset.shape[0]

        # Compute the kernel gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)

        # Regularise the feature gramian ready for inversion
        regularised_coreset_feature_gramian = (
            coreset_feature_gramian
            + self.coreset_regularisation_parameter * jnp.eye(coreset_size)
        )

        # First term is equivalent to Tr(K_tilde @ W_tilde @ L_tilde @ W_tilde)
        term_1 = (
            jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=coreset_feature_gramian,  # m x m,
                assume_a="sym",
            )
            * jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=coreset_response_gramian,  # m x m,
                assume_a="sym",
            )
        ).sum()

        # Compute the exact conditional expectations for each possible coreset pair
        vmapped_conditional_expectation = vmap(
            vmap(
                self._compute_conditional_expectation,
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )
        conditional_expectations = jnp.squeeze(
            vmapped_conditional_expectation(x_coreset, y_coreset),
            axis=-1,
        )

        # Second term is equivalent to \langle true_KCME, coreset_KCME \rangle
        term_2 = jnp.trace(
            jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=conditional_expectations,  # m x m,
                assume_a="sym",
            ),
        )

        return term_1 - 2 * term_2


class AverageConditionalKIP(_KIPSolver):
    r"""
    An implementation of Average Conditional Kernel Herding.

    Average Conditional Kernel Herding is a batch stochastic gradient descent algorithm
    which learns a coreset by targeting the Average Maximum Conditional Mean Discrepancy
    (AMCMD) between the true conditional distribution, and the conditional
    distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param regularisation_parameter: Regularisation parameter for estimation
        of the KCME with the coreset.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    regularisation_parameter: float = 1e-3

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x_coreset: Shaped[Array, "M d"],
        x_target: Shaped[Array, "B d"],
    ) -> Optional[list]:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        """
        coreset_size = x_coreset.shape[0]
        common_term = jsp.linalg.solve(
            a=self.feature_kernel.compute(x_coreset, x_coreset)
            + self.regularisation_parameter * jnp.eye(coreset_size),  # m x m
            b=self.feature_kernel.compute(x_coreset, x_target),
            assume_a="sym",
        )
        return [common_term.dot(common_term.T), common_term.T]

    def _classification_loss_function(
        self,
        index: Shaped[Array, ""],
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_target: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`KIPExhaustiveSearch` solver targets.

        :param index: Scalar array with the index of the coreset we are optimising.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        # Ensure the index is a scalar array
        index = jnp.squeeze(index)

        # Update the current coreset with the xs and ys under consideration
        y_coreset = y_coreset.at[index].set(y)
        target_size = y_target.shape[0]

        # Compute the non-invariant terms
        coreset_response_vector = self.response_kernel.compute(
            y_coreset[index],
            y_coreset,
        )
        full_response_vector = self.response_kernel.compute(
            y_coreset[index],
            y_target,
        )

        # Compute loss function
        term_1 = coreset_response_vector.dot(invariant_terms[0][:, index])
        term_2 = full_response_vector.dot(invariant_terms[1][:, index])

        return jnp.squeeze(2 / target_size * (term_1 - term_2))

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Average Conditional Kernel Herding loss function."""
        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)
        coreset_size = x_coreset.shape[0]
        target_size = x_target.shape[0]

        if invariant_terms is not None:
            common_term = invariant_terms[0]
        else:
            # The first and second terms of the loss function share K_bar @ W_tilde:
            common_term = jsp.linalg.solve(
                a=self.feature_kernel.compute(x_coreset, x_coreset)
                + self.regularisation_parameter * jnp.eye(coreset_size),  # m x m
                b=self.feature_kernel.compute(x_coreset, x_target),
                assume_a="sym",
            )

        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)
        cross_response_gramian = self.response_kernel.compute(y_coreset, y_target)

        # First term is equivalent to Tr(K_bar @ W_tilde @ L_tilde @ W_tilde @ K_bar^T)
        term_1 = (coreset_response_gramian.dot(common_term) * common_term).sum()

        # Second term is equivalent to Tr(K_bar @ W_tilde @ L_bar^T)
        term_2 = (cross_response_gramian * common_term).sum()

        return (term_1 - 2 * term_2) / target_size


class ExactAverageConditionalKIP(_KIPSolver):
    r"""
    An implementation of Average Conditional Kernel Herding.

    Average Conditional Kernel Herding is a batch stochastic gradient descent algorithm
    which learns a coreset by targeting the Average Maximum Mean Discrepancy (AMCMD)
    between the true conditional distribution, and the conditional
    distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` on the
        response space
    :param regularisation_parameter: Regularisation parameter for estimation
        of the KCME with the coreset.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param target_sample_size: Number of data pairs we sample at each iteration to
        estimate the gradients. Defaults to :data:`None`, indicating the entire dataset
        is used.
    :param coreset_sample_size: Number of coreset pairs we sample at each iteration for
        optimisation. Defaults to :data:`None`, indicating every coreset pair is
        optimised.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    regularisation_parameter: float = 1e-3
    bias: float = 0.0
    slope: float = 1.0
    feature_mean: float = 0.0
    feature_standard_deviation: float = 1
    response_standard_deviation: float = 0.25

    @eqx.filter_jit
    def _compute_marginal_expectation(
        self, x_1: Shaped[Array, " 1 d"], x_2: Shaped[Array, " 1 p"]
    ):
        r"""Compute :math:`E[k(X, x_1)k(X, x_2)]` for :math:`k` the RBF kernel."""
        if not isinstance(self.feature_kernel, SquaredExponentialKernel):
            raise ValueError(
                "The feature kernel must be an instance of the"
                + " SquaredExponentialKernel class."
            )

        # This approach is only valid for 1-dimensional features
        x_1, x_2 = jnp.squeeze(x_1), jnp.squeeze(x_2)

        # Rename variables to improve formatting
        mu, sd, length_scale = (
            self.feature_mean,
            self.feature_standard_deviation,
            self.feature_kernel.length_scale,
        )
        constant_1 = (2 / length_scale**2) + (1 / sd**2)
        constant_2 = ((x_1 + x_2) / length_scale**2) + (mu / sd**2)

        return jnp.sqrt(1 / (constant_1 * sd**2)) * jnp.exp(
            (constant_1 / 2) * ((constant_2 / constant_1) ** 2)
            - (x_1**2 + x_2**2) / (2 * length_scale**2)
            - mu**2 / (2 * sd**2)
        )

    @eqx.filter_jit
    def _compute_joint_expectation(
        self, x: Shaped[Array, " 1 d"], y: Shaped[Array, " 1 p"]
    ):
        r"""Compute :math:`E[k(X, X)l(Y, y)]` for :math:`l, k` RBF kernels."""
        if not isinstance(
            self.response_kernel, SquaredExponentialKernel
        ) or not isinstance(self.feature_kernel, SquaredExponentialKernel):
            raise ValueError(
                "The feature and response kernels must be instances of the"
                + " SquaredExponentialKernel class."
            )
        # Only valid for 1-dimensional features and responses
        x = jnp.squeeze(x)
        y = jnp.squeeze(y)

        # Rename variables to improve formatting
        x_length_scale, y_length_scale = (
            self.feature_kernel.length_scale,
            self.response_kernel.length_scale,
        )
        x_sd, y_sd = self.feature_standard_deviation, self.response_standard_deviation
        x_mu, a_0, a_1 = self.feature_mean, self.bias, self.slope

        # Compute terms relating to k(x, X)l(y, Y)
        constant_matrix_1 = jnp.array(
            [
                [1 / x_length_scale**2, 0],
                [0, 1 / y_length_scale**2],
            ]
        )
        constant_vector_1 = jnp.array(
            [
                [x / x_length_scale**2],
                [y / y_length_scale**2],
            ]
        )
        constant_scalar_1 = -(x**2 / (2 * x_length_scale**2)) - (
            y**2 / (2 * y_length_scale**2)
        )

        # Compute terms relating to f_{X, Y}(x, y)
        constant_matrix_2 = jnp.array(
            [
                [
                    1 / x_sd**2 + (a_1**2 / y_sd**2),
                    -(a_1 / y_sd**2),
                ],
                [
                    -(a_1 / y_sd**2),
                    1 / y_sd**2,
                ],
            ]
        )
        constant_vector_2 = jnp.array(
            [
                [x_mu / x_sd**2 - (a_0 * a_1) / y_sd**2],
                [a_0 / y_sd**2],
            ]
        )
        constant_scalar_2 = -(a_0**2) / (2 * y_sd**2) - x_mu**2 / (2 * x_sd**2)

        # Compute combined terms
        constant_matrix, constant_vector, constant_scalar = (
            constant_matrix_1 + constant_matrix_2,
            constant_vector_1 + constant_vector_2,
            constant_scalar_1 + constant_scalar_2,
        )

        # Compute integral
        return jnp.squeeze(
            (1 / jnp.sqrt(x_sd**2 * y_sd**2 * jnp.linalg.det(constant_matrix)))
            * jnp.exp(
                constant_scalar
                + (1 / 2)
                * constant_vector.T.dot(jnp.linalg.inv(constant_matrix)).dot(
                    constant_vector
                )
            )
        )

    def _set_up_loss_function(
        self, data: Shaped[Array, "N d"], supervision: Shaped[Array, "N p"]
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x_coreset: Shaped[Array, "M d"],
        x_target: Shaped[Array, "B d"],
    ) -> Optional[list]:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_target: A two-dimensional array of features used to estimate the loss
            function.
        """

    def _classification_loss_function(
        self,
        index: Shaped[Array, ""],
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_target: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`KIPExhaustiveSearch` solver targets.

        :param index: Scalar array with the index of the coreset we are optimising.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        return jnp.array([None])

    @eqx.filter_jit
    def _loss_function(
        self,
        indices: Shaped[Array, " J"],
        xs: Shaped[Array, "J d"],
        ys: Shaped[Array, "J p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_target: Shaped[Array, "B d"],
        y_target: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the exact Average Conditional Kernel Herding loss function."""
        # Delete unused invariant terms
        del x_target, y_target, invariant_terms

        # Update the current coreset with the xs and ys under consideration
        x_coreset = x_coreset.at[indices].set(xs)
        y_coreset = y_coreset.at[indices].set(ys)
        coreset_size = x_coreset.shape[0]

        # Compute the kernel gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)

        # Regularise the feature gramian ready for inversion
        regularised_coreset_feature_gramian = (
            coreset_feature_gramian
            + self.regularisation_parameter * jnp.eye(coreset_size)
        )

        # Vmap the expectation functions
        marginal_expectation_vmapped = vmap(
            vmap(self._compute_marginal_expectation, in_axes=(None, 0)),
            in_axes=(0, None),
        )
        joint_expectation_vmapped = vmap(
            vmap(self._compute_joint_expectation, in_axes=(None, 0)),
            in_axes=(0, None),
        )

        # Compute expectations exactly
        marginal_expectations = marginal_expectation_vmapped(x_coreset, x_coreset)
        joint_expectations = joint_expectation_vmapped(x_coreset, y_coreset)

        # First term is equivalent to Tr(K_tilde @ W_tilde @ L_tilde @ W_tilde)
        term_1 = (
            jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,
                b=coreset_response_gramian,
                assume_a="sym",
            )
            * jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,
                b=marginal_expectations,
                assume_a="sym",
            ).T
        ).sum()
        term_2 = jnp.trace(
            jsp.linalg.solve(
                a=regularised_coreset_feature_gramian,
                b=joint_expectations,
                assume_a="sym",
            )
        )

        return term_1 - 2 * term_2
