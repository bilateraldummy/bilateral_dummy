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
from jax import grad, vmap
from jaxtyping import Array, Shaped
from optax import OptState
from tqdm import tqdm as LoudTQDM  # noqa: N812

from coreax.coreset import Coreset
from coreax.data import SupervisedData
from coreax.kernels import ScalarValuedKernel, SquaredExponentialKernel
from coreax.solvers.base import CoresetSolver, _State
from coreax.util import KeyArrayLike, SilentTQDM

# pylint:disable=too-many-positional-arguments
# pylint:disable=too-many-locals
# pylint:disable=duplicate-code
# pylint:disable=too-many-statements
# pylint:disable=too-many-branches


class GradientHerdingState(eqx.Module):
    """Optimisation results for :class:`_GradientHerdingSolver`."""

    losses: Optional[list] = None
    gradient_norms: Optional[list] = None
    losses_iterations: Optional[list] = None
    gradient_norms_iterations: Optional[list] = None
    step_sizes: Optional[list] = None
    x_coresets: Optional[list] = None
    y_coresets: Optional[list] = None
    times: Optional[list] = None


class HerdingExhaustiveSearch(eqx.Module):
    """
    Exhaustive search of responses in herding-style classification problems.

    :param classes: Array of possible classes
    """

    classes: Shaped[Array, "C 1"]

    def update(
        self,
        loss: Callable[
            [
                Shaped[Array, "1 1"],
                Shaped[Array, "M 1"],
                Shaped[Array, "B 1"],
                list,
            ],
            Shaped[Array, " 1 1"],
        ],
        y_coreset: Shaped[Array, "M p"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: list,
    ) -> Shaped[Array, "J 1"]:
        """
        Optimise the responses point-by-point by exhaustive search.

        :param loss: Callable with type signature of _HerdingSolver._loss_function
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_batch: A two-dimensional array of supervision used to estimate the loss
            function.
        """
        # Vmap the loss function across the choice of class
        vmapped_loss_function = losses = vmap(loss, in_axes=(0, None, None, None))
        losses = vmapped_loss_function(
            self.classes,
            y_coreset,
            y_batch,
            invariant_terms,
        )
        return jnp.argmin(losses).astype(jnp.float64).reshape(-1, 1)


class _GradientHerdingSolver(CoresetSolver[SupervisedData, GradientHerdingState]):
    r"""
    Generic class for solving Herding-class problems via gradient descent.

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
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the features. Defaults to the Stochastic Gradient Descent (SGD) optimiser with a
        constant step schedule of 1e-3.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser for
        the responses. Defaults to the Stochastic Gradient Descent (SGD) optimiser with
        a constant step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    response_kernel: ScalarValuedKernel
    feature_optimiser: Optional[optax.GradientTransformation] = optax.sgd(
        optax.constant_schedule(1e-3)
    )
    response_optimiser: Optional[
        Union[optax.GradientTransformation, HerdingExhaustiveSearch]
    ] = optax.sgd(optax.constant_schedule(1e-3))
    batch_size: Optional[int] = None
    max_steps: int = 100
    convergence_parameter: float = 1e-3
    num_seeds: Optional[int] = 10
    track_info: bool = False

    @abstractmethod
    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    @abstractmethod
    def _set_up_classification_loss_function(
        self,
        x: Shaped[Array, "1 d"],
        x_coreset: Shaped[Array, "M d"],
        x_batch: Shaped[Array, "B d"],
    ) -> list:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x: The newest coreset feature added.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        """

    @abstractmethod
    def _classification_loss_function(
        self,
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_batch: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`HerdingExhaustiveSearch` solver targets.

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
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """
        Loss function that the solver batches.

        :param x: A two-dimensional array containing the coreset feature under
            consideration.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        :param y_batch: A two-dimensional array of supervision used to estimate the loss
            function.
        """

    @eqx.filter_jit
    def _step(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_opt_state: Optional[OptState],
        y_opt_state: Optional[OptState],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> tuple[
        Shaped[Array, "1 d"],
        Shaped[Array, "1 p"],
        Optional[OptState],
        Optional[OptState],
        Shaped[Array, "1 d + p"],
    ]:
        """
        Do gradient descent step.

        :param x: A two-dimensional array containing the coreset feature under
            consideration.
        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :x_opt_state: Current state of the :class:`~optax.GradientTransformation`
            feature optimiser.
        :y_opt_state: Current state of the :class:`~optax.GradientTransformation`
            supervision optimiser.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        :param y_batch: A two-dimensional array of supervision used to estimate the loss
            function.
        """
        # Do feature step according to chosen solver
        if (
            isinstance(self.feature_optimiser, optax.GradientTransformation)
            and x_opt_state is not None
        ):
            x_grad = grad(self._loss_function, argnums=0)(
                x, y, x_coreset, y_coreset, x_batch, y_batch, invariant_terms
            )
            x_update, x_opt_state = self.feature_optimiser.update(
                x_grad, x_opt_state, x
            )
            x_ = jnp.asarray(optax.apply_updates(x, x_update))
        else:
            # Default is to do nothing
            x_, x_grad = x, jnp.zeros((1, x.shape[1]))

        # Do response step according to chosen solver
        if (
            isinstance(self.response_optimiser, optax.GradientTransformation)
            and y_opt_state is not None
        ):
            y_grad = grad(self._loss_function, argnums=1)(
                x, y, x_coreset, y_coreset, x_batch, y_batch, invariant_terms
            )
            y_update, y_opt_state = self.response_optimiser.update(
                y_grad, y_opt_state, y
            )
            y_ = jnp.asarray(optax.apply_updates(y, y_update))
        elif isinstance(self.response_optimiser, HerdingExhaustiveSearch):
            # Compute the terms that are invariant when doing exhaustive search
            classification_invariant_terms = self._set_up_classification_loss_function(
                x_,  # Make sure we use the updated x
                x_coreset,
                x_batch,
            )
            y_ = self.response_optimiser.update(
                loss=self._classification_loss_function,
                y_coreset=y_coreset,
                y_batch=y_batch,
                invariant_terms=classification_invariant_terms,
            )
            y_grad = jnp.zeros((1, y.shape[1]))
        else:
            # Default is to do nothing
            y_, y_grad = y, jnp.zeros((1, y.shape[1]))

        return (
            x_,
            y_,
            x_opt_state,
            y_opt_state,
            jnp.hstack((x_grad, y_grad)),
        )

    def reduce(  # noqa:PLR0912,PLR0914,PLR0915,C901
        self, dataset: SupervisedData, solver_state: Optional[_State] = None
    ) -> tuple[Coreset[SupervisedData], GradientHerdingState]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The (potentially weighted and supervised) data to generate the
            coreset from
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information
        :return: a tuple of the solved coreset and intermediate solver state information
        """
        # Solver state is unused
        del solver_state

        # Extract features and response from dataset
        data, supervision = dataset.data, dataset.supervision

        # Check if we need to compute invariant terms
        invariant_timer = time()
        invariant_terms, batch_indices = None, None
        if self.batch_size is None:
            invariant_terms = self._set_up_loss_function(data, supervision)
        invariant_timer = time() - invariant_timer

        # Initialise the coreset vectors
        x_coreset = jnp.zeros((0, data.shape[1]))
        y_coreset = jnp.zeros((0, supervision.shape[1]))

        # Get the random keys used for batching and initialisation
        batch_keys = jr.split(self.random_key, (self.coreset_size, self.max_steps))
        initialise_keys = jr.split(self.random_key, (self.coreset_size, 2))

        # Initialise trackers
        (
            losses,
            gradient_norms,
            losses_iterations,
            gradient_norms_iterations,
            step_sizes,
            x_coresets,
            y_coresets,
            times,
        ) = [], [], [], [], [], [], [], []
        if self.track_info:
            # Suppress progress bar as we print our own custom one
            progress_bar = SilentTQDM
        else:
            progress_bar = LoudTQDM

        for i in progress_bar(range(self.coreset_size)):
            seed_timer = time()
            if self.num_seeds is None:
                # Initialise pair with random sample
                best_index = jr.choice(
                    initialise_keys[i, 0], len(dataset), shape=(1,), replace=False
                )[0]
                loss_value = jnp.array([])
                if self.track_info:
                    # Record the loss value pre-gradient step
                    loss_value = self._loss_function(
                        data[[best_index], :],
                        supervision[[best_index], :],
                        x_coreset,
                        y_coreset,
                        data[batch_indices] if batch_indices is not None else data,
                        supervision[batch_indices]
                        if batch_indices is not None
                        else supervision,
                        invariant_terms,
                    )
            else:
                # Sample indices to select seed pairs
                seed_indices = jr.choice(
                    initialise_keys[i, 0],
                    len(dataset),
                    shape=(self.num_seeds,),
                    replace=False,
                )
                x_seeds, y_seeds = data[seed_indices, :], supervision[seed_indices, :]

                # Sample a batch we will use to estimate the loss function
                if self.batch_size is not None:
                    batch_indices = jr.choice(
                        initialise_keys[i, 1],
                        len(dataset),
                        shape=(self.batch_size,),
                        replace=False,
                    )

                # Compute the loss for each seed pair and choose the best one
                initial_losses = vmap(
                    self._loss_function,
                    in_axes=(0, 0, None, None, None, None, None),
                )(
                    x_seeds,
                    y_seeds,
                    x_coreset,
                    y_coreset,
                    data[batch_indices] if batch_indices is not None else data,
                    supervision[batch_indices]
                    if batch_indices is not None
                    else supervision,
                    invariant_terms,
                )
                best_index = seed_indices[initial_losses.argmin()]
                loss_value = initial_losses.min()
            seed_timer = time() - seed_timer

            # Initialise the optimisers
            x, y = data[[best_index], :], supervision[[best_index], :]
            x_opt_state, y_opt_state = None, None
            if isinstance(self.feature_optimiser, optax.GradientTransformation):
                x_opt_state = self.feature_optimiser.init(x)
            if isinstance(self.response_optimiser, optax.GradientTransformation):
                y_opt_state = self.response_optimiser.init(y)

            if self.track_info:
                # Record the loss value pre-gradient step
                losses.append(loss_value.item())

                # Track position of coresets
                x_coresets.append(jnp.vstack((x_coreset, x)))
                y_coresets.append(jnp.vstack((y_coreset, y)))

                # Track timings
                if i == 0:
                    times.append(invariant_timer + seed_timer)
                else:
                    times.append(seed_timer)

            iteration_number = 0
            for j in range(self.max_steps):
                iteration_number += 1
                # Randomly sample a batch of data pairs to estimate the loss
                if self.batch_size is not None:
                    batch_indices = jr.choice(
                        batch_keys[i, j],
                        len(dataset),
                        shape=(self.batch_size,),
                        replace=False,
                    )

                # Copy current coreset for tracking purposes
                old_x, old_y = None, None
                if self.track_info:
                    old_x, old_y = (jnp.copy(x), jnp.copy(y))

                # Do gradient step
                step_timer = time()
                x, y, x_opt_state, y_opt_state, grads = self._step(
                    x=x,
                    y=y,
                    x_opt_state=x_opt_state,
                    y_opt_state=y_opt_state,
                    x_coreset=x_coreset,
                    y_coreset=y_coreset,
                    x_batch=data[batch_indices] if batch_indices is not None else data,
                    y_batch=supervision[batch_indices]
                    if batch_indices is not None
                    else supervision,
                    invariant_terms=invariant_terms,
                )
                step_timer = time() - step_timer

                # Compute norm of the gradient for convergence purposes
                gradient_norm = jnp.linalg.norm(grads)

                # Keep track of progress
                if self.track_info:
                    # Track timings
                    times.append(step_timer)

                    # Compute the step size
                    step_size = jnp.linalg.norm(
                        jnp.hstack((x, y)) - jnp.hstack((old_x, old_y))
                    )
                    step_sizes.append(step_size.item())

                    # Add coreset
                    x_coresets.append(jnp.vstack((x_coreset, x)))
                    y_coresets.append(jnp.vstack((y_coreset, y)))

                    # Add gradient path
                    gradient_norms.append(gradient_norm)

                    # Store loss value
                    loss_value = self._loss_function(
                        x=x,
                        y=y,
                        x_coreset=x_coreset,
                        y_coreset=y_coreset,
                        x_batch=data[batch_indices]
                        if batch_indices is not None
                        else data,
                        y_batch=supervision[batch_indices]
                        if batch_indices is not None
                        else supervision,
                        invariant_terms=invariant_terms,
                    )
                    losses.append(loss_value.item())

                # Check convergence
                if gradient_norm < self.convergence_parameter:
                    break

            if self.track_info:
                losses_iterations.append(losses[-1])
                gradient_norms_iterations.append(gradient_norms[-1])

                itr_string = f"{i + 1}/{self.coreset_size}"
                statement = (
                    f"Iteration {itr_string:<8} | "
                    + f"Stopped at step {iteration_number:<8} | "
                    + f"Loss = {losses[-1]:<12.10f} | "
                    + f"Gradient Norm = {gradient_norms[-1]:<12.10f} | "
                    + f"Coreset Delta Norm = {step_sizes[-1]:<12.10f}"
                    + "                                                  "
                )
                print(statement)

            # Update the coreset with the solved pair
            # debug.print("After {x}", x=(x, y))
            x_coreset = jnp.vstack((x_coreset, x))
            y_coreset = jnp.vstack((y_coreset, y))

        return (
            Coreset(SupervisedData(x_coreset, y_coreset), dataset),
            GradientHerdingState(
                losses,
                gradient_norms,
                losses_iterations,
                gradient_norms_iterations,
                step_sizes,
                x_coresets,
                y_coresets,
                times,
            ),
        )


class PseudoJointKernelHerding(_GradientHerdingSolver):
    r"""
    Joint Kernel Herding - a batch stochastic gradient descent coreset solver.

    Joint Kernel Herding is a batch stochastic gradient descent algorithm which
    learns a coreset by batching the Joint Maximum Mean Discrepancy (JMMD)
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
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x: Shaped[Array, "1 d"],
        x_coreset: Shaped[Array, "M d"],
        x_batch: Shaped[Array, "B d"],
    ) -> list:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x: The newest coreset feature added.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        """
        batch_feature_evaluations = self.feature_kernel.compute(x, x_batch)
        coreset_feature_evaluations = self.feature_kernel.compute(x, x_coreset)
        return [batch_feature_evaluations, coreset_feature_evaluations]

    def _classification_loss_function(
        self,
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_batch: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`HerdingExhaustiveSearch` solver targets.

        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        # Extract invariant terms
        batch_feature_evaluations, coreset_feature_evaluations = invariant_terms

        # Compute required kernel matrices
        batch_response_evaluations = self.response_kernel.compute(y, y_batch)
        coreset_response_evaluations = self.response_kernel.compute(y, y_coreset)

        # First term is (1/(m+1)) * \sum_{i=1}^m k(x, x_i) l(y, y_i)
        term_1 = (coreset_feature_evaluations * coreset_response_evaluations).sum() / (
            y_coreset.shape[0] + 1
        )

        # First term is approximation of E_{X, Y}[k(x, X)l(y, Y)]
        term_2 = (
            batch_feature_evaluations * batch_response_evaluations
        ).sum() / y_batch.shape[0]

        return term_1 - term_2

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Joint Kernel Herding loss function."""
        del invariant_terms

        # Compute the required kernel matrices
        batch_feature_evaluations = self.feature_kernel.compute(x, x_batch)
        coreset_feature_evaluations = self.feature_kernel.compute(x, x_coreset)
        batch_response_evaluations = self.response_kernel.compute(y, y_batch)
        coreset_response_evaluations = self.response_kernel.compute(y, y_coreset)

        # First term is (1/(m+1)) * \sum_{i=1}^m k(x, x_i) l(y, y_i)
        term_1 = (coreset_feature_evaluations * coreset_response_evaluations).sum() / (
            x_coreset.shape[0] + 1
        )

        # First term is approximation of E_{X, Y}[k(x, X)l(y, Y)]
        term_2 = (
            batch_feature_evaluations * batch_response_evaluations
        ).sum() / x_batch.shape[0]

        return term_1 - term_2


class ExactPseudoJointKernelHerding(_GradientHerdingSolver):
    r"""
    Joint Kernel Herding - a batch stochastic gradient descent coreset solver.

    Joint Kernel Herding is a batch stochastic gradient descent algorithm which
    learns a coreset by batching the Joint Maximum Mean Discrepancy (JMMD)
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
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    :param bias: The parameter :math:`\a_0`.
    :param slope: The parameter :math:`\a_1`.
    :param feature_mean: The parameter :math:`\mu_x`.
    :param feature_standard_deviation: The parameter :math:`\sigma_x`.
    :param response_standard_deviation: The parameter :math:`\sigma_y`.
    """

    regularisation_parameter: float = 1e-3
    bias: float = 0.0
    slope: float = 1.0
    feature_mean: float = 0.0
    feature_standard_deviation: float = 1
    response_standard_deviation: float = 0.25

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x: Shaped[Array, "1 d"],
        x_coreset: Shaped[Array, "M d"],
        x_batch: Shaped[Array, "B d"],
    ) -> list:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x: The newest coreset feature added.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        """
        return [None]

    def _classification_loss_function(
        self,
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_batch: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`HerdingExhaustiveSearch` solver targets.

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
    def _compute_joint_expectation(self, x, y):
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

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Joint Kernel Herding loss function."""
        # Delete unused terms
        del invariant_terms, x_batch, y_batch

        # Compute the required kernel matrices
        coreset_feature_evaluations = self.feature_kernel.compute(x, x_coreset)
        coreset_response_evaluations = self.response_kernel.compute(y, y_coreset)

        # First term is (1/(m+1)) * \sum_{i=1}^m k(x, x_i) l(y, y_i)
        term_1 = (coreset_feature_evaluations * coreset_response_evaluations).sum() / (
            x_coreset.shape[0] + 1
        )

        # First term is E_{X, Y}[k(x, X)l(y, Y)]
        term_2 = self._compute_joint_expectation(x, y)

        return term_1 - term_2


class ConditionalKernelHerding(_GradientHerdingSolver):
    r"""
    Conditional Kernel Herding - a batch stochastic gradient descent coreset solver.

    Conditional Kernel Herding is a batch stochastic gradient descent algorithm which
    learns a coreset by batching the Conditional Maximum Mean Discrepancy (CMMD)
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
    :param regularisation_parameter: Regularisation parameter for stable inversion
            of arrays, negative values will be converted to positive.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    regularisation_parameter: float = 1e-3
    coreset_regularisation_parameter: float = 1e-3

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """
        # Delete unused terms
        del supervision

        # Compute the batch feature gramian
        batch_feature_gramian = self.feature_kernel.compute(data, data)

        # Regularise the feature gramian ready for inversion
        identity = jnp.eye(data.shape[0])
        regularised_batch_feature_gramian = (
            batch_feature_gramian + self.regularisation_parameter * identity
        )

        # Compute the LU factorisation of the regularised feature gramian
        lu_factor = jsp.linalg.lu_factor(regularised_batch_feature_gramian)
        return [lu_factor]

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function."""
        # Expand the coreset
        x_coreset = jnp.vstack((x, x_coreset))
        y_coreset = jnp.vstack((y, y_coreset))
        coreset_size = x_coreset.shape[0]
        batch_size = x_batch.shape[0]

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

        # Compute the cross-gramians between the coreset and the batches
        cross_feature_gramian = self.feature_kernel.compute(x_coreset, x_batch)
        cross_response_gramian = self.response_kernel.compute(y_batch, y_coreset)

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

        # Check if we have precomputed the batch feature gramian
        if invariant_terms is not None:
            batch_lu_factor = invariant_terms[0]
        else:
            # Compute the batch feature gramian
            batch_feature_gramian = self.feature_kernel.compute(x_batch, x_batch)

            # Regularise the feature gramian ready for inversion
            regularised_batch_feature_gramian = (
                batch_feature_gramian
                + self.regularisation_parameter * jnp.eye(batch_size)
            )

            batch_lu_factor = jsp.linalg.lu_factor(regularised_batch_feature_gramian)

        # Evaluate the batch-trained KCME on the coreset
        batch_kcme_evaluations = jsp.linalg.lu_solve(
            lu_and_piv=batch_lu_factor,
            b=cross_response_gramian,
        )

        # Second term is equivalent to Tr(K_bar @ W_tilde @ L_bar^T @ W_batch)
        term_2 = (
            jsp.linalg.lu_solve(
                lu_and_piv=coreset_lu_factor,
                b=cross_feature_gramian,
            )
            * batch_kcme_evaluations.T  # m x b
        ).sum()

        return term_1 - 2 * term_2


class ExactConditionalKernelHerding(_GradientHerdingSolver):
    r"""
    Exact Conditional Kernel Herding - a gradient descent coreset solver.

    Given a function :math:`f:\mathcal{X} \to \mathbb{R}`, we define the conditional
    distribution to be :math:`\mathbb{P}(Y|X=x) = \mathcal{N}(f(x), \sigma^2)`. Given
    this batch distribution, Exact Conditional Kernel Herding is a gradient descent
    algorithm which learns a coreset by batching the Conditional Maximum Mean
    Discrepancy (CMMD) between the true conditional distribution, and the conditional
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
    :param regularisation_parameter: Regularisation parameter for stable inversion
            of arrays, negative values will be converted to positive.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    :param generating_function: The function :math:`f:\mathcal{X} \to \mathbb{R}`.
    :param standard_deviation: The parameter :math:`\sigma`.
    """

    regularisation_parameter: float = 1e-3
    generating_function: Callable = lambda x: x
    standard_deviation: float = 0.25

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """
        # Delete unused terms
        del data, supervision

    @eqx.filter_jit
    def _compute_conditional_expectation(self, x, y):
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

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function exactly."""
        # Delete unused terms
        del x_batch, y_batch, invariant_terms

        # Expand the coreset
        x_coreset = jnp.vstack((x, x_coreset))
        y_coreset = jnp.vstack((y, y_coreset))
        coreset_size = x_coreset.shape[0]

        # Compute the kernel gramians
        coreset_feature_gramian = self.feature_kernel.compute(x_coreset, x_coreset)
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)

        # Regularise the feature gramian ready for inversion
        regularised_coreset_feature_gramian = (
            coreset_feature_gramian
            + self.regularisation_parameter * jnp.eye(coreset_size)
        )

        # First term is equivalent to Tr(K_tilde @ W_tilde @ L_tilde @ W_tilde)
        term_1 = (
            jnp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=coreset_feature_gramian,  # m x m
            )
            * jnp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=coreset_response_gramian,  # m x m
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
            jnp.linalg.solve(
                a=regularised_coreset_feature_gramian,  # m x m
                b=conditional_expectations,  # m x m
            )
        )

        return term_1 - 2 * term_2


class AverageConditionalKernelHerding(_GradientHerdingSolver):
    r"""
    An implementation of Average Conditional Kernel Herding.

    Average Conditional Kernel Herding is a batch stochastic gradient descent algorithm
    which learns a coreset by batching the Average Maximum Conditional Mean Discrepancy
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
    :param regularisation_parameter: Regularisation parameter for stable inversion
            of arrays, negative values will be converted to positive.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    """

    regularisation_parameter: float = 1e-3

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x: Shaped[Array, "1 d"],
        x_coreset: Shaped[Array, "M d"],
        x_batch: Shaped[Array, "B d"],
    ) -> list:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x: The newest coreset feature added.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        """
        x_coreset = jnp.vstack((x, x_coreset))
        common_term = jsp.linalg.solve(
            a=self.feature_kernel.compute(x_coreset, x_coreset)
            + self.regularisation_parameter * jnp.eye(x_coreset.shape[0]),  # m x m
            b=self.feature_kernel.compute(x_coreset, x_batch),
            assume_a="sym",
        )
        return [common_term.dot(common_term[0, :]).T, common_term[0, :].T]

    def _classification_loss_function(
        self,
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_batch: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`HerdingExhaustiveSearch` solver targets.

        :param y: A two-dimensional array containing the coreset supervision under
            consideration.
        :param y_coreset: A two-dimensional array containing the current coreset
            supervision.
        :param y_target: A two-dimensional array of supervision used to estimate the
            loss function.
        :invariant_terms: Terms which are constant with respect to the features of the
            coreset, and can be precomputed and stored.
        """
        # Expand the coreset
        y_coreset = jnp.vstack((y, y_coreset))

        # Compute the non-invariant terms
        coreset_response_vector = self.response_kernel.compute(y, y_coreset)
        full_response_vector = self.response_kernel.compute(y, y_batch)

        # Compute loss function
        term_1 = coreset_response_vector.dot(invariant_terms[0])
        term_2 = full_response_vector.dot(invariant_terms[1])

        return jnp.squeeze(2 / y_batch.shape[0] * (term_1 - term_2))

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Average Conditional Kernel Herding loss function."""
        del invariant_terms

        # Expand the coreset
        x_coreset = jnp.vstack((x, x_coreset))
        y_coreset = jnp.vstack((y, y_coreset))
        coreset_size = x_coreset.shape[0]
        batch_size = x_batch.shape[0]

        # The first and second terms of the loss function share K_bar @ W_tilde:
        common_term = jsp.linalg.solve(
            a=self.feature_kernel.compute(x_coreset, x_coreset)
            + self.regularisation_parameter * jnp.eye(coreset_size),  # m x m
            b=self.feature_kernel.compute(x_coreset, x_batch),
            assume_a="sym",
        )

        # Compute the cross-gramians between the coreset and the batches
        coreset_response_gramian = self.response_kernel.compute(y_coreset, y_coreset)
        cross_response_gramian = self.response_kernel.compute(y_coreset, y_batch)

        # First term is equivalent to Tr(K_bar @ W_tilde @ L_tilde @ W_tilde @ K_bar^T)
        term_1 = (coreset_response_gramian.dot(common_term) * common_term).sum()

        # Second term is equivalent to Tr(K_bar @ W_tilde @ L_bar^T)
        term_2 = (cross_response_gramian * common_term).sum()

        return (term_1 - 2 * term_2) / batch_size


class ExactAverageConditionalKernelHerding(_GradientHerdingSolver):
    r"""
    Exact Average Conditional Kernel Herding - a gradient descent coreset solver.

    Given the function :math:`f:\mathcal{X} \to \mathbb{R}`, :math:`f(x) := a_0 + a_1x`,
    we define the conditional distribution to be
    :math:`\mathbb{P}(Y|X=x) = \mathcal{N}(f(x), \sigma_y^2)`, and the marginal
    distribution as :math:`\mathbb{P}(X) = \mathcal{N}(\mu_x, \sigma_x^2)`. Given these
    distributions, Exact Average Conditional Kernel Herding is a gradient descent
    algorithm which learns a coreset by batching the Average Maximum Conditional Mean
    Discrepancy (AMCMD) between the true conditional distribution, and the conditional
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
    :param regularisation_parameter: Regularisation parameter for stable inversion
            of arrays, negative values will be converted to positive.
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Stochastic Gradient Descent (SGD) optimiser with a constant
        step schedule of 1e-3.
    :param batch_size: An integer representing the number of data pairs to use for
        estimation of joint expectations. Defaults to :data:`None`, uses the entire
        dataset.
    :param max_steps: An integer representing the maximum permitted number of gradient
        steps. Defaults to :math:`100`.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param num_seeds: Number of points to sample to act as initial seeds for
        optimisation. Defaults to :math:`10`.
    :param metric: Instance of class:`~coreax.metrics.Metric` to compute at the end of
        every iteration. Defaults to :data:`None`.
    :param track_info: Whether or not to print and store optimisation information.
        Defaults to :data:`False`.
    :param bias: The parameter :math:`\a_0`.
    :param slope: The parameter :math:`\a_1`.
    :param feature_mean: The parameter :math:`\mu_x`.
    :param feature_standard_deviation: The parameter :math:`\sigma_x`.
    :param response_standard_deviation: The parameter :math:`\sigma_y`.
    """

    regularisation_parameter: float = 1e-3
    bias: float = 0.0
    slope: float = 1.0
    feature_mean: float = 0.0
    feature_standard_deviation: float = 1
    response_standard_deviation: float = 0.25

    def _set_up_loss_function(
        self,
        data: Shaped[Array, "N d"],
        supervision: Shaped[Array, "N p"],
    ) -> Optional[list]:
        """
        Precompute and store invariant terms.

        :param data: Array containing the features of the dataset.
        :param supervision: Array containing the responses of the dataset.
        """

    def _set_up_classification_loss_function(
        self,
        x: Shaped[Array, "1 d"],
        x_coreset: Shaped[Array, "M d"],
        x_batch: Shaped[Array, "B d"],
    ) -> list:
        """
        Precompute invariant terms to reduce cost when using KIPExhaustiveSearch.

        :param x: The newest coreset feature added.
        :param x_coreset: A two-dimensional array containing the current coreset
            features.
        :param x_batch: A two-dimensional array of features used to estimate the loss
            function.
        """
        return [None]

    def _classification_loss_function(
        self,
        y: Shaped[Array, "1 1"],
        y_coreset: Shaped[Array, "M 1"],
        y_batch: Shaped[Array, "B 1"],
        invariant_terms: list,
    ) -> Shaped[Array, ""]:
        """
        Loss function that the :class:`HerdingExhaustiveSearch` solver targets.

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
    def _compute_marginal_expectation(self, x_1, x_2):
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
    def _compute_joint_expectation(self, x, y):
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

        # Compute combined constants
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

    @eqx.filter_jit
    def _loss_function(
        self,
        x: Shaped[Array, "1 d"],
        y: Shaped[Array, "1 p"],
        x_coreset: Shaped[Array, "M d"],
        y_coreset: Shaped[Array, "M p"],
        x_batch: Shaped[Array, "B d"],
        y_batch: Shaped[Array, "B p"],
        invariant_terms: Optional[list],
    ) -> Shaped[Array, "1 1"]:
        """Compute the Conditional Kernel Herding loss function exactly."""
        # Delete unused terms
        del x_batch, y_batch, invariant_terms

        # Expand the coreset
        x_coreset = jnp.vstack((x, x_coreset))
        y_coreset = jnp.vstack((y, y_coreset))
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
            jnp.linalg.solve(
                a=regularised_coreset_feature_gramian, b=coreset_response_gramian
            )
            * jnp.linalg.solve(
                a=regularised_coreset_feature_gramian, b=marginal_expectations
            ).T
        ).sum()
        term_2 = jnp.trace(
            jnp.linalg.solve(
                a=regularised_coreset_feature_gramian, b=joint_expectations
            )
        )

        return term_1 - 2 * term_2
