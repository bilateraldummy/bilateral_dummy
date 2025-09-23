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

r"""
Classes and associated functionality to use kernel functions.

A kernel is a non-negative, real-valued integrable function that can take two inputs,
``x`` and ``y``, and returns a value that decreases as ``x`` and ``y`` move further away
in space from each other. Note that *further* here may account for cyclic behaviour in
the data, for example.

In this library, we often use kernels as a smoothing tool: given a dataset of distinct
points, we can reconstruct the underlying data generating distribution through smoothing
of the data with kernels.

Some kernels are parameterizable and may represent other well known kernels when given
appropriate parameter values. For example, the
:class:`~coreax.kernels.SquaredExponentialKernel`,

.. math::

    k(x,y) = \text{output_scale} * \exp (-||x-y||^2/2 * \text{length_scale}^2),

which if parameterized with an ``output_scale`` of
:math:`\frac{1}{\sqrt{2\pi} \,*\, \text{length_scale}}`, yields the well known
Gaussian kernel.

A :class:`~coreax.kernels.ScalarValuedKernel` must implement a
:meth:`~coreax.kernels.ScalarValuedKernel.compute_elementwise` method, which returns the
floating point value after evaluating the kernel on two floats, ``x`` and ``y``.
Additional methods, such as :meth:`ScalarValuedKernel.grad_x_elementwise`, can
optionally be overridden to improve performance. The canonical example is when a
suitable closed-form representation of a higher-order gradient can be used to avoid the
expense of performing automatic differentiation.

Such an example can be seen in :class:`coreax.kernels.SteinKernel`, where the analytic
forms of :meth:`ScalarValuedKernel.divergence_x_grad_y` are significantly cheaper to
compute than the automatic differentiated default.
"""

from abc import abstractmethod
from typing import Any, Callable, Literal, Optional, Union, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, grad, jacrev
from jaxtyping import Shaped
from typing_extensions import override

from coreax.data import Data, SupervisedData, as_supervised_data
from coreax.kernels.util import _block_data_convert, _block_supervised_data_convert
from coreax.util import pairwise, pairwise_tuple, tree_leaves_repeat


class ScalarValuedKernel(eqx.Module):  # noqa: PLR0904
    """Abstract base class for scalar-valued kernels."""

    def __add__(
        self, addition: Union["ScalarValuedKernel", int, float]
    ) -> "AdditiveKernel":
        """Overload `+` operator."""
        if isinstance(addition, (int, float)):
            return AdditiveKernel(self, _Constant(addition))
        if isinstance(addition, ScalarValuedKernel):
            return AdditiveKernel(self, addition)
        raise ValueError(
            "'addition' must be an instance of a 'ScalarValuedKernel', "
            + "an integer or a float"
        )

    def __radd__(
        self, addition: Union["ScalarValuedKernel", int, float]
    ) -> "AdditiveKernel":
        """Overload right `+` operator, order is mathematically irrelevant."""
        return self.__add__(addition)

    def __mul__(
        self, product: Union["ScalarValuedKernel", int, float]
    ) -> "ProductKernel":
        """Overload `*` operator."""
        if isinstance(product, (int, float)):
            return ProductKernel(self, _Constant(product))
        if isinstance(product, ScalarValuedKernel):
            return ProductKernel(self, product)
        raise ValueError(
            "'product' must be an instance of a 'ScalarValuedKernel', "
            + "an integer or a float"
        )

    def __rmul__(
        self, product: Union["ScalarValuedKernel", int, float]
    ) -> "ProductKernel":
        """Overload right `*` operator, order is mathematically irrelevant."""
        return self.__mul__(product)

    def __pow__(self, power: int) -> "PowerKernel":
        """
        Overload `**` operator.

        .. note::
            The positive semi-definiteness of the `PowerKernel` can only be guaranteed
            for positive integer powers. An error is thrown for float or negative
            `power`.

        """
        return PowerKernel(self, power)

    def __mod__(self, tensor: "ScalarValuedKernel") -> "TensorProductKernel":
        """Overload `%` operator to produce tensor products of kernel functions."""
        if not isinstance(tensor, ScalarValuedKernel):
            return NotImplemented
        return TensorProductKernel(self, tensor)

    @overload
    def compute(
        self, x: Shaped[Array, " n d"], y: Shaped[Array, " m d"]
    ) -> Shaped[Array, " n m"]: ...

    @overload
    def compute(  # pyright: ignore[reportOverlappingOverload]
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, " 1 1"]: ...

    def compute(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
    ) -> Union[Shaped[Array, " n m"], Shaped[Array, "1 1"]]:
        r"""
        Evaluate the kernel on input data ``x`` and ``y``.

        The 'data' can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        Evaluation is always vectorised.

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: Kernel evaluations between points in ``x`` and ``y``. If ``x`` = ``y``,
            then this is the Gram matrix corresponding to the RKHS inner product.
        """
        return pairwise(self.compute_elementwise)(x, y)

    @abstractmethod
    def compute_elementwise(
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, ""]:
        r"""
        Evaluate the kernel on individual input vectors ``x`` and ``y``, not-vectorised.

        Vectorisation only becomes relevant in terms of computational speed when we
        have multiple ``x`` or ``y``.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """

    @overload
    def grad_x(
        self, x: Shaped[Array, " n d"], y: Shaped[Array, " m d"]
    ) -> Shaped[Array, " n m d"]: ...

    @overload
    def grad_x(  # pyright: ignore[reportOverlappingOverload]
        self, x: Shaped[Array, " d"], y: Shaped[Array, " d"]
    ) -> Shaped[Array, " 1 1 d"]: ...

    @overload
    def grad_x(
        self,
        x: Union[Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, " 1 1 1"]: ...

    def grad_x(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
    ) -> Union[
        Shaped[Array, " n m d"], Shaped[Array, " 1 1 d"], Shaped[Array, "1 1 1"]
    ]:
        r"""
        Evaluate the gradient (Jacobian) of the kernel function w.r.t. ``x``.

        The function is vectorised, so ``x`` or ``y`` can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`n \times m \times d` array of pairwise Jacobians
        """
        return pairwise(self.grad_x_elementwise)(x, y)

    @overload
    def grad_y(
        self, x: Shaped[Array, " n d"], y: Shaped[Array, " m d"]
    ) -> Shaped[Array, " n m d"]: ...

    @overload
    def grad_y(  # pyright: ignore[reportOverlappingOverload]
        self, x: Shaped[Array, " d"], y: Shaped[Array, " d"]
    ) -> Shaped[Array, " 1 1 d"]: ...

    @overload
    def grad_y(
        self,
        x: Union[Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, " 1 1 1"]: ...

    def grad_y(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
    ) -> Union[
        Shaped[Array, " n m d"], Shaped[Array, " 1 1 d"], Shaped[Array, "1 1 1"]
    ]:
        r"""
        Evaluate the gradient (Jacobian) of the kernel function w.r.t. ``y``.

        The function is vectorised, so ``x`` or ``y`` can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`m \times n \times d` array of pairwise Jacobians
        """
        return pairwise(self.grad_y_elementwise)(x, y)

    @overload
    def grad_x_elementwise(
        self, x: Shaped[Array, " d"], y: Shaped[Array, " d"]
    ) -> Shaped[Array, " d"]: ...

    @overload
    def grad_x_elementwise(
        self,
        x: Union[Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, ""]: ...

    def grad_x_elementwise(
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Union[Shaped[Array, " d"], Shaped[Array, ""]]:
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``x``.

        The gradient (Jacobian) of the kernel function w.r.t. ``x``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays.
        :meth:`coreax.kernels.ScalarValuedKernel.grad_x` provides a vectorised version
        of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return grad(self.compute_elementwise, 0)(x, y)

    @overload
    def grad_y_elementwise(
        self, x: Shaped[Array, " d"], y: Shaped[Array, " d"]
    ) -> Shaped[Array, " d"]: ...

    @overload
    def grad_y_elementwise(
        self,
        x: Union[Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, ""]: ...

    def grad_y_elementwise(
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Union[Shaped[Array, " d"], Shaped[Array, ""]]:
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``y``.

        The gradient (Jacobian) of the kernel function w.r.t. ``y``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays.
        :meth:`coreax.kernels.ScalarValuedKernel.grad_y` provides a vectorised version
        of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return grad(self.compute_elementwise, 1)(x, y)

    @overload
    def divergence_x_grad_y(
        self, x: Shaped[Array, " n d"], y: Shaped[Array, " m d"]
    ) -> Shaped[Array, " n m"]: ...

    @overload
    def divergence_x_grad_y(
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, " 1 1"]: ...

    def divergence_x_grad_y(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            bool,
        ],
    ) -> Union[Shaped[Array, " n m"], Shaped[Array, " 1 1"]]:
        r"""
        Evaluate the divergence operator w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        This function is vectorised, so it accepts vectors or arrays.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Array of Laplace-style operator traces :math:`n \times m` array
        """
        return pairwise(self.divergence_x_grad_y_elementwise)(x, y)

    def divergence_x_grad_y_elementwise(
        self,
        x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
        y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int, bool],
    ) -> Shaped[Array, ""]:
        r"""
        Evaluate the element-wise divergence w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`~coreax.kernels.ScalarValuedKernel.divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        pseudo_hessian = jacrev(self.grad_y_elementwise, 0)(x, y)
        return pseudo_hessian.trace()

    def gramian_row_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        *,
        block_size: Union[int, None, tuple[Optional[int], Optional[int]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Shaped[Array, " n"]:
        r"""
        Compute the (blocked) row-mean of the kernel's Gramian matrix.

        A convenience method for calling :meth:`compute_mean`. Equivalent to the call
        :code:`compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)`.

        :param x: Data matrix, :math:`n \times d`
        :param block_size: Block size parameter passed to :meth:`compute_mean`
        :param unroll: Unroll parameter passed to :meth:`compute_mean`
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        return self.compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)

    @overload
    def compute_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        axis: Literal[0] = 0,
        *,
        block_size: Union[int, None, tuple[Optional[int], Optional[int]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Shaped[Array, " #m"]: ...

    @overload
    def compute_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        axis: Literal[1] = 1,
        *,
        block_size: Union[int, None, tuple[Optional[int], Optional[int]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Shaped[Array, " #n"]: ...

    @overload
    def compute_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        axis: None = None,
        *,
        block_size: Union[int, None, tuple[Optional[int], Optional[int]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Shaped[Array, ""]: ...

    def compute_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        y: Union[
            Shaped[Array, " m d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        axis: Optional[int] = None,
        *,
        block_size: Union[int, None, tuple[Optional[int], Optional[int]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Union[Shaped[Array, " n"], Shaped[Array, " m"], Shaped[Array, ""]]:
        r"""
        Compute the (blocked) mean of the matrix :math:`K_{ij} = k(x_i, y_j)`.

        The :math:`n \times m` kernel matrix :math:`K_{ij} = k(x_i, y_j)`, where
        ``x`` and ``y`` are respectively :math:`n \times d` and :math:`m \times d`
        (weighted) data matrices, has the following (weighted) means:

        - mean (:code:`axis=None`) :math:`\frac{1}{n m}\sum_{i,j=1}^{n, m} K_{ij}`
        - row-mean (:code:`axis=0`) :math:`\frac{1}{n}\sum_{i=1}^{n} K_{ij}`
        - column-mean (:code:`axis=1`) :math:`\frac{1}{m}\sum_{j=1}^{m} K_{ij}`

        If ``x`` and ``y`` are of type :class:`~coreax.data.Data`, their weights are
        used to compute the weighted mean as defined in :func:`jax.numpy.average`.

        .. note::
            The conventional 'mean' is a scalar, the 'row-mean' is an :math:`m`-vector,
            while the 'column-mean' is an :math:`n`-vector.

        To avoid materializing the entire matrix (memory cost :math:`\mathcal{O}(n m)`),
        we accumulate the mean over blocks (memory cost :math:`\mathcal{O}(B_x B_y)`,
        where ``B_x`` and ``B_y`` are user-specified block-sizes for blocking the ``x``
        and ``y`` parameters respectively.

        .. note::
            The data ``x`` and/or ``y`` are padded with zero-valued and zero-weighted
            data points, when ``B_x`` and/or ``B_y`` are non-integer divisors of ``n``
            and/or ``m``. Padding does not alter the result, but does provide the block
            shape stability required by :func:`jax.lax.scan` (used for block iteration).

        :param x: Data matrix, :math:`n \times d`
        :param y: Data matrix, :math:`m \times d`
        :param axis: Which axis of the kernel matrix to compute the mean over; a value
            of `None` computes the mean over both axes
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: The (weighted) mean of the kernel matrix :math:`K_{ij}`
        """
        operands = x, y
        inner_unroll, outer_unroll = tree_leaves_repeat(unroll, len(operands))
        _block_size = tree_leaves_repeat(block_size, len(operands))
        # Row-mean is the argument reversed column-mean due to symmetry k(x,y) = k(y,x)
        if axis == 0:
            operands = operands[::-1]
            _block_size = _block_size[::-1]

        def _is_data(x: Any) -> bool:
            """Return boolean indicating if ``x`` is an instance of `Data`."""
            return isinstance(x, Data)

        (block_x, unpadded_len_x), (block_y, _) = jtu.tree_map(
            _block_data_convert,
            operands,
            tuple(_block_size),
            is_leaf=_is_data,
        )

        def block_sum(
            accumulated_sum: Shaped[Array, ""], x_block: Data
        ) -> tuple[Array, Array]:
            """Block reduce/accumulate over ``x``."""

            def slice_sum(
                accumulated_sum: Shaped[Array, ""], y_block: Data
            ) -> tuple[Array, Array]:
                """Block reduce/accumulate over ``y``."""
                x_, w_x = x_block.data, x_block.weights
                y_, w_y = y_block.data, y_block.weights
                column_sum_slice = jnp.dot(self.compute(x_, y_), w_y)
                accumulated_sum += jnp.dot(w_x, column_sum_slice)
                return accumulated_sum, column_sum_slice

            accumulated_sum, column_sum_slices = jax.lax.scan(
                slice_sum, accumulated_sum, block_y, unroll=inner_unroll
            )
            return accumulated_sum, jnp.sum(column_sum_slices, axis=0)

        accumulated_sum, column_sum_blocks = jax.lax.scan(
            block_sum, jnp.asarray(0.0), block_x, unroll=outer_unroll
        )
        if axis is None:
            return accumulated_sum
        num_rows_padded = block_x.data.shape[0] * block_x.data.shape[1]
        column_sum_padded = column_sum_blocks.reshape(num_rows_padded, -1).sum(axis=1)
        return column_sum_padded[:unpadded_len_x]


class _Constant(ScalarValuedKernel):
    r"""Define a helper class to add additional functionality to magic methods."""

    constant: float = eqx.field(default=1, converter=float)

    def __check_init__(self):
        """Check that the constant is not negative."""
        if self.constant < 0:
            raise ValueError(
                "'constant' must not be negative in order to retain positive"
                + " semi-definiteness"
            )

    @override
    def compute_elementwise(self, x, y):
        return jnp.asarray(self.constant)

    @override
    def grad_x_elementwise(self, x, y):
        return jnp.asarray(0)

    @override
    def grad_y_elementwise(self, x, y):
        return jnp.asarray(0)

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        return jnp.asarray(0)


class UniCompositeKernel(ScalarValuedKernel):
    """
    Abstract base class for kernels that compose/wrap one scalar-valued kernel.

    :param base_kernel: kernel to be wrapped/used in composition
    """

    base_kernel: ScalarValuedKernel

    def __check_init__(self):
        """Check that 'base_kernel' is of the required type."""
        if not isinstance(self.base_kernel, ScalarValuedKernel):
            raise TypeError(
                "'base_kernel' must be an instance of "
                + f"'{ScalarValuedKernel.__module__}.{ScalarValuedKernel.__qualname__}'"
            )


class PowerKernel(UniCompositeKernel, ScalarValuedKernel):
    r"""
    Define a kernel function which is an integer power of a base kernel function.

    Given a kernel function :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`
    define the power kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`
    where :math:`p(x,y) = k(x,y)^n` and :math:`n\in\mathbb{N}`.

    :param base_kernel: Instance of :class:`ScalarValuedKernel`
    :param power:
    """

    power: int

    def __check_init__(self):
        """Check that we have an integer power that is greater than 1."""
        min_power = 1
        if not isinstance(self.power, int) or self.power < min_power:
            raise ValueError(
                "'power' must be a positive integer to ensure positive"
                + " semi-definiteness"
            )

    @override
    def compute_elementwise(self, x, y):
        return self.base_kernel.compute_elementwise(x, y) ** self.power

    @override
    def grad_x_elementwise(self, x, y):
        return (
            self.power
            * self.base_kernel.grad_x_elementwise(x, y)
            * self.base_kernel.compute_elementwise(x, y) ** (self.power - 1)
        )

    @override
    def grad_y_elementwise(self, x, y):
        return (
            self.power
            * self.base_kernel.grad_y_elementwise(x, y)
            * self.base_kernel.compute_elementwise(x, y) ** (self.power - 1)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        n = self.power
        compute = self.base_kernel.compute_elementwise(x, y)
        return n * (
            (
                compute ** (n - 1)
                * self.base_kernel.divergence_x_grad_y_elementwise(x, y)
            )
            + (n - 1)
            * compute ** (n - 2)
            * (
                self.base_kernel.grad_x_elementwise(x, y).dot(
                    self.base_kernel.grad_y_elementwise(x, y)
                )
            )
        )


class PullBackKernel(UniCompositeKernel, ScalarValuedKernel):
    r"""
    Define a kernel function which is a pull-back of a base kernel function.

    Given a kernel function :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`
    and a decoder :math:`\phi:\mathbb{R}^p \to \mathbb{R}^d`, define the pull-back
    kernel :math:`p:\mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}`
    where :math:`p(x,y) = k(\phi(x),\phi(y))`, :math:`x,y \in \mathbb{R}^p`.

    :param base_kernel: Instance of :class:`ScalarValuedKernel`
    :param decoder: Decoder function
    """

    decoder: Callable[[Shaped[Array, " p"]], Shaped[Array, " d"]]

    @override
    def compute_elementwise(self, x, y):
        return self.base_kernel.compute_elementwise(self.decoder(x), self.decoder(y))

    @override
    def grad_x_elementwise(self, x, y):
        return grad(self.compute_elementwise, 0)(x, y)

    @override
    def grad_y_elementwise(self, x, y):
        return grad(self.compute_elementwise, 1)(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        return pairwise(self.divergence_x_grad_y_elementwise)(x, y)


class DuoCompositeKernel(ScalarValuedKernel):
    """
    Abstract base class for kernels that compose/wrap two scalar-valued kernels.

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
    """

    first_kernel: ScalarValuedKernel
    second_kernel: ScalarValuedKernel

    def __check_init__(self):
        """Ensure attributes are instances of Kernel class."""
        if not (
            isinstance(self.first_kernel, ScalarValuedKernel)
            and isinstance(self.second_kernel, ScalarValuedKernel)
        ):
            raise TypeError(
                "'first_kernel'and `second_kernel` must be an instance of "
                + f"'{ScalarValuedKernel.__module__}.{ScalarValuedKernel.__qualname__}'"
            )


class AdditiveKernel(DuoCompositeKernel):
    r"""
    Define a kernel which is a summation of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the additive
    kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) := k(x,y) + l(x,y)`

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
    """

    @override
    def compute_elementwise(self, x, y):
        return self.first_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x, y):
        return self.first_kernel.grad_x_elementwise(
            x, y
        ) + self.second_kernel.grad_x_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x, y):
        return self.first_kernel.grad_y_elementwise(
            x, y
        ) + self.second_kernel.grad_y_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        return self.first_kernel.divergence_x_grad_y_elementwise(
            x, y
        ) + self.second_kernel.divergence_x_grad_y_elementwise(x, y)


class ProductKernel(DuoCompositeKernel, ScalarValuedKernel):
    r"""
    Define a kernel which is a product of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the product kernel
    :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) = k(x,y)l(x,y)`

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
    """

    @override
    def compute_elementwise(self, x, y):
        if self.first_kernel == self.second_kernel:
            return self.first_kernel.compute_elementwise(x, y) ** 2
        return self.first_kernel.compute_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x, y):
        if self.first_kernel == self.second_kernel:
            return (
                2
                * self.first_kernel.grad_x_elementwise(x, y)
                * self.first_kernel.compute_elementwise(x, y)
            )
        return self.first_kernel.grad_x_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.grad_x_elementwise(
            x, y
        ) * self.first_kernel.compute_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x, y):
        if self.first_kernel == self.second_kernel:
            return (
                2
                * self.first_kernel.grad_y_elementwise(x, y)
                * self.first_kernel.compute_elementwise(x, y)
            )
        return self.first_kernel.grad_y_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.grad_y_elementwise(
            x, y
        ) * self.first_kernel.compute_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        if self.first_kernel == self.second_kernel:
            return 2 * (
                self.first_kernel.grad_x_elementwise(x, y).dot(
                    self.first_kernel.grad_y_elementwise(x, y)
                )
                + self.first_kernel.compute_elementwise(x, y)
                * self.first_kernel.divergence_x_grad_y_elementwise(x, y)
            )
        return (
            self.first_kernel.grad_x_elementwise(x, y).dot(
                self.second_kernel.grad_y_elementwise(x, y)
            )
            + self.first_kernel.grad_y_elementwise(x, y).dot(
                self.second_kernel.grad_x_elementwise(x, y)
            )
            + self.first_kernel.compute_elementwise(x, y)
            * self.second_kernel.divergence_x_grad_y_elementwise(x, y)
            + self.second_kernel.compute_elementwise(x, y)
            * self.first_kernel.divergence_x_grad_y_elementwise(x, y)
        )


class TensorProductKernel(eqx.Module):
    r"""
    Define a kernel which is a tensor product of two kernels.

    Given kernel functions :math:`r:\mathcal{X} \times \mathcal{X} \to \mathbb{R}` and
    :math:`l:\mathcal{Y} \times \mathcal{Y} \to \mathbb{R}`, define the tensor product
    kernel :math:`k:(\mathcal{X} \times \mathcal{Y}) \otimes
    (\mathcal{X} \times \mathcal{Y}) \to \mathbb{R}` where
    :math:`k((x_1, y_1),(x_2, y_2)) := r(x_1,x_2)l(y_1,y_2)`

    :param feature_kernel: Instance of :class:`ScalarValuedKernel` acting on
        data :math:`x`
    :param response_kernel: Instance of :class:`ScalarValuedKernel` acting on
        supervision :math:`y`
    """

    feature_kernel: ScalarValuedKernel
    response_kernel: ScalarValuedKernel

    @overload
    def compute(
        self,
        a: tuple[Shaped[Array, " n d"], Shaped[Array, " n p"]],
        b: tuple[Shaped[Array, " n d"], Shaped[Array, " n p"]],
    ) -> Shaped[Array, " n m"]: ...

    @overload
    def compute(  # pyright: ignore[reportOverlappingOverload]
        self,
        a: tuple[
            Union[Shaped[Array, " d"], Shaped[Array, ""]],
            Union[Shaped[Array, " p"], Shaped[Array, ""]],
        ],
        b: tuple[
            Union[Shaped[Array, " d"], Shaped[Array, ""]],
            Union[Shaped[Array, " p"], Shaped[Array, ""]],
        ],
    ) -> Shaped[Array, " 1 1"]: ...

    def compute(
        self,
        a: Union[
            tuple[Shaped[Array, " n d"], Shaped[Array, " n p"]],
            tuple[
                Union[Shaped[Array, " d"], Shaped[Array, ""]],
                Union[Shaped[Array, " p"], Shaped[Array, ""]],
            ],
        ],
        b: Union[
            tuple[Shaped[Array, " m d"], Shaped[Array, " m p"]],
            tuple[
                Union[Shaped[Array, " d"], Shaped[Array, ""]],
                Union[Shaped[Array, " p"], Shaped[Array, ""]],
            ],
        ],
    ) -> Union[Shaped[Array, " n m"], Shaped[Array, "1 1"]]:
        r"""
        Evaluate the kernel on input data.

        The 'data' can be tuples of any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        Evaluation is always vectorised.

        :param a: A tuple with first element consisting of :math:`n \times d` dataset
            (array), and second element consisting of :math:`n \times p` dataset
            (array) or both single values (points)
        :param b: A tuple with first element consisting of :math:`m \times d` dataset
            (array), and second element consisting of :math:`m \times p` dataset
            (array) or both single values (points)
        :return: Kernel evaluations between pairs in `a` and `b`. If `a`:math:`=``b`,
            then this is the Gram matrix corresponding to the tensor-product RKHS inner
            product.
        """
        return pairwise_tuple(self.compute_elementwise)(a, b)

    def compute_elementwise(
        self,
        a: tuple[
            Union[Shaped[Array, " d"], Shaped[Array, ""]],
            Union[Shaped[Array, " p"], Shaped[Array, ""]],
        ],
        b: tuple[
            Union[Shaped[Array, " d"], Shaped[Array, ""]],
            Union[Shaped[Array, " p"], Shaped[Array, ""]],
        ],
    ) -> Shaped[Array, ""]:
        r"""
        Evaluate the kernel on pairs of individual input vectors.

        Vectorisation only becomes relevant in terms of computational speed when we
        have multiple pairs `a`:math:`=(x_1, y_1)`, or `b`:math:`=(x_2, y_2)`.

        :param a: Tuple of vectors :math:`(x_1, y_1)`
        :param b: Tuple of vectors :math:`(x_2, y_2)`
        :return: Kernel evaluated at :math:`((x_1, y_1), (x_2, y_2))`
        """
        return self.feature_kernel.compute_elementwise(
            a[0], b[0]
        ) * self.response_kernel.compute_elementwise(a[1], b[1])

    def gramian_row_mean(
        self,
        a: Union[
            tuple[
                Union[
                    Shaped[Array, " n d"],
                    Shaped[Array, " d"],
                    Shaped[Array, ""],
                ],
                Union[
                    Shaped[Array, " m p"],
                    Shaped[Array, " p"],
                    Shaped[Array, ""],
                ],
            ],
            SupervisedData,
        ],
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute the (blocked) row-mean of the kernel's Gramian matrix.

        A convenience method for calling meth:`compute_mean`. Equivalent to the call
        :code:`compute_mean(a, a, axis=0, block_size=block_size, unroll=unroll)`.

        :param a: Instance of :class:`~coreax.data.SupervisedData` or tuple of two
            arrays where the order of the elements corresponds to inputs to the first
            and second kernels
        :param block_size: Block size parameter passed to :meth:`compute_mean`
        :param unroll: Unroll parameter passed to :meth:`compute_mean`
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        return self.compute_mean(a, a, axis=0, block_size=block_size, unroll=unroll)

    def compute_mean(
        self,
        a: Union[
            tuple[
                Union[
                    Shaped[Array, " n d"],
                    Shaped[Array, " d"],
                    Shaped[Array, ""],
                ],
                Union[
                    Shaped[Array, " n p"],
                    Shaped[Array, " p"],
                    Shaped[Array, ""],
                ],
            ],
            SupervisedData,
        ],
        b: Union[
            tuple[
                Union[
                    Shaped[Array, " m d"],
                    Shaped[Array, " d"],
                    Shaped[Array, ""],
                ],
                Union[
                    Shaped[Array, " m p"],
                    Shaped[Array, " p"],
                    Shaped[Array, ""],
                ],
            ],
            SupervisedData,
        ],
        axis: Union[int, None] = None,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute the mean of the matrix :math:`K_{ij} = k((x_a, y_a)_i, (x_b, y_b)_j))`.

        The :math:`n \times m` tensor-product kernel matrix
        :math:`K_{ij} = k((x_a, y_a)_i, (x_b, y_b)_j)`, where ``a`` and ``b`` are
        instances of :class:`~coreax.data.SupervisedData` containing :math:`n` and
        :math:`m` (weighted) data pairs respectively, has the following
        (weighted) means:

        - mean (:code:`axis=None`) :math:`\frac{1}{n m}\sum_{i,j=1}^{n, m} K_{ij}`
        - row-mean (:code:`axis=0`) :math:`\frac{1}{n}\sum_{i=1}^{n} K_{ij}`
        - column-mean (:code:`axis=1`) :math:`\frac{1}{m}\sum_{j=1}^{m} K_{ij}`

        The weights of `a` and `b` are used to compute the weighted mean as defined in
        :func:`jax.numpy.average`.

        .. note::
            The conventional 'mean' is a scalar, the 'row-mean' is an :math:`m`-vector,
            while the 'column-mean' is an :math:`n`-vector.

        To avoid materializing the entire matrix (memory cost :math:`\mathcal{O}(n m)`),
        we accumulate the mean over blocks (memory cost :math:`\mathcal{O}(B_a B_b)`,
        where ``B_a`` and ``B_b`` are user-specified block-sizes for blocking the ``a``
        and ``b`` parameters respectively.

        .. note::
            The supervised data ``a`` and/or ``b`` are padded with zero-valued and
            zero-weighted data points, when ``B_a`` and/or ``B_b`` are non-integer
            divisors of ``n`` and/or ``m``. Padding does not alter the result, but does
            provide the block shape stability required by :func:`jax.lax.scan`
            (used for block iteration).

        :param a: Instance of :class:`~coreax.data.SupervisedData` containing :math:`n`
            (weighted) data pairs or tuple of two arrays where the order of the elements
            corresponds to inputs to the first and second kernels
        :param b: Instance of :class:`~coreax.data.SupervisedData` containing :math:`m`
            (weighted) data pairs or tuple of two arrays where the order of the elements
            corresponds to inputs to the first and second kernels
        :param axis: Which axis of the kernel matrix to compute the mean over; a value
            of :data:`None` computes the mean over both axes
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: The (weighted) mean of the kernel matrix :math:`K_{ij}`
        """
        datasets = as_supervised_data(a), as_supervised_data(b)
        inner_unroll, outer_unroll = tree_leaves_repeat(unroll, len(datasets))
        _block_size = tree_leaves_repeat(block_size, len(datasets))

        # Row-mean is the argument reversed column-mean due to symmetry k(x,y) = k(y,x)
        if axis == 0:
            datasets = datasets[::-1]
            _block_size = _block_size[::-1]

        def is_supervised_data(x: Any) -> bool:
            """Return boolean indicating if ``x`` is an instance of `SupervisedData`."""
            return isinstance(x, SupervisedData)

        (block_a, unpadded_len_a), (block_b, _) = jtu.tree_map(
            _block_supervised_data_convert,
            datasets,
            tuple(_block_size),
            is_leaf=is_supervised_data,
        )

        def block_sum(
            accumulated_sum: Array, a_block: SupervisedData
        ) -> tuple[Array, Array]:
            """Block reduce/accumulate over ``a``."""

            def slice_sum(
                accumulated_sum: Array, b_block: SupervisedData
            ) -> tuple[Array, Array]:
                """Block reduce/accumulate over ``b``."""
                x_a, y_a, w_a = a_block.data, a_block.supervision, a_block.weights
                x_b, y_b, w_b = b_block.data, b_block.supervision, b_block.weights
                column_sum_slice = jnp.dot(self.compute((x_a, y_a), (x_b, y_b)), w_b)
                accumulated_sum += jnp.dot(w_a, column_sum_slice)
                return accumulated_sum, column_sum_slice

            accumulated_sum, column_sum_slices = jax.lax.scan(
                slice_sum, accumulated_sum, block_b, unroll=inner_unroll
            )
            return accumulated_sum, jnp.sum(column_sum_slices, axis=0)

        accumulated_sum, column_sum_blocks = jax.lax.scan(
            block_sum, jnp.asarray(0.0), block_a, unroll=outer_unroll
        )
        if axis is None:
            return accumulated_sum
        num_rows_padded = block_a.data.shape[0] * block_a.data.shape[1]
        column_sum_padded = column_sum_blocks.reshape(num_rows_padded, -1).sum(axis=1)
        return column_sum_padded[:unpadded_len_a]
