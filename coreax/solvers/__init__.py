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

"""Solvers for generating coresets."""

from coreax.solvers.autoencoders import (
    Autoencoder,
    BaseCoder,
    Decoder,
    Encoder,
    ImageDecoder,
    ImageEncoder,
)
from coreax.solvers.base import (
    CoresetSolver,
    CoresubsetSolver,
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
    Solver,
)
from coreax.solvers.bilateral import (
    LinearBilateralDistributionCompression,
    NonLinearBilateralDistributionCompression,
)
from coreax.solvers.bilateral_supervised import (
    ExhaustiveSearch,
    SupervisedLinearBilateralDistributionCompression,
    SupervisedNonLinearBilateralDistributionCompression,
)
from coreax.solvers.composite import CompositeSolver, MapReduce
from coreax.solvers.coreset import (
    AverageConditionalKernelHerding,
    ConditionalKernelHerding,
    ExactAverageConditionalKernelHerding,
    ExactConditionalKernelHerding,
    ExactPseudoJointKernelHerding,
    HerdingExhaustiveSearch,
    PseudoJointKernelHerding,
)
from coreax.solvers.coreset_kip import (
    AverageConditionalKIP,
    ConditionalKIP,
    ExactAverageConditionalKIP,
    ExactConditionalKIP,
    ExactJointKIP,
    JointKIP,
    KernelInducingPoints,
    KIPExhaustiveSearch,
    KIPState,
)
from coreax.solvers.coresubset import (
    GreedyKernelPoints,
    GreedyKernelPointsState,
    HerdingState,
    JointKernelHerding,
    KernelHerding,
    RandomSample,
    RPCholesky,
    RPCholeskyState,
    SteinThinning,
)
from coreax.solvers.distillation import M3D
from coreax.solvers.recombination import (
    CaratheodoryRecombination,
    RecombinationSolver,
    TreeRecombination,
)

__all__ = [
    "Solver",
    "KIPState",
    "CoresubsetSolver",
    "RefinementSolver",
    "ExplicitSizeSolver",
    "PaddingInvariantSolver",
    "CompositeSolver",
    "MapReduce",
    "RandomSample",
    "HerdingState",
    "KernelHerding",
    "SteinThinning",
    "RPCholeskyState",
    "RPCholesky",
    "GreedyKernelPointsState",
    "GreedyKernelPoints",
    "RecombinationSolver",
    "CaratheodoryRecombination",
    "TreeRecombination",
    "JointKernelHerding",
    "AverageConditionalKernelHerding",
    "ConditionalKernelHerding",
    "PseudoJointKernelHerding",
    "ExactConditionalKernelHerding",
    "ExactAverageConditionalKernelHerding",
    "ConditionalKIP",
    "AverageConditionalKIP",
    "ExactAverageConditionalKIP",
    "ExactConditionalKIP",
    "ExactPseudoJointKernelHerding",
    "ExactJointKIP",
    "JointKIP",
    "HerdingExhaustiveSearch",
    "KIPExhaustiveSearch",
    "KernelInducingPoints",
    "CoresetSolver",
    "LinearBilateralDistributionCompression",
    "SupervisedLinearBilateralDistributionCompression",
    "ExhaustiveSearch",
    "Autoencoder",
    "Decoder",
    "Encoder",
    "ImageDecoder",
    "ImageEncoder",
    "NonLinearBilateralDistributionCompression",
    "M3D",
    "SupervisedNonLinearBilateralDistributionCompression",
    "BaseCoder",
]
