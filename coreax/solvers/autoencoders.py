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

"""Classes for autoencoders."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from coreax.util import KeyArrayLike


class BaseCoder(eqx.Module):
    """Abstract base class for en/decoder."""

    layers: list

    def __call__(self, x):
        """Call encoder."""
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(BaseCoder):
    """Encoder."""

    num_hidden_layers: int

    def __init__(
        self,
        random_key: KeyArrayLike,
        ambient_dimension: int,
        intrinsic_dimension: int,
        num_hidden_layers: int,
        hidden_layer_sizes: list,
    ):
        """Initialise encoder."""
        assert (
            len(hidden_layer_sizes) == num_hidden_layers
        ), "Number of hidden layers should equal number of provided layer sizes."
        input_key, output_key = jax.random.split(random_key)
        layers = [
            eqx.nn.Linear(
                ambient_dimension,
                hidden_layer_sizes[0],
                key=input_key,
            ),
            jax.nn.relu,
        ]

        hidden_keys = jax.random.split(input_key, num_hidden_layers)
        for i in range(num_hidden_layers):
            if i == num_hidden_layers - 1:
                layers.append(
                    eqx.nn.Linear(
                        hidden_layer_sizes[i],
                        intrinsic_dimension,
                        key=output_key,
                    )
                )
            else:
                layers.append(
                    eqx.nn.Linear(
                        hidden_layer_sizes[i],
                        hidden_layer_sizes[i + 1],
                        key=hidden_keys[i],
                    )
                )
                layers.append(jax.nn.relu)
        self.layers = layers
        self.num_hidden_layers = num_hidden_layers


class Decoder(BaseCoder):
    """Decoder."""

    num_hidden_layers: int

    def __init__(
        self,
        random_key: KeyArrayLike,
        ambient_dimension: int,
        intrinsic_dimension: int,
        num_hidden_layers: int,
        hidden_layer_sizes: list,
        output_transformation: Callable = jax.nn.identity,
    ):
        """Initialise Decoder."""
        assert (
            len(hidden_layer_sizes) == num_hidden_layers
        ), "Number of hidden layers should equal number of provided layer sizes."

        input_key, output_key = jax.random.split(random_key)
        layers = [
            eqx.nn.Linear(
                intrinsic_dimension,
                hidden_layer_sizes[0],
                key=input_key,
            ),
            jax.nn.relu,
        ]

        hidden_keys = jax.random.split(input_key, num_hidden_layers)
        for i in range(num_hidden_layers):
            if i == num_hidden_layers - 1:
                layers.append(
                    eqx.nn.Linear(
                        hidden_layer_sizes[i],
                        ambient_dimension,
                        key=output_key,
                    )
                )
            else:
                layers.append(
                    eqx.nn.Linear(
                        hidden_layer_sizes[i],
                        hidden_layer_sizes[i + 1],
                        key=hidden_keys[i],
                    )
                )
                layers.append(jax.nn.relu)

        # Add output layer
        layers.append(output_transformation)
        self.layers = layers
        self.num_hidden_layers = num_hidden_layers


class ImageEncoder(BaseCoder):
    """Image Encoder."""

    layers: list

    def __init__(self, intrinsic_dimension, key):
        """Initialise Image Encoder."""
        keys = jr.split(key, 4)

        self.layers = [
            lambda x: jnp.reshape(x, (1, 28, 28)),
            eqx.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, key=keys[0]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2, 2),
            eqx.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, key=keys[1]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2, 2),
            jnp.ravel,
            eqx.nn.Linear(64 * 7 * 7, 128, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Linear(128, intrinsic_dimension, key=keys[3]),
        ]


class ImageDecoder(BaseCoder):
    """Image Decoder."""

    def __init__(self, intrinsic_dimension, key):
        """Initialise Image Decoder."""
        keys = jr.split(key, 4)

        self.layers = [
            eqx.nn.Linear(intrinsic_dimension, 128, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(128, 64 * 7 * 7, key=keys[1]),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (64, 7, 7)),
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, key=keys[2]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, key=keys[3]),
            lambda x: jnp.reshape(x, (784)),
        ]


class Autoencoder(eqx.Module):
    """Autoencoder."""

    encoder: BaseCoder
    decoder: BaseCoder

    def __call__(self, x):
        """Call autoencoder."""
        return self.decoder(self.encoder(x))
