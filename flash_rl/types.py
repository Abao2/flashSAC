from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

try:
    import jax.numpy as jnp

    JaxArray = jnp.ndarray
except ModuleNotFoundError:  # JAX is unnecessary for the torch/Isaac backends.
    JaxArray = Any

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, JaxArray, torch.Tensor]
