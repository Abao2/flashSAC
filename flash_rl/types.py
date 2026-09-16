from typing import TYPE_CHECKING, Any, Union

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from jax import Array as JaxArray
else:
    try:
        from jax import Array as JaxArray
    except ModuleNotFoundError:  # JAX is unnecessary for the torch/Isaac backends.
        JaxArray = Any

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, JaxArray, torch.Tensor]
