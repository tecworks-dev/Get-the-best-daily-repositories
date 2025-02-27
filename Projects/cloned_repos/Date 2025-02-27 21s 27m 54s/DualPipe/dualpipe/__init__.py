__version__ = "1.0.0"

from dualpipe.dualpipe import (
    DualPipe,
    WeightGradStore,
)
from dualpipe.comm import (
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
)

__all__ = [
    DualPipe,
    WeightGradStore,
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
]
