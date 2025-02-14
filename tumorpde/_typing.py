from numpy.typing import ArrayLike 
from torch import Tensor

from typing import Union, List

TensorLike = Union[ArrayLike, Tensor]
TensorLikeFloat = Union[List[float], TensorLike]
NDArrayLikeFloat = Union[ArrayLike, List[float]]
FloatLike = Union[float, int, ArrayLike]  # ndarray must have shape (1,), but we cannot specify it
