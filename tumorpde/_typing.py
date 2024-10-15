from numpy import ndarray
from torch import Tensor

from typing import Union, List

TensorLike = Union[ndarray, Tensor]
TensorLikeFloat = Union[List[float], TensorLike]
NDArrayLikeFloat = Union[ndarray, List[float]]
FloatLike = Union[float, int, ndarray[float]]  # ndarray must have shape (1,), but we cannot specify it
