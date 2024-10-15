
from typing import Optional, Callable, Dict, Any, Literal, Tuple, List

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from scipy.optimize import minimize
from tqdm import tqdm

from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike
from tumorpde.volume_domain import VolumeDomain
from tumorpde.models._base import TumorBaseModel

class TumorDeformFD(TumorBaseModel):

    def __init__(self, domain: VolumeDomain, x0: TensorLikeFloat = None,
                 D: TensorLikeFloat = 1., rho: TensorLikeFloat = 1.,
                 M: TensorLikeFloat = 1., kappa: TensorLikeFloat = 1.,
                 init_density_func: Optional[Callable] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):

        # TODO: Lipkova recommends using a scientifically meaningful value for M
        
        super().__init__(domain, x0, D, rho,
                         init_density_func, init_density_deriv, init_density_params,
                         False, dtype, device)

        self.M = torch.as_tensor(M, **self.factory_args)
        self.kappa = torch.as_tensor(kappa, **self.factory_args)
        self._M_old = None
        self._kappa_old = None

    @property
    def M(self):
        return self._M

    def _format_M(self, value: TensorLikeFloat) -> Tensor:
        value = torch.as_tensor(value, **self.factory_args)
        assert value.ndim == 0 or len(value) == 1
        return value.squeeze()

    @M.setter
    def M(self, value: TensorLikeFloat):
        self._M = self._format_M(value)

    @property
    def kappa(self):
        return self._kappa

    def _format_kappa(self, value: TensorLikeFloat) -> Tensor:
        value = torch.as_tensor(value, **self.factory_args)
        assert value.ndim == 0 or len(value) == 1
        return value.squeeze()

    @kappa.setter
    def kappa(self, value: TensorLikeFloat):
        self._kappa = self._format_kappa(value)

