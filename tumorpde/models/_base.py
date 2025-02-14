from typing import Callable, Optional, Dict, List

import torch
from torch import Tensor

from tumorpde.models.comp_utils import _diff_map_auxiliaries
from tumorpde.volume_domain import VolumeDomain


def default_init_density(x: Tensor, x0: Tensor, w: float, h: float, rmax: float = 1.,
                         log_transform: bool = False, eps: float = 1e-6) -> Tensor:
    """ 
    Args:
        x: Tensor, the input tensor.
        x0: Tensor, the initial position of the tumor.
        w: float, the width of the Gaussian.
        h: float, the height of the Gaussian.
        rmax: float, the maximum radius of the Gaussian.
        log_transform: bool, whether to apply a logit transform to the density.
        eps: float, the epsilon value for the logit transform.
    Returns:
        Tensor, the density value, shape (*voxel_shape).
    """
    dim = len(x0)
    assert x.shape[0] == dim
    diff = x - x0.view(dim, *[1] * (x.ndim - 1))
    diff = x - x0.view(dim, *[1] * (x.ndim - 1))
    r2 = torch.sum(diff ** 2, dim=0)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= dim * rmax**2)
    if log_transform:
        density_value = (1 - eps) * density_value + 0.5 * eps
        density_value = torch.logit(density_value)
    return density_value.to(device=x.device, dtype=x.dtype)


def default_init_density_deriv(
        x: Tensor, x0: Tensor, w: float, h: float, rmax: float = 3.,
        to_x0: bool = False) -> Tensor:
    """ 
    Args:
        x: Tensor, the input tensor.
        x0: Tensor, the initial position of the tumor.
        w: float, the width of the Gaussian.
        h: float, the height of the Gaussian.
        rmax: float, the maximum radius of the Gaussian.
        to_x0: bool, whether to return the derivative with respect to x0.
    Returns:
        Tensor, the derivative of the density with respect to x0, shape (dim, *voxel_shape).
    """
    dim = len(x0)
    assert x.shape[0] == dim
    diff = x - x0.view(dim, *[1] * (x.ndim - 1))
    r2 = torch.sum(diff ** 2, dim=0)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= dim * rmax**2)
    deriv = (-2 / w**2) * diff * density_value
    if to_x0:
        deriv = -deriv
    return deriv.to(device=x.device, dtype=x.dtype)


class TumorBase:

    def __init__(self, domain: VolumeDomain,
                 init_density_func: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 init_learnable_params: Optional[List[str]] = None,
                 init_density_deriv: Optional[Callable] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """ 
        Args:

        domain:
            VolumeDomain, the domain of the tumor growth model.
        init_density_func:
            callable, the initial density function, (*voxel_shape) -> (*voxel_shape)
        init_density_params:
            dict, the parameters of the initial density function.
        init_learnable_params:
            list[str], names of the learnable parameters in init_density_params
        init_density_deriv:
            callable, the derivative of the initial density function, (*voxel_shape) -> (len(init_params), *voxel_shape).
        autograd:
            bool, whether to use autograd.
        dtype:
            torch.dtype, the data type of the parameters.
        device:
            torch.device, the device of the parameters.
        """
        factory_args = {
            "dtype": dtype,
            "device": device,
        }

        # Domain specs
        self.domain = domain
        self.dim = domain.dim
        self.xmin = torch.as_tensor(self.domain.xmin, **factory_args)
        self.xmax = torch.as_tensor(self.domain.xmax, **factory_args)
        self.x = tuple(torch.as_tensor(xi, **factory_args)
                       for xi in self.domain.voxel_marginal_coords)
        self.nx = torch.Size([len(xi) for xi in self.x])
        self.x_mesh = torch.stack(torch.meshgrid(
            *self.x, indexing="ij"))  # (ndim, *nx)
        self.dx = torch.as_tensor(self.domain.voxel_widths, **factory_args)
        self.idx2 = 1 / (self.dx ** 2)
        self.domain_mask = torch.as_tensor(domain.voxel > 0., **factory_args)

        # Torch factory settings
        self.factory_args = factory_args

        # Using autograd for param estimation?
        self.auto_grad = autograd

        # Init_density
        if init_density_func is None:
            self.init_density = default_init_density
            self.init_density_deriv = default_init_density_deriv
            if init_density_params is not None:
                if not all(key in init_density_params for key in ['x0', 'w', 'h']):
                    raise ValueError("x0, w, h are required for the default initial density function.")
                self.init_density_params = init_density_params
            else:
                self.init_density_params = {
                    "x0": torch.tensor(self.domain.bbox_widths / 2, **self.factory_args, requires_grad=autograd),
                    "w": float((self.domain.bbox_widths / 20).mean()),
                    "h": 0.01, "rmax": 3. * self.dim
                }
            self.init_learnable_params = ["x0"]
        else:
            self.init_density = init_density_func
            self.init_density_deriv = init_density_deriv
            self.init_density_params = init_density_params if init_density_params is not None else {}
            self.init_learnable_params = init_learnable_params if init_learnable_params is not None else []
            for param_key in self.init_learnable_params:
                if self.init_density_params.get(param_key) is None:
                    raise ValueError(f"Learnable parameter {param_key} not found in init_density_params.")
                self.init_density_params[param_key] = torch.tensor(
                    self.init_density_params[param_key], **self.factory_args, requires_grad=autograd)

        self.nparam_init = sum(self.init_density_params[k].numel() for k in self.init_learnable_params)
    
    def _cat_init_params(self) -> Tensor:
        return torch.cat(
            [self.init_density_params[k].view(-1) for k in self.init_learnable_params], dim=0)

    def _clip_state(self, u: Tensor) -> None:
        u.clamp_(0.0, 1.0)

    def _loss_func(self, u: Tensor, obs: Tensor) -> Tensor:
        # TODO: allow logit type loss
        return 0.5 * torch.mean((u - obs)**2)


class TumorFixedFieldBase(TumorBase):

    def __init__(self, domain: VolumeDomain,
                 init_density_func: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 init_learnable_params: Optional[List[str]] = None,
                 init_density_deriv: Optional[Callable] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """ 
        Args:
            domain: VolumeDomain, the domain of the tumor growth model.
            init_density_func: callable, the initial density function.
            init_density_deriv: callable, the derivative of the initial density function.
            init_density_params: dict, the parameters of the initial density function.
            autograd: bool, whether to use autograd.
            dtype: torch.dtype, the data type of the parameters.
            device: torch.device, the device of the parameters.
        """
        super().__init__(domain, init_density_func, init_density_params,
                         init_learnable_params, init_density_deriv,
                         autograd, dtype, device)

        self.diff_map = torch.as_tensor(domain.voxel, **self.factory_args)
        self.diff_map_aux = _diff_map_auxiliaries(self.diff_map)
