from typing import Callable, Optional, Dict, List
from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from torch import Tensor

from tumorpde.models.comp_utils import _fd_auxiliaries, half_points_eval
from tumorpde.models.comp_utils import _get_slicing_positions
from tumorpde.models.comp_utils import phase_field
from tumorpde.volume_domain import VolumeDomain
from tumorpde._typing import TensorLikeFloat
from tumorpde.calc.dist_boundary import signed_distance_and_nearest_index

from tumorpde.calc.loss import empty_loss_deriv, mse_loss, deriv_mse_loss


def default_init_density(x: Tensor, x0: Tensor,
                         w: float, h: float, rmax: float = 1.) -> Tensor:
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
    return density_value.to(device=x.device, dtype=x.dtype)


def default_init_density_deriv(
        x: Tensor, x0: Tensor, w: float, h: float, rmax: float = 3.) -> Tensor:
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
    deriv = 2 / w**2 * diff * density_value
    return deriv.to(device=x.device, dtype=x.dtype)


def empty_init_density_deriv(
        x: Tensor, params: Tensor) -> Tensor:
    """
    Args:
        x: Tensor, the input tensor.
        params: Tensor, phandom parameters.
    Returns:
        Tensor, an empty tensor of shape (0, *voxel_shape).
    """
    return torch.empty((0, *x.shape), device=x.device, dtype=x.dtype)


def default_loss_func(x: Tensor, y: Tensor) -> Tensor:
    return mse_loss(x, y)

def default_loss_deriv(x: Tensor, y: Tensor) -> Tensor:
    return deriv_mse_loss(x, y)


class TumorBase(ABC):

    def __init__(self, domain: VolumeDomain,
                 init_density_func: Optional[Callable] = None,
                 init_learnable_params: Optional[TensorLikeFloat] = None,
                 init_other_params: Optional[Dict] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_param_min: Optional[TensorLikeFloat] = None,
                 init_param_max: Optional[TensorLikeFloat] = None,
                 loss_func: Optional[Callable] = None,
                 loss_other_params: Optional[Dict] = None,
                 loss_deriv: Optional[Callable] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """
        Args:

        domain:
            VolumeDomain, the domain of the tumor growth model.
        init_density_func:
            callable, the initial density function,
            arg1: evaluation grid, arg2: learnable parameters, **kwargs: other parameters
            (*voxel_shape) -> (*voxel_shape)
        init_learnable_params:
            floats, learnable parameters in the initial density function.
        init_other_params:
            dict, the parameters of the initial density function.
        init_density_deriv:
            callable, the derivative of the initial density function w.r.t. the learnable parameters,
            can be None if no learnable parameters are included,
            (*voxel_shape) -> (len(init_params), *voxel_shape).
        init_param_min, init_param_max:
            list[float], min/max bound of the parameters. Set to None is no parameters in the
            init_density_func are learnable -> (len(init_params)).
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
        self.n_points = int(math.prod(list(self.nx)))
        self.x_mesh = torch.stack(torch.meshgrid(
            *self.x, indexing="ij"))  # (ndim, *nx)
        self.dx = torch.as_tensor(self.domain.voxel_widths, **factory_args)
        self.idx2 = 1 / (self.dx ** 2)
        self.domain_mask = torch.as_tensor(domain.voxel > 0.,
                                           dtype=torch.bool, device=device)
        self.n_in_points = int(self.domain_mask.sum().item())  # interior points

        # Torch factory settings
        self.factory_args = factory_args
        self.dtype = dtype
        self.device = device

        # Using autograd for param estimation?
        self.auto_grad = autograd

        # Set init condition functions
        self.compile_init(init_density_func,
                          init_learnable_params,
                          init_other_params,
                          init_density_deriv,
                          init_param_min,
                          init_param_max)

        # Set loss functions
        self.compile_loss(loss_func, loss_other_params, loss_deriv)

        # Parameters
        # indexes
        # 0:nparam_init
        # nparam_init:nparam_init+nparam_loss
        self.param_names = []  # parameters used to define PDEs
        self.nparam_main = len(self.param_names)
        self.parameters = self._format_parameters(init_params=self.init_learnable_params)
        self.rescale_params = False
        self.params_min = self.init_param_min.view(-1).clone()
        self.params_max = self.init_param_max.view(-1).clone()
        self.params_scale = (self.init_param_max - self.init_param_min).view(-1).clone()

    def compile_init(
        self,
        init_density_func: Optional[Callable] = None,
        init_learnable_params: Optional[TensorLikeFloat] = None,
        init_other_params: Optional[Dict] = None,
        init_density_deriv: Optional[Callable] = None,
        init_param_min: Optional[TensorLikeFloat] = None,
        init_param_max: Optional[TensorLikeFloat] = None,
        deriv_free: bool = False
    ):
        # Initial state and its parameter settings
        if init_density_func is None:
            # default: Gaussian density concentrated around an initial point
            # The initial location is learned through optimization
            self.init_density = default_init_density
            self.init_density_deriv = default_init_density_deriv if not deriv_free else empty_init_density_deriv
            if init_learnable_params is None:
                init_learnable_params = self.domain.bbox_widths / 2
            self.init_learnable_params = \
                torch.tensor(init_learnable_params, **self.factory_args, requires_grad=self.auto_grad)
            assert self.init_learnable_params.numel() == self.dim
            if init_other_params is not None:
                assert isinstance(init_other_params, dict)
                if not all(key in init_other_params for key in ['w', 'h', 'rmax']):
                    raise ValueError("w, h, rmax are required for the default initial density function.")
                self.init_other_params = dict(init_other_params)
            else:
                self.init_other_params = {
                    "w": float((self.domain.bbox_widths / 20).mean()),
                    "h": 0.01, "rmax": 3. * self.dim
                }
            self.init_param_min = self.xmin.to(**self.factory_args)
            self.init_param_max = self.xmax.to(**self.factory_args)
            self.nparam_init = self.dim
        else:
            self.init_density = init_density_func
            if init_learnable_params is None:
               init_learnable_params = []
            self.init_learnable_params = torch.tensor(
                init_learnable_params, **self.factory_args, requires_grad=self.auto_grad)
            self.nparam_init = self.init_learnable_params.numel()
            self.init_other_params = dict(init_other_params) if init_other_params is not None else {}
            self.init_param_min = torch.as_tensor(init_param_min, **self.factory_args) \
                  if init_param_min is not None else torch.tensor([], **self.factory_args)
            self.init_param_max = torch.as_tensor(init_param_max, **self.factory_args) \
                  if init_param_max is not None else torch.tensor([], **self.factory_args)
            if self.nparam_init != 0:
                if self.init_param_min.numel() != self.nparam_init or \
                    self.init_param_max.numel() != self.nparam_init:
                    raise ValueError("Length of init_param_min/mas should equal the number of learnable init parameters")
            if init_density_deriv is None:
                if (not deriv_free) and (self.nparam_init != 0):
                    raise ValueError("""init_density_deriv cannot be None if
                                     learnable parameters exist and derivative-based algorithm is used""")
                else:
                    self.init_density_deriv = empty_init_density_deriv
            else:
                self.init_density_deriv = init_density_deriv
    
    def compile_loss(self,
        loss_func: Optional[Callable] = None,
        loss_other_params: Optional[Dict] = None,
        loss_deriv: Optional[Callable] = None
    ):
        if loss_func is None:
            # default choice: MSE loss
            self.loss_func = default_loss_func
            self.loss_deriv = default_loss_deriv
            self.loss_other_params = {}
        else:
            self.loss_func = loss_func
            if loss_deriv is None:
                raise ValueError("loss_deriv cannot be None if loss_func is provided")
            else:
                self.loss_deriv = loss_deriv
            self.loss_other_params = dict(loss_other_params) if loss_other_params is not None else {}

    def _format_parameters(self, **kwargs) -> Tensor:
        """Helper function to format parameters into a vector"""
        params = []
        for name, value in kwargs.items():
            if name not in self.param_names and name != "init_params":
                raise ValueError(f"Parameter '{name}' is not a valid parameter name")
            if value is None:
                params.append(getattr(self, name))
            elif not isinstance(value, (Tensor, np.ndarray, float)) and \
               not ((isinstance(value, (list, tuple)) and all(isinstance(x, float) for x in value))):
                raise TypeError(f"Parameter '{name}' is not a torch.Tensor or a list of float")
            else:
                params.append(torch.as_tensor(value, **self.factory_args).reshape(-1))
        
        # Concatenate along the only dimension
        if params:
            return torch.cat([p.view(-1) for p in params], dim=0)
        else:
            # No tensors passed: return an empty tensor
            return torch.tensor([], **self.factory_args)

    @abstractmethod
    def _set_parameters(self, params: Tensor):
        """Set self.parameters, its min, max, and scale"""

    @property
    def nparam(self):
        return self.nparam_main + self.nparam_init

    @property
    def init_params(self):
        return self._init_params(self.parameters)

    @abstractmethod
    def _init_params(self, params: Tensor) -> Tensor:
        """Extract the intial parameters from all params"""

    def _update_parameters(self, params: Tensor) -> None:
        """ Update the parameters. """
        assert params.numel() == self.nparam
        self.parameters = params.view(-1)

    def _to_working_params(self, params: Tensor) -> Tensor:
        """ Convert params from original scale to the working scale"""
        wkparams = (params - self.params_min) / self.params_scale
        return wkparams

    def _to_original_params(self, wkparams: Tensor) -> Tensor:
        """ Convert working parameters to their original scale"""
        params = wkparams * self.params_scale + self.params_min
        return params

    def _clip_state(self, u: Tensor) -> None:
        u.clamp_(0.0, 1.0)

class TumorFixedFieldBase(TumorBase):

    def __init__(self, domain: VolumeDomain,
                 init_density_func: Optional[Callable] = None,
                 init_learnable_params: Optional[TensorLikeFloat] = None,
                 init_other_params: Optional[Dict] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_param_min: Optional[TensorLikeFloat] = None,
                 init_param_max: Optional[TensorLikeFloat] = None,
                 loss_func: Optional[Callable] = None,
                 loss_other_params: Optional[Dict] = None,
                 loss_deriv: Optional[Callable] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """
        Args: Same as TumorBase
        """
        super().__init__(domain,
                         init_density_func,
                         init_learnable_params,
                         init_other_params,
                         init_density_deriv,
                         init_param_min, init_param_max,
                         loss_func, loss_other_params, loss_deriv,
                         autograd, dtype, device)

        self.diff_map = torch.as_tensor(domain.voxel, **self.factory_args)
        self.diff_map_aux = _fd_auxiliaries(self.diff_map)


# TODO: TumorVarFieldBase, allow an undetermined number of brain tissues

class TumorVarFieldBase(TumorBase):

    def __init__(self,
                 domain: VolumeDomain,
                 matters: Tensor,
                 init_density_func: Optional[Callable] = None,
                 init_learnable_params: Optional[TensorLikeFloat] = None,
                 init_other_params: Optional[Dict] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_param_min: Optional[TensorLikeFloat] = None,
                 init_param_max: Optional[TensorLikeFloat] = None,
                 loss_func: Optional[Callable] = None,
                 loss_other_params: Optional[Dict] = None,
                 loss_deriv: Optional[Callable] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            Domain: typically a skull mask instead of a diffusion field
            matters: three brain matters (gm, wm, csf) with deformable shapes
        Other parameters are the same
        """
        super().__init__(domain,
                         init_density_func, init_learnable_params,
                         init_other_params, init_density_deriv,
                         init_param_min, init_param_max,
                         loss_func, loss_other_params, loss_deriv,
                         autograd, dtype, device)

        self.matters = torch.as_tensor(matters, dtype=dtype)
        assert self.matters.shape[0] == 3
        assert self.matters.shape[1:] == torch.Size(self.domain.voxel_shape)

        # NOTE: current phase field function does not support anisotropic voxel widths
        self.phase_field = torch.tensor(phase_field(self.domain.voxel, 2), **self.factory_args)
        self.phase_aux = half_points_eval(self.phase_field)

        # indexing all grid points
        self.indices = torch.arange(
            self.n_points, dtype=torch.long, device=device).reshape(*self.nx)

        # indexing in-domain grid points, -1 for out-of-domain points
        self.interior_indices = -torch.ones(*self.nx, dtype=torch.long, device=device)
        self.interior_indices[self.domain_mask] = torch.arange(
            self.domain_mask.sum().item()).to(dtype=torch.long, device=device)

        # boundary mask and the coordinates
        self.boundary_mask = torch.tensor(self.domain.boundary_mask, dtype=torch.bool, device=device)
        self.boundary_coords = torch.where(self.boundary_mask)

        # nearest coordinates for each boundary voxel
        _, nearest_indices = signed_distance_and_nearest_index(self.domain.voxel)
        nearest_coords = np.unravel_index(nearest_indices, self.domain.voxel.shape)
        self.boundary_nearest_coords = tuple(
            torch.as_tensor(coords, dtype=torch.long, device=device)[self.boundary_mask] \
                for coords in nearest_coords
        )

        # Is the boundary voxel on the "left" or "right" of its in-domain neighbour
        self.boundary_direc = tuple(
            torch.sign(bnd_coord - in_coord) for bnd_coord, in_coord in \
                zip(self.boundary_coords, self.boundary_nearest_coords)
        )

    def pad_boundary(self, u: Tensor) -> None:
        """
        Pad the boundary of the input tensor u with the values of its nearest in-domain neighbour.
        Args:
            u: Tensor, the input tensor.
        Returns:
            Tensor, the padded tensor.
        """
        assert u.shape == self.nx
        u[self.boundary_coords] = u[self.boundary_nearest_coords]
