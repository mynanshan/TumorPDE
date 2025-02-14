from typing import Optional, Callable, Dict, Any, Literal, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from scipy.optimize import minimize
from tqdm import tqdm

from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike
from tumorpde.volume_domain import VolumeDomain
from tumorpde.models._base import TumorFixedFieldBase
from tumorpde.models.comp_utils import _get_slicing_positions, _nabla_d_nabla_f
from tumorpde.models.specs import _spec_dt

# TODO:
# - allow a user-specified initial density func with learnable params defined inside
# - remove built-in support for learning x0
# - user needs to provide a list of external learnable parameters

class TumorInfiltraFD(TumorFixedFieldBase):

    def __init__(self, domain: VolumeDomain,
                 D: TensorLikeFloat = 1., rho: TensorLikeFloat = 1.,
                 init_density_func: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 init_learnable_params: Optional[List[str]] = None,
                 init_density_deriv: Optional[Callable] = None,
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

        super().__init__(domain,
                         init_density_func, init_density_params,
                         init_learnable_params, init_density_deriv,
                         False, dtype, device)

        # Parameters
        self.nparam_main = 2
        self.nparam = self.nparam_main + self.nparam_init
        self.model_param_specs = {
            'D': {
                'id': 0,
                'min': torch.tensor(0.).to(**self.factory_args),
                'max': torch.tensor(torch.inf).to(**self.factory_args),
                'log_avail': True,
                'scaling': 'current',
                'centering': 'current'
            },
            'rho': {
                'id': 1,
                'min': torch.tensor(0.).to(**self.factory_args),
                'max': torch.tensor(torch.inf).to(**self.factory_args),
                'log_avail': True,
                'scaling': 'current',
                'centering': 'current'
            }
        }
        # ,
        # 'x0': {
        #     'id': list(range(2, 2+self.dim)),
        #     'min': self.xmin.to(**self.factory_args),
        #     'max': self.xmax.to(**self.factory_args),
        #     'log_avail': False,
        #     'scaling': (0.5 * (self.xmax - self.xmin).to(
        #         **self.factory_args)),
        #     'centering': (0.5 * (self.xmax - self.xmin).to(
        #         **self.factory_args))
        # }

        # x0 = torch.as_tensor(self.domain.bbox_widths / 2, **self.factory_args) \
        #     if x0 is None else torch.as_tensor(x0, **self.factory_args)
        D = torch.as_tensor(D, **self.factory_args)
        rho = torch.as_tensor(rho, **self.factory_args)
        self.parameters = torch.cat(
            [D.view(-1), rho.view(-1)], dim=0)
        self.parameters = torch.cat(
            [D.view(-1), rho.view(-1), x0.view(-1)], dim=0)

        # Settings for the training process
        # parameters with proper resizing are called working parameters
        self.log_transform = False
        self.log_transform_id = []
        self.log_mask = None
        self.rescale_params = False
        self.wkparams_scale = None
        self.wkparams_center = None
        self.wkparams_min = None
        self.wkparams_max = None

    @property
    def D(self):
        return self._D(self.parameters)

    def _D(self, params):
        return params[self.model_param_specs['D']['id']]

    @property
    def rho(self):
        return self._rho(self.parameters)

    def _rho(self, params):
        return params[self.model_param_specs['rho']['id']]

    # @property
    # def x0(self):
    #     return self._x0(self.parameters)

    # def _x0(self, params):
    #     return params[self.model_param_specs['x0']['id']]

    def _format_parameters(self, D: Optional[TensorLikeFloat] = None,
                           rho: Optional[TensorLikeFloat] = None,
                           x0: Optional[TensorLikeFloat] = None) -> Tensor:
        """ Helper function to format a new set parameters. 
        Ease the trial of new parameters. If not available,
        return the current parameters. """
        D = self.D if D is None else torch.as_tensor(D, **self.factory_args)
        rho = self.rho if rho is None else torch.as_tensor(rho, **self.factory_args)
        x0 = self.x0 if x0 is None else torch.as_tensor(x0, **self.factory_args)
        params = torch.cat(
            [D.view(-1), rho.view(-1), x0.view(-1)], dim=0)
        assert params.numel() == self.nparam
        return params

    def _update_parameters(self, params: Tensor) -> None:
        """ Update the parameters. """
        assert params.numel() == self.nparam
        self.parameters = params.view(-1)

    # def _update_parameters(self, D: Optional[Tensor] = None, rho: Optional[Tensor] = None,
    #                        x0: Optional[Tensor] = None) -> None:
    #     """ Update the parameters. """
    #     if D is not None:
    #         self._D_old = self.D
    #         self.D = D
    #     if rho is not None:
    #         self._rho_old = self.rho
    #         self.rho = rho
    #     if x0 is not None:
    #         self._x0_old = self.x0
    #         self.x0 = x0

    def _get_wkparam_settings(
        self, log_transform: bool = False,
        rescale_params: bool = False,
        eps = 1e-6
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tuple[Tensor, Tensor]]:
        if not rescale_params:
            params_scale = torch.ones(self.nparam, **self.factory_args)
            params_center = torch.zeros(self.nparam, **self.factory_args)
        else:
            params_scale = torch.empty(self.nparam, **self.factory_args)
            params_center = torch.empty(self.nparam, **self.factory_args)
        params_min = torch.empty(self.nparam, **self.factory_args)
        params_max = torch.empty(self.nparam, **self.factory_args)

        def get_val(model, key, value):
            if value == 'current':
                return getattr(model, key)
            assert isinstance(value, Tensor)
            return value

        for key, specs in self.model_param_specs.items():
            idx = specs['id']
            scale = get_val(self, key, specs['scaling'])
            center = get_val(self, key, specs['centering'])
            pmin = specs['min']
            pmax = specs['max']
            if log_transform and specs['log_avail']:
                scale = torch.log(scale + eps)
                center = torch.log(center + eps)
                pmin = torch.log(pmin + eps)
                pmax = torch.log(pmax + eps)
            scale = torch.maximum(
                torch.abs(scale), torch.tensor(0.1))
            params_scale[idx] = scale
            params_center[idx] = center
            params_min[idx] = (pmin - center) / scale
            params_max[idx] = (pmax - center) / scale
        return params_scale, params_center, (params_min, params_max)

    def compile(self, log_transform: bool = False,
                rescale_params: bool = False) -> None:
        self.log_transform = log_transform
        self.log_transform_id = []
        if log_transform:
            for _, specs in self.model_param_specs.items():
                if specs['log_avail']:
                    if isinstance(specs['id'], int):
                        self.log_transform_id.append(specs['id'])
                    elif isinstance(specs['id'], list):
                        self.log_transform_id.extend(specs['id'])
        self.log_mask = torch.zeros(self.nparam, dtype=torch.bool)
        self.log_mask[self.log_transform_id] = True
        self.rescale_params = rescale_params
        scale, center, pbound = self._get_wkparam_settings(
            log_transform, rescale_params)
        self.wkparams_center = center
        self.wkparams_scale = scale
        self.wkparams_min, self.wkparams_max = pbound

    # def _get_param_scale_and_bound(
    #         self, log_transform: bool = False,
    #         rescale_params: bool = False
    # ) -> Tuple[Optional[Tensor], Tuple[Tuple[float, float], ...]]:

    #     if log_transform:
    #         if rescale_params:
    #             # actual variables:
    #             # log(D) / log(D_hat), log(rho) / log(rho_hat), (x0 - xmin) / (xmax - xmin)
    #             params_scale = torch.cat((
    #                 torch.log(torch.tensor(
    #                     [self._D, self._rho], **self.factory_args)),
    #                 self.xmax - self.xmin
    #             ), dim=0)
    #             params_bound = ((-float("Inf"), float("Inf")),
    #                             (-float("Inf"), float("Inf")), *([[0., 1.]] * self.dim))
    #         else:
    #             params_scale = None
    #             params_bound = ((-float("Inf"), float("Inf")),
    #                             (-float("Inf"), float("Inf")),
    #                             *[(self.xmin[i].item(), self.xmax[i].item()) for i in range(self.dim)])
    #     else:
    #         if rescale_params:
    #             # actual variables:
    #             # D / D_hat, rho / rho_hat, (x0 - xmin) / (xmax - xmin)
    #             params_scale = torch.cat((
    #                 torch.tensor([self._D, self._rho], **self.factory_args),
    #                 self.xmax - self.xmin
    #             ), dim=0)
    #             params_bound = ((1e-3, float("Inf")),
    #                             (1e-3, float("Inf")), *([[0., 1.]] * self.dim))
    #         else:
    #             params_scale = None
    #             params_bound = ((1e-3, float("Inf")), (1e-3, float("Inf")),
    #                             *[(self.xmin[i].item(), self.xmax[i].item()) for i in range(self.dim)])
    #     return params_scale, params_bound

    def _to_working_params(self, params: Tensor) -> Tensor:
        """ Convert params from original scale to the working scale"""
        if self.log_transform:
            # Use torch.log for the parameters that need log transform
            params = torch.where(self.log_mask, torch.log(params + 1e-8), params)
        wkparams = (params - self.wkparams_center) / self.wkparams_scale
        return wkparams

    def _to_original_params(self, wkparams: Tensor) -> Tensor:
        """ Convert working parameters to their original scale"""
        params = wkparams * self.wkparams_scale + self.wkparams_center
        if self.log_transform:
            # Use torch.exp for the parameters that need log transform
            params = torch.where(self.log_mask, torch.exp(params), params)
        return params

    # def _transform_parameters(self, params: Tensor, mode: Literal["forward", "back"] = "forward",
    #                           log_transform: bool = False, params_scale: Optional[Tensor] = None) -> Tensor:

    #     params_new = torch.zeros_like(params, **self.factory_args)
    #     if mode == "back":
    #         if params_scale is not None:
    #             if log_transform:  # exp(log(D)), exp(log(rho)), x0
    #                 params_new[:2] += torch.exp(params[:2] * params_scale[:2])
    #             else:
    #                 params_new[:2] += params[:2] * params_scale[:2]
    #             params_new[2:] += params[2:] * params_scale[2:] + self.xmin
    #         else:
    #             if log_transform:
    #                 params_new[:2] += torch.exp(params[:2])
    #             else:
    #                 params_new[:2] += params[:2]
    #             params_new[2:] += params[2:]
    #     else:
    #         if params_scale is not None:
    #             if log_transform:
    #                 params_new[:2] += torch.log(
    #                     torch.clamp(params[:2], min=1e-8)) / params_scale[:2]
    #             else:
    #                 params_new[:2] += params[:2] / params_scale[:2]
    #             params_new[2:] += (params[2:] - self.xmin) / params_scale[2:]
    #         else:
    #             if log_transform:
    #                 params_new[:2] += torch.log(
    #                     torch.clamp(params[:2], min=1e-8))
    #             else:
    #                 params_new[:2] += params[:2]
    #             params_new[2:] += params[2:]
    #     return params_new

    def solve(self, dt: FloatLike = 0.01, t1: FloatLike = 1.,
              D: TensorLikeFloat = None, rho: TensorLikeFloat = None,
              x0: TensorLikeFloat = None,
              verbose: bool = True, save_all: bool = False,
              plot_func: Optional[Callable] = None,
              plot_period: int = 10, plot_args: Optional[dict] = None
        ) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        dt = float(dt)
        t1 = float(t1)
        params = self._format_parameters(D, rho, x0)
        D = self._D(params)
        rho = self._rho(params)
        x0 = self._x0(params)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = _spec_dt(dt, t_span, self.dx, D)

        # initial state
        u = self.init_density(self.x_mesh, x0, **self.init_density_params)
        if save_all:
            u_hist = torch.zeros((nt, *self.nx), **self.factory_args)
            u_hist[0] = u.clone()
        else:
            u_hist = torch.empty(0)

        if plot_args is None:
            plot_args = {}    
        assert isinstance(plot_args, dict)

        # finite difference forward
        ti = t0
        for i in tqdm(range(nt-1), desc="Forward Simulation", disable=not verbose):
            self._forward_update(u, dt, D, rho)
            self._clip_state(u)
            u.mul_(self.domain_mask)

            ti += dt
            if save_all:
                u_hist[i+1] = u.clone()

            if plot_func is not None and i % plot_period == 0:
                curr_t = round(ti, ndigits=4)
                plot_func(u, i, curr_t, **plot_args)

        return u, torch.linspace(t0, t1, nt), u_hist

    # def solve(self, dt: FloatLike = 0.01, t1: FloatLike = 1.,
    #           D: TensorLikeFloat = None, rho: TensorLikeFloat = None, x0: TensorLikeFloat = None,
    #           verbose: bool = True, save_all: bool = False,
    #           plot_func: Optional[Callable] = None, plot_period: int = 10, plot_args: dict = {}) -> Tuple[Tensor, Tensor, Tensor]:

    #     # settings
    #     dt = float(dt)
    #     t1 = float(t1)
    #     D, rho, x0 = self._format_parameters(D, rho, x0)
    #     t0 = 0.
    #     assert t1 > t0
    #     t_span = t1 - t0

    #     # temporal grid for the discretized pde
    #     dt, nt, t1 = _spec_dt(dt, t_span, self.dx, D)

    #     # initial state
    #     u = self.init_density(self.x_mesh, x0, **self.init_density_params)
    #     if save_all:
    #         u_hist = torch.zeros((nt, *self.nx), **self.factory_args)
    #         u_hist[0] = u.clone()
    #     else:
    #         u_hist = torch.empty(0)

    #     # finite difference forward
    #     ti = t0
    #     for i in tqdm(range(nt-1), desc="Forward Simulation", disable=not verbose):
    #         self._forward_update(u, dt, D, rho)
    #         self._clip_state(u)
    #         u.mul_(self.domain_mask)

    #         ti += dt
    #         if save_all:
    #             u_hist[i+1] = u.clone()

    #         if plot_func is not None and i % plot_period == 0:
    #             curr_t = round(ti, ndigits=4)
    #             plot_func(u, i, curr_t, **plot_args)

    #     return u, torch.linspace(t0, t1, nt), u_hist

    def _forward_update(self, u: Tensor, dt: float, D: Tensor, rho: Tensor) -> None:

        _, _, c_sl = _get_slicing_positions(self.dim)
        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, self.diff_map_aux)

        du = dt * (D * nab_d_nab_u + rho * u[c_sl] * (1 - u[c_sl]))
        u[c_sl] += du
        # TODO: enforce boundary conditions

    def solve_with_grad(self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
                        D: TensorLikeFloat = None, rho: TensorLikeFloat = None,
                        x0: TensorLikeFloat = None) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        params = self._format_parameters(D, rho, x0)
        D = self._D(params)
        rho = self._rho(params)
        x0 = self._x0(params)
        t0 = 0.
        dt = float(dt)
        t1 = float(t1)
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = _spec_dt(dt, t_span, self.dx, D)

        return self._solve_with_grad(obs, dt, t1, D, rho, x0)

    def _solve_with_grad(self, obs: Tensor, dt: float, t1: float,
                         D: Tensor, rho: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        # initial state
        u = self.init_density(self.x_mesh, x0, **self.init_density_params)
        phi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d D
        psi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d rho
        eta = self.init_density_deriv(
            self.x_mesh, x0, **self.init_density_params, to_x0=True)  # eta = d u / d x0

        # finite difference forward
        t0 = 0.
        nt = int((t1 - t0) / dt) + 1
        for _ in range(nt-1):
            self._forward_with_sensitivities(u, phi, psi, eta, dt, D, rho)
            self._clip_state(u)
            u.mul_(self.domain_mask)
            phi.mul_(self.domain_mask)
            psi.mul_(self.domain_mask)
            eta.mul_(self.domain_mask)

        # gradients to parameters
        grads = self._sensitivities_to_grads(u, phi, psi, eta, obs)
        return u, torch.linspace(t0, t1, nt), grads

    def _forward_with_sensitivities(self, u: Tensor, phi: Tensor, psi: Tensor, eta: Tensor,
                                    dt: float, D: Tensor, rho: Tensor) -> None:

        _, _, c_sl = _get_slicing_positions(self.dim)

        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, self.diff_map_aux)

        du = dt * (D * nab_d_nab_u + rho * u[c_sl] * (1 - u[c_sl]))

        dphi = dt * (
            nab_d_nab_u + D * _nabla_d_nabla_f(phi, self.idx2, self.diff_map_aux) +
            rho * (1 - 2 * u[c_sl]) * phi[c_sl])

        dpsi = dt * (
            D * _nabla_d_nabla_f(psi, self.idx2, self.diff_map_aux) +
            u[c_sl] * (1 - u[c_sl]) +
            rho * (1 - 2 * u[c_sl]) * psi[c_sl]
        )

        deta = dt * (
            D * _nabla_d_nabla_f(eta, self.idx2, self.diff_map_aux) +
            rho * (1 - 2 * u[c_sl]) * eta[[slice(None, None)] + c_sl]
        )

        u[c_sl] += du
        phi[c_sl] += dphi
        psi[c_sl] += dpsi
        eta[[slice(None, None)] + c_sl] += deta
        # TODO: enforce boundary conditions

    def _sensitivities_to_grads(
            self, u: Tensor, phi: Tensor, psi: Tensor,
            eta: Tensor, obs: Tensor) -> Tensor:
        resid = u - obs
        grads = (
            torch.mean(resid * phi), torch.mean(resid * psi),
            torch.mean(resid * eta, dim=tuple(range(1, self.dim+1)))
        )
        return torch.cat([g.view(-1) for g in grads]).to(**self.factory_args)

    def _calibrate_loss_grads(
            self, wkparams: Tensor, obs: Tensor,
            dt: float = 0.01, t1: float = 1.) -> Tuple[Tensor, Tensor]:
        params = self._to_original_params(wkparams)
        u, _, grads = self.solve_with_grad(
            obs, dt, t1, self._D(params), self._rho(params), self._x0(params))
        if self.log_transform:
            grads = torch.where(self.log_mask, grads * params, grads)
        grads *= self.wkparams_scale
        return self._loss_func(u, obs), grads

    # def _calibrate_loss_grads(self, params: Tensor, obs: Tensor, dt: float = 0.01, t1: float = 1.,
    #                           log_transform: bool = False, params_scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    #     params_original = self._transform_parameters(
    #         params, 'back', log_transform, params_scale)
    #     u, _, grads = self._solve_with_grad(
    #         obs, dt, t1, params_original[0], params_original[1],
    #         params_original[2:2+self.dim])
    #     if log_transform:
    #         grads[:2] *= params_original[:2]
    #     if params_scale is not None:
    #         grads *= params_scale
    #     return self._loss_func(u, obs), grads

    def check_gradients(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            D: Optional[TensorLikeFloat] = None,
            rho: Optional[TensorLikeFloat] = None,
            x0: Optional[TensorLikeFloat] = None,
            epsilon: float = 1e-6) -> None:
        print(f"Use parameter scaling: {self.wkparams_scale}, "
              f"Log-transform: {self.log_transform}")

        params = self._format_parameters(D, rho, x0)
        wkparams = self._to_working_params(params)

        _, grads = self._calibrate_loss_grads(
            wkparams, obs, dt, t1)

        # Compute finite difference approximations of gradients
        finite_diff_grads = []
        for i in range(len(params)):
            wkparams_plus = wkparams.clone()
            wkparams_plus[i] += epsilon
            wkparams_minus = wkparams.clone()
            wkparams_minus[i] -= epsilon

            loss_plus, _ = self._calibrate_loss_grads(
                wkparams_plus, obs, dt, t1)
            loss_minus, _ = self._calibrate_loss_grads(
                wkparams_minus, obs, dt, t1)

            finite_diff_grad = (loss_plus - loss_minus) / (2 * epsilon)
            finite_diff_grads.append(finite_diff_grad.item())

        grads = grads.detach().cpu().numpy()
        finite_diff_grads = np.array(finite_diff_grads)

        # Compare gradients
        diff = np.abs(grads - finite_diff_grads)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)

        print("Gradient Check Results:")
        print(f"Max Difference: {max_diff}")
        print(f"Average Difference: {avg_diff}")
        print("Gradients:")
        print(grads)
        print("Finite Difference Approximations:")
        print(finite_diff_grads)


    # def check_gradients(self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
    #                     D: Optional[TensorLikeFloat] = None, rho: Optional[TensorLikeFloat] = None,
    #                     x0: Optional[TensorLikeFloat] = None,
    #                     epsilon: float = 1e-6, log_transform: bool = False,
    #                     rescale_params: bool = False) -> None:
    #     params_scale, _ = self._get_param_scale_and_bound(
    #         log_transform, rescale_params)
    #     print(
    #         f"Use parameter scaling: {params_scale}, Log-transform: {log_transform}")

    #     D, rho, x0 = self._format_parameters(D, rho, x0)
    #     params = torch.as_tensor([D, rho, *x0], **self.factory_args)
    #     params = self._transform_parameters(
    #         params, 'forward', log_transform, params_scale)

    #     _, grads = self._calibrate_loss_grads(
    #         params, obs, dt, t1, log_transform, params_scale)

    #     # Compute finite difference approximations of gradients
    #     finite_diff_grads = []
    #     for i in range(len(params)):
    #         params_plus = params.clone()
    #         params_plus[i] += epsilon
    #         params_minus = params.clone()
    #         params_minus[i] -= epsilon

    #         loss_plus, _ = self._calibrate_loss_grads(
    #             params_plus, obs, dt, t1, log_transform, params_scale)
    #         loss_minus, _ = self._calibrate_loss_grads(
    #             params_minus, obs, dt, t1, log_transform, params_scale)

    #         finite_diff_grad = (loss_plus - loss_minus) / (2 * epsilon)
    #         finite_diff_grads.append(finite_diff_grad.item())

    #     grads = grads.detach().cpu().numpy()
    #     finite_diff_grads = np.array(finite_diff_grads)

    #     # Compare gradients
    #     diff = np.abs(grads - finite_diff_grads)
    #     max_diff = np.max(diff)
    #     avg_diff = np.mean(diff)

    #     print("Gradient Check Results:")
    #     print(f"Max Difference: {max_diff}")
    #     print(f"Average Difference: {avg_diff}")
    #     print("Gradients:")
    #     print(grads)
    #     print("Finite Difference Approximations:")
    #     print(finite_diff_grads)

    def calibrate_model(
            self, obs: Tensor, dt: float = 0.01,
            t1: float = 1., max_iter: int = 100,
            method: Literal['L-BFGS-B', 'Nelder-Mead'] = 'L-BFGS-B',
            verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

        # check inputs
        assert isinstance(obs, Tensor)
        obs = obs.to(**self.factory_args)

        # formulate working parameters
        print(f"Use parameter scaling: {self.wkparams_scale}, "
              f"Log-transform: {self.log_transform}")
        wkparams = self._to_working_params(self.parameters)
        wkparams_np = wkparams.clone().detach().cpu().numpy()

        loss_history = {'data': []}

        if method in ['L-BFGS-B']:
            use_grad = True
            # methods that accept gradients

            def loss_func(wkparams_np):
                wkparams_tensor = torch.as_tensor(
                    wkparams_np, **self.factory_args)
                loss, grads = self._calibrate_loss_grads(
                    wkparams_tensor, obs, dt, t1)
                grads_np = grads.detach().cpu().numpy()
                print(f"loss: {loss.item()},\n"
                      f"params: {self._to_original_params(wkparams_tensor).tolist()},\n"
                      f"grads: {grads_np}")
                return loss.item(), grads_np
        else:
            # methods that do not accept gradients
            use_grad = False

            def loss_func(wkparams_np):
                wkparams_tensor = torch.as_tensor(
                    wkparams_np, **self.factory_args)
                loss, _ = self._calibrate_loss_grads(
                    wkparams_tensor, obs, dt, t1)
                print(f"loss: {loss.item()},\n"
                      f"params: {self._to_original_params(wkparams_tensor).tolist()}")
                return loss.item()

        params_bound = list(zip(
            self.wkparams_min.cpu().numpy(),
            self.wkparams_max.cpu().numpy()))

        result = minimize(loss_func, wkparams_np, method=method,
                          jac=use_grad, bounds=params_bound, tol=1e-8,
                          options={'maxiter': max_iter, 'disp': verbose})

        # Convert the optimized parameters back to tensor
        wkparams = torch.as_tensor(result.x, **self.factory_args)

        # Transform the parameters back to the original scale
        params = self._to_original_params(wkparams)

        # renew model parameters
        self._update_parameters(params)

        # format an output
        result = {
            "D": self._D(params), "rho": self._rho(params),
            "x0": self._x0(params), "loss_history": loss_history
        }

        return result

    ###########################################################################
    #########              Multiscan Calibration                ###############
    ###########################################################################

    def _sensitivities_to_grads_multiscan(
            self, u_t: Tensor, phi_t: Tensor, psi_t: Tensor,
            eta_t: Tensor, obs: Tensor) -> Tensor:
        resid = u_t - obs
        grads = (
            torch.mean(resid * phi_t), torch.mean(resid * psi_t),
            torch.mean(resid * eta_t, dim=tuple(range(1, self.dim+2)))
        )
        return torch.cat([g.view(-1) for g in grads]).to(**self.factory_args)

    def solve_with_grad_multiscan(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            D: Optional[FloatLike] = None, rho: Optional[FloatLike] = None,
            x0: Optional[NDArrayLikeFloat] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # settings
        params = self._format_parameters(D, rho, x0)
        D = self._D(params)
        rho = self._rho(params)
        x0 = self._x0(params)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, _, t1 = _spec_dt(dt, t_span, self.dx, D)

        nscan = obs.shape[0]
        assert nscan > 1

        return self._solve_with_grad_multiscan(obs, dt, t1, D, rho, x0)

    def _solve_with_grad_multiscan(
            self, obs: Tensor, dt: float, t1: float,
            D: Tensor, rho: Tensor, x0: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        t0 = 0.
        nt = int((t1 - t0) / dt) + 1

        nscan = obs.shape[0]
        mses = [float("Inf") for _ in range(nscan-1)]
        minse_t_scan = [t0 for _ in range(nscan-1)]
        minse_t_id = [0 for _ in range(nscan-1)]

        # initial state
        u = self.init_density(
            self.x_mesh, x0, **self.init_density_params)
        phi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d D
        psi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d rho
        eta = self.init_density_deriv(
            self.x_mesh, x0, **self.init_density_params, to_x0=True)  # eta = d u / d x0
        u_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        phi_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        psi_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        eta_t = torch.zeros((self.dim, nscan, *self.nx), **self.factory_args)

        # finite difference forward
        ti = t0
        for i in range(nt-1):
            self._forward_with_sensitivities(
                u, phi, psi, eta, dt, D, rho)
            self._clip_state(u)
            u.mul_(self.domain_mask)
            ti += dt
            for k in range(nscan-1):
                mse = torch.mean((u - obs[k])**2).item()
                if mse < mses[k]:
                    mses[k] = mse
                    minse_t_scan[k] = ti
                    minse_t_id[k] = i+1
                    u_t[k] = u.clone()
                    phi_t[k] = phi.clone()
                    psi_t[k] = psi.clone()
                    eta_t[:, k, ...] = eta.clone()

        u_t[-1] = u.clone()
        phi_t[-1] = phi.clone()
        psi_t[-1] = psi.clone()
        eta_t[:, -1, ...] = eta.clone()
        minse_t_scan.append(t1)

        # sort the scans, make sure t_scan in ascending order
        sorted_indices = sorted(range(
            len(minse_t_scan)), key=lambda i: minse_t_scan[i])
        minse_t_scan = [minse_t_scan[i] for i in sorted_indices]
        u_t = u_t[sorted_indices]
        phi_t = phi_t[sorted_indices]
        psi_t = psi_t[sorted_indices]
        eta_t = eta_t[:, sorted_indices, ...]

        # grad and loss
        loss = 0.
        for k in range(nscan):
            loss += self._loss_func(u_t[k], obs[k])
        grads = self._sensitivities_to_grads_multiscan(
            u_t, phi_t, psi_t, eta_t, obs)

        return u_t, minse_t_scan, loss, grads

    def _calibrate_loss_grads_multiscan(
            self, wkparams: Tensor, obs: Tensor,
            dt: float = 0.01, t1: float = 1.) -> Tuple[Tensor, Tensor, Tensor]:
        params = self._to_original_params(wkparams)
        _, t_scan, loss, grads = self.solve_with_grad_multiscan(
            obs, dt, t1, self._D(params), self._rho(params), self._x0(params))
        if self.log_transform:
            grads = torch.where(self.log_mask, grads * params, grads)
        grads *= self.wkparams_scale
        return t_scan, loss, grads

    def calibrate_model_multiscan(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            max_iter: int = 100, prepare_stage: bool = False,
            max_iter_prepare: int = 100,
            method: Literal['L-BFGS-B', 'Nelder-Mead'] = 'L-BFGS-B',
            verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

        # preprocess the data
        assert isinstance(obs, Tensor)
        obs = obs.to(**self.factory_args)
        # obs shape: (nscan, *nx)
        if obs.ndim == self.dim:
            obs = obs.unsqueeze(0)
        assert obs.ndim == self.dim + 1
        nscan = obs.shape[0]

        # Stage 1: fit the last scan
        if nscan == 1 or prepare_stage:
            print("Stage 0: fit the last scan")
            result = self.calibrate_model(
                obs[-1], dt, t1, max_iter_prepare, method,
                verbose, message_period)
            if nscan == 1:
                return result

        # Stage 2: fit all scans
        print("Main Stage: fit all scans")
        print(f"Use parameter scaling: {self.wkparams_scale}, "
              f"Log-transform: {self.log_transform}")
        wkparams = self._to_working_params(self.parameters)
        wkparams_np = wkparams.clone().detach().cpu().numpy()

        loss_history = {'data': []}  # To store the loss value

        if method in ['L-BFGS-B']:
            # methods that accept gradients
            use_grad = True

            def loss_func(wkparams_np):
                wkparams_tensor = torch.as_tensor(
                    wkparams_np, **self.factory_args)
                _, loss, grads = self._calibrate_loss_grads_multiscan(
                    wkparams_tensor, obs, dt, t1)
                grads_np = grads.detach().cpu().numpy()
                print(f"loss: {loss.item()},\n"
                      f"params: {self._to_original_params(wkparams_tensor).tolist()},\n"
                      f"grads: {grads_np}")
                return loss.item(), grads_np
        else:
            # methods that do not accept gradients
            use_grad = False

            def loss_func(wkparams_np):
                wkparams_tensor = torch.as_tensor(
                    wkparams_np, **self.factory_args)
                _, loss, _ = self._calibrate_loss_grads_multiscan(
                    wkparams_tensor, obs, dt, t1)
                print(f"loss: {loss.item()},\n"
                      f"params: {self._to_original_params(wkparams_tensor).tolist()}")
                return loss.item()

        params_bound = list(zip(
            self.wkparams_min.cpu().numpy(),
            self.wkparams_max.cpu().numpy()))

        result = minimize(loss_func, wkparams_np, method=method,
                          jac=use_grad, bounds=params_bound, tol=1e-8,
                          options={'maxiter': max_iter, 'disp': verbose})

        # Convert the optimized parameters back to tensor
        wkparams = torch.as_tensor(result.x, **self.factory_args)

        t_scan, _, _ = self._calibrate_loss_grads_multiscan(
            wkparams, obs, dt, t1)

        # Transform the parameters back to the original scale
        params = self._to_original_params(wkparams)

        # renew model parameters
        self._update_parameters(params)

        # format an output
        result = {
            "D": self._D(params), "rho": self._rho(params),
            "x0": self._x0(params), "loss_history": loss_history,
            "t_scan": t_scan
        }

        return result


    # def calibrate_model_multiscan(
    #         self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
    #         max_iter: int = 100, prepare_stage: bool = True, max_iter_prepare: int = 100,
    #         method: Literal['L-BFGS-B', 'Nelder-Mead'] = 'L-BFGS-B',
    #         log_transform: bool = False, rescale_params: bool = False,
    #         verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

    #     # preprocess the data
    #     assert isinstance(obs, Tensor)
    #     obs = obs.to(**self.factory_args)
    #     # obs shape: (nscan, *nx)
    #     if obs.ndim == self.dim:
    #         obs = obs.unsqueeze(0)
    #     assert obs.ndim == self.dim + 1
    #     nscan = obs.shape[0]

    #     # Stage 1: fit the last scan
    #     if nscan == 1 or prepare_stage:
    #         print("Stage 0: fit the last scan")
    #         result = self.calibrate_model(
    #             obs[-1], dt, t1, max_iter_prepare, method,
    #             log_transform, rescale_params, verbose, message_period)
    #         if nscan == 1:
    #             return result

    #     # Stage 2: fit all scans
    #     print("Main Stage: fit all scans")
    #     params_scale, params_bound = self._get_param_scale_and_bound(
    #         log_transform, rescale_params)
    #     print(
    #         f"Parameter scaling: {params_scale}, Log-transform: {log_transform}")

    #     params = self.parameters(log_transform, params_scale)
    #     params_np = params.clone().detach().cpu().numpy()
    #     loss_history = {'data': []}  # To store the loss value

    #     if method in ['L-BFGS-B']:
    #         # methods that accept gradients
    #         use_grad = True

    #         def loss_func(params_np):
    #             params_tensor = torch.as_tensor(params_np, **self.factory_args)
    #             _, loss, grads = self._calibrate_loss_grads_multiscan(
    #                 params_tensor, obs, dt, t1, log_transform, params_scale)
    #             print(
    #                 f"params: {params_tensor}, loss: {loss.item()},\ngrads: {grads}, dt: {dt}, t1: {t1}")
    #             return loss.item(), grads.cpu().numpy()
    #     else:
    #         # methods that do not accept gradients
    #         use_grad = False

    #         def loss_func(params_np):
    #             params_tensor = torch.as_tensor(params_np, **self.factory_args)
    #             _, loss, _ = self._calibrate_loss_grads_multiscan(
    #                 params_tensor, obs, dt, t1, log_transform, params_scale)
    #             print(
    #                 f"params: {params_tensor}, loss: {loss.item()}, dt: {dt}, t1: {t1}")
    #             return loss.item()

    #     result = minimize(loss_func, params_np, method=method,
    #                       jac=use_grad, bounds=params_bound,
    #                       options={'maxiter': max_iter, 'disp': verbose})

    #     # Convert the optimized parameters back to tensor
    #     params = torch.as_tensor(result.x, **self.factory_args)
    #     t_scan, _, _ = self._calibrate_loss_grads_multiscan(
    #         params, obs, dt, t1, log_transform, params_scale)

    #     # Transform the parameters back to the original scale
    #     params_new = self._transform_parameters(
    #         params, "back", log_transform, params_scale)

    #     # renew model parameters
    #     self._update_parameters(params_new[0], params_new[1], params_new[2:])

    #     # format an output
    #     result = {
    #         "D": params_new[0], "rho": params_new[1],
    #         "x0": params_new[2:], "loss_history": loss_history,
    #         "t_scan": t_scan
    #     }

    #     return result
