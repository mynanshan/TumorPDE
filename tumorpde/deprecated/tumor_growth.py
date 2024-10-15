from typing import List, Union, Tuple, Literal, Callable, Optional, Dict, Any
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import torch
from torch import nn
from torch import optim
from torch import Tensor

from tumorpde.helpers import _get_slicing_positions
from tumorpde.volume_domain import VolumeDomain
from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike


def default_init_density(x: Tensor, x0: Tensor, w: float, h: float, rmax: float = 1.,
                         log_transform: bool = False, eps: float = 1e-6) -> Tensor:
    # x shape: (ndim, *voxel_shape)
    # x0 shape: (ndim)
    # w, h, rmax: float
    # log_transform: bool
    # eps: float
    dim = len(x0)
    assert x.shape[0] == dim
    diff = x.clone()
    for k in range(dim):
        diff[k] -= x0[k]
    r2 = torch.sum(diff ** 2, dim=0)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= rmax**2)
    if log_transform:
        density_value = (1 - eps) * density_value + 0.5 * eps
        density_value = torch.logit(density_value)
    return density_value


def default_init_density_deriv(
        x: Tensor, x0: Tensor, w: float, h: float, rmax: float = 3.,
        to_x0: bool = False) -> List[Tensor]:
    # x shape: (ndim, *voxel_shape)
    dim = len(x0)
    assert x.shape[0] == dim
    diff = x.clone()
    for k in range(dim):
        diff[k] -= x0[k]
    r2 = torch.sum(diff ** 2, dim=0)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= rmax**2)
    deriv = (2 / w**2) * diff * density_value
    deriv = deriv.tolist() if not to_x0 else (-deriv).tolist()
    return deriv


class TumorGrowthGridModel():

    def __init__(self, domain: VolumeDomain, x0: NDArrayLikeFloat = None,
                 D: FloatLike = 1., rho: FloatLike = 1.,
                 init_density_func: Optional[Callable] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 autograd: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """ 
        Args:
            domain: VolumeDomain, the domain of the tumor growth model.
            x0: NDArrayLikeFloat, the initial position of the tumor.
            D: FloatLike, the diffusion coefficient.
            rho: FloatLike, the growth rate.
            init_density_func: callable, the initial density function.
            init_density_deriv: callable, the derivative of the initial density function.
            init_density_params: dict, the parameters of the initial density function.
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
        self.nx = [len(xi) for xi in self.x]
        self.x_mesh = torch.stack(torch.meshgrid(
            *self.x, indexing="ij"))  # (ndim, *nx)
        self.dx = tuple(self.domain.voxel_widths)
        self.idx2 = tuple(1 / (self.domain.voxel_widths)**2)
        self.diff_map = torch.as_tensor(domain.voxel, **factory_args)
        self.domain_mask = self.diff_map > 0.

        # Torch factory settings
        self.device = device
        self.dtype = dtype
        self.factory_args = factory_args

        # Parameters
        self.x0 = list(self.domain.bbox_widths / 2) if x0 is None else list(x0)
        self.D = D
        self.rho = rho
        # save the previous parameters for convenience
        self._x0_old = None
        self._D_old = None
        self._rho_old = None
        # Using autograd for param estimation?
        self.auto_grad = autograd

        # Init_density
        if init_density_func is None:
            self.init_density = default_init_density
            self.init_density_deriv = default_init_density_deriv
        else:
            self.init_density = init_density_func
            if init_density_deriv is None:
                raise ValueError(
                    "Derivative of the initial density function is required.")
            self.init_density_deriv = init_density_deriv
        if init_density_params is None:
            self.init_density_params = {
                "w": (self.domain.bbox_widths / 20).mean(),
                "h": 0.01, "rmax": 3. * self.dim
            }
        else:
            self.init_density_params = init_density_params

        # Computational helpers
        self._slice_positions = _get_slicing_positions(self.dim)
        self.diff_map_aux = self._diff_map_auxiliaries()

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value: FloatLike):
        self._D = float(value)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value: FloatLike):
        self._rho = float(value)

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value: NDArrayLikeFloat):
        assert len(value) == self.dim
        self._x0 = list(value)

    def parameters(self, log_transform: bool = False, params_scale: Optional[Tensor] = None) -> nn.Parameter:
        """ Collect the parameters in a single tensor, return an nn.Parameter.
         Only used when calibrating the model. """
        params = torch.tensor([self.D, self.rho] +
                              self.x0, **self.factory_args)
        params = self._transform_parameters(
            params, 'forward', log_transform, params_scale)
        return nn.Parameter(params, requires_grad=self.auto_grad)

    def _get_parameters(self, D: Optional[FloatLike] = None, rho: Optional[FloatLike] = None,
                        x0: Optional[NDArrayLikeFloat] = None) -> Tuple[float, float, List[float]]:
        """ Helper function to format a new set parameters. 
        Ease the trial of new parameters. If not available,
        return the current parameters. """
        if D is None:
            D = self.D
        else:
            D = float(D)
        if rho is None:
            rho = self.rho
        else:
            rho = float(rho)
        if x0 is None:
            x0 = self.x0
        else:
            x0 = list(x0)
        return (D, rho, x0)

    def _update_parameters(self, D: Optional[float] = None, rho: Optional[float] = None,
                           x0: Optional[List[float]] = None) -> None:
        """ Update the parameters. """
        if D is not None:
            self._D_old = self.D
            self.D = float(D)
        if rho is not None:
            self._rho_old = self.rho
            self.rho = float(rho)
        if x0 is not None:
            self._x0_old = self.x0
            self.x0 = list(x0)

    def _get_param_scale_and_bound(self, log_transform: bool = False, rescale_params: bool = False) -> Tuple[Optional[Tensor], Tuple[Tuple[float, float], ...]]:

        if log_transform:
            if rescale_params:
                # actual variables:
                # log(D) / log(D_hat), log(rho) / log(rho_hat), (x0 - xmin) / (xmax - xmin)
                params_scale = torch.cat((
                    torch.log(torch.tensor([self._D, self._rho], **self.factory_args)),
                    self.xmax - self.xmin
                ), dim=0)
                params_bound = ((-float("Inf"), float("Inf")),
                                (-float("Inf"), float("Inf")), *([[0., 1.]] * self.dim))
            else:
                params_scale = None
                params_bound = ((-float("Inf"), float("Inf")),
                                (-float("Inf"), float("Inf")),
                                *[(self.xmin[i].item(), self.xmax[i].item()) for i in range(self.dim)])
        else:
            if rescale_params:
                # actual variables:
                # D / D_hat, rho / rho_hat, (x0 - xmin) / (xmax - xmin)
                params_scale = torch.cat((
                    torch.tensor([self._D, self._rho], **self.factory_args),
                    self.xmax - self.xmin
                ), dim=0)
                params_bound = ((0., float("Inf")),
                                (0., float("Inf")), *([[0., 1.]] * self.dim))
            else:
                params_scale = None
                params_bound = ((0., float("Inf")), (0., float("Inf")),
                                *[(self.xmin[i].item(), self.xmax[i].item()) for i in range(self.dim)])
        return params_scale, params_bound

    def _transform_parameters(self, params: Tensor, mode: Literal["forward", "back"] = "forward",
                              log_transform: bool = False, params_scale: Optional[Tensor] = None) -> Tensor:

        params_new = torch.zeros_like(params, **self.factory_args)
        if mode == "back":
            if params_scale is not None:
                if log_transform:  # exp(log(D)), exp(log(rho)), x0
                    params_new[:2] += torch.exp(params[:2] * params_scale[:2])
                else:
                    params_new[:2] += params[:2] * params_scale[:2]
                params_new[2:] += params[2:] * params_scale[2:] + self.xmin
            else:
                if log_transform:
                    params_new[:2] += torch.exp(params[:2])
                else:
                    params_new[:2] += params[:2]
                params_new[2:] += params[2:]
        else:
            if params_scale is not None:
                if log_transform:
                    params_new[:2] += torch.log(
                        torch.clamp(params[:2], min=1e-8)) / params_scale[:2]
                else:
                    params_new[:2] += params[:2] / params_scale[:2]
                params_new[2:] += (params[2:] - self.xmin) / params_scale[2:]
            else:
                if log_transform:
                    params_new[:2] += torch.log(
                        torch.clamp(params[:2], min=1e-8))
                else:
                    params_new[:2] += params[:2]
                params_new[2:] += params[2:]
        return params_new

    def _clip_state(self, u: Tensor) -> None:
        u.clamp_(0.0, 1.0)

    def _loss_func(self, u: Tensor, obs: Tensor) -> Tensor:
        # TODO: allow logit type loss
        return 0.5 * torch.mean((u - obs)**2)

    def _spec_dt(self, dt: float, t_span: float, dx: Tuple[float, ...], D: float) -> Tuple[float, int, float]:

        # grid for the discretized pde
        nt = int(t_span / dt) + 1
        if nt <= 0:
            raise ValueError("Improper dt")

        # check if dt is small enough
        ref_dt = min(dx)**2 / (4 * D)
        if dt > ref_dt:
            dt = ref_dt
            nt = int(t_span / dt) + 1
            print(
                f"WARNING: dt is not small enough, change nt, dt to {nt} and {dt}.")

        t1 = (nt - 1) * dt  # the actual t1

        return dt, nt, t1

    def _diff_map_auxiliaries(self) -> Tuple[List[Tensor], List[Tensor]]:
        """ auxiliaries for numerical approximation on a grid """

        d = self.diff_map
        dim = self.dim

        dp_d = []  # d^{+} + d
        dm_d = []  # d^{-} + d

        # slicing helpers
        p1_sl, m1_sl, c_sl = self._slice_positions

        for i in range(dim):

            dp_d.append(d[p1_sl[i]] + d[c_sl])
            dm_d.append(d[m1_sl[i]] + d[c_sl])

        return dp_d, dm_d

    def _nabla_d_nabla_f(self, f: Tensor) -> Tensor:
        r""" \nabla \cdot (d \nabla f) for numerical approximation on a grid """

        idx2 = self.idx2
        d_aux = self.diff_map_aux

        dp_d = d_aux[0]
        dm_d = d_aux[1]

        p1_sl, m1_sl, c_sl = self._slice_positions

        if self.dim == f.ndim:

            if f.shape != tuple(self.nx):
                raise ValueError("Incorrect shape of f")

            nab_d_nab_f = torch.zeros(
                *[n-2 for n in self.nx], device=self.device)

            for i in range(self.dim):
                nab_d_nab_f += 0.5 * idx2[i] * (
                    dp_d[i] * (f[p1_sl[i]] - f[c_sl]) +
                    dm_d[i] * (f[m1_sl[i]] - f[c_sl])
                )

        elif self.dim == f.ndim - 1:

            if f.shape[1:] != tuple(self.nx):
                raise ValueError("Incorrect shape of f")

            # a time dimension is included in f, need broadcasting
            t_m1 = [slice(None, -1)]
            nab_d_nab_f = torch.zeros(
                f.shape[0]-1, *[n-2 for n in self.nx], device=self.device)

            for i in range(self.dim):
                nab_d_nab_f += 0.5 * idx2[i] * (
                    dp_d[i] * (f[t_m1 + p1_sl[i]] - f[t_m1 + c_sl]) +
                    dm_d[i] * (f[t_m1 + m1_sl[i]] - f[t_m1 + c_sl])
                )
        else:
            raise ValueError("Dimension mismatch between 'f' and 'self.dim'")

        return nab_d_nab_f


class TumorGrowthFD(TumorGrowthGridModel):

    def __init__(self, domain: VolumeDomain, x0: NDArrayLikeFloat = None, D: FloatLike = 1., rho: FloatLike = 1.,
                 init_density_func: Optional[Callable] = None, init_density_deriv: Optional[Callable] = None,
                 init_density_params: Optional[Dict] = None,
                 dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):

        super().__init__(domain, x0, D, rho,
                         init_density_func, init_density_deriv, init_density_params,
                         False, dtype, device)

    def solve(self, dt: float = 0.01, t1: float = 1.,
              D: FloatLike = None, rho: FloatLike = None, x0: NDArrayLikeFloat = None,
              verbose: bool = True, save_all: bool = False,
              plot_func: Optional[Callable] = None, plot_period: int = 10, plot_args: dict = {}) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        D, rho, x0 = self._get_parameters(D, rho, x0)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = self._spec_dt(dt, t_span, self.dx, D)

        # initial state
        # updated when computing resids
        u = self.init_density(
            self.x_mesh, x0, **self.init_density_params)
        if save_all:
            u_hist = torch.zeros((nt, *self.nx), device=self.device)
            u_hist[0] = u.clone()
        else:
            u_hist = torch.empty(0)

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

    def _forward_update(self, u: Tensor, dt: float, D: float, rho: float) -> None:

        _, _, c_sl = self._slice_positions
        nab_d_nab_u = self._nabla_d_nabla_f(u)

        du = dt * (D * nab_d_nab_u + rho * u[c_sl] * (1 - u[c_sl]))
        u[c_sl] += du
        # TODO: enforce boundary conditions

    def solve_with_grad(self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
                        D: FloatLike = None, rho: FloatLike = None, x0: NDArrayLikeFloat = None) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        D, rho, x0 = self._get_parameters(D, rho, x0)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = self._spec_dt(dt, t_span, self.dx, D)

        # initial state
        u = self.init_density(self.x_mesh, x0, **self.init_density_params)
        phi = torch.zeros(self.nx, device=self.device)  # phi = d u / d D
        psi = torch.zeros(self.nx, device=self.device)  # phi = d u / d rho
        eta = self.init_density_deriv(
            self.x_mesh, x0, **self.init_density_params, to_x0=True)  # eta = d u / d x0
        # finite difference forward
        for _ in range(nt-1):
            self._forward_with_sensitivities(u, phi, psi, eta, dt, D, rho)
            self._clip_state(u)
            u.mul_(self.domain_mask)

        # gradients to parameters
        grads = self._sensitivities_to_grads(u, phi, psi, eta, obs)

        return u, torch.linspace(t0, t1, nt), grads

    def _forward_with_sensitivities(self, u: Tensor, phi: Tensor, psi: Tensor, eta: List[Tensor],
                                    dt: float, D: float, rho: float) -> None:

        _, _, c_sl = self._slice_positions

        nab_d_nab_u = self._nabla_d_nabla_f(u)

        du = dt * (D * nab_d_nab_u + rho * u[c_sl] * (1 - u[c_sl]))

        dphi = dt * (
            nab_d_nab_u + D * self._nabla_d_nabla_f(phi) +
            rho * (1 - 2 * u[c_sl]) * phi[c_sl])

        dpsi = dt * (
            D * self._nabla_d_nabla_f(psi) +
            u[c_sl] * (1 - u[c_sl]) +
            rho * (1 - 2 * u[c_sl]) * psi[c_sl]
        )

        deta = [
            dt * (
                D * self._nabla_d_nabla_f(eta[i]) +
                rho * (1 - 2 * u[c_sl]) * eta[i][c_sl]
            ) for i in range(self.dim)
        ]

        u[c_sl] += du
        phi[c_sl] += dphi
        psi[c_sl] += dpsi
        for i in range(self.dim):
            eta[i][c_sl] += deta[i]
            # TODO: enforce boundary conditions

    def _sensitivities_to_grads(self, u: Tensor, phi: Tensor, psi: Tensor, eta: List[Tensor], obs: Tensor) -> Tensor:
        resid = u - obs
        grads = (
            torch.mean(resid * phi), torch.mean(resid * psi),
            *[torch.mean(resid * eta[i]) for i in range(self.dim)]
        )
        return torch.cat([g.view(-1) for g in grads])

    def _calibrate_loss_grads(self, params: Tensor, obs: Tensor, dt: float = 0.01, t1: float = 1.,
                              log_transform: bool = False, params_scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        params_original = self._transform_parameters(
            params, 'back', log_transform, params_scale)
        u, _, grads = self.solve_with_grad(
            obs, dt, t1, params_original[0], params_original[1], params_original[2:])
        if log_transform:
            grads[:2] *= params_original[:2]
        if params_scale is not None:
            grads *= params_scale
        return self._loss_func(u, obs), grads

    def check_gradients(self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
                        D: Optional[FloatLike] = None, rho: Optional[FloatLike] = None, x0: Optional[NDArrayLikeFloat] = None,
                        epsilon: float = 1e-6, log_transform: bool = False, rescale_params: bool = False) -> None:
        params_scale, _ = self._get_param_scale_and_bound(
            log_transform, rescale_params)
        print(
            f"Use parameter scaling: {params_scale}, Log-transform: {log_transform}")

        D, rho, x0 = self._get_parameters(D, rho, x0)
        params = torch.as_tensor([D, rho, *x0], **self.factory_args)
        params = self._transform_parameters(
            params, 'forward', log_transform, params_scale)

        _, grads = self._calibrate_loss_grads(
            params, obs, dt, t1, log_transform, params_scale)

        # Compute finite difference approximations of gradients
        finite_diff_grads = []
        for i in range(len(params)):
            params_plus = params.clone()
            params_plus[i] += epsilon
            params_minus = params.clone()
            params_minus[i] -= epsilon

            loss_plus, _ = self._calibrate_loss_grads(
                params_plus, obs, dt, t1, log_transform, params_scale)
            loss_minus, _ = self._calibrate_loss_grads(
                params_minus, obs, dt, t1, log_transform, params_scale)

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

    def calibrate_model(self, obs: Tensor, dt: float = 0.01, t1: float = 1., max_iter: int = 100,
                        backend: Literal['scipy', 'torch'] = 'scipy',
                        log_transform: bool = False, rescale_params: bool = False,
                        verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

        params_scale, params_bound = self._get_param_scale_and_bound(
            log_transform, rescale_params)
        print(
            f"Parameter scaling: {params_scale}, Log-transform: {log_transform}")

        params = self.parameters(log_transform, params_scale)
        loss_history = {'data': []}

        if backend == 'torch':

            if params_bound is not None:
                params_bound_min = torch.tensor(
                    [p[0] for p in params_bound], **self.factory_args)
                params_bound_max = torch.tensor(
                    [p[1] for p in params_bound], **self.factory_args)

            loss_tmp = [float("Inf")]
            optimizer = optim.LBFGS([params], max_iter=max_iter)

            def closure():
                optimizer.zero_grad()
                loss, grads = self._calibrate_loss_grads(
                    params, obs, dt, t1, log_transform, params_scale)
                # Manually set the gradients
                params.grad = grads.clone()
                loss_tmp[0] = loss.item()
                return loss.item()  # Return a scalar

            # Perform multiple steps of optimization
            for it in tqdm(range(max_iter), desc="Optimization", disable=not verbose):
                optimizer.step(closure)
                # Enforce bounds after each optimization step
                with torch.no_grad():
                    if params_bound is not None:
                        params.clamp_(min=params_bound_min, max=params_bound_max)

                if it % message_period == 0:
                    loss_history['data'].append(loss_tmp[0])
                    if verbose:
                        with torch.no_grad():
                            params_original = self._transform_parameters(
                                params, 'back', log_transform, params_scale)
                        print(f"Loss: {loss_history['data'][-1]:.4f}")
                        print(
                            f"Parameters: {params_original.detach().tolist()}")

        elif backend == 'scipy':
            # Convert the initial parameters to numpy array
            params_np = params.clone().detach().cpu().numpy()

            # TODO: add loss history

            # Define the loss function
            def loss_func(params_np):
                # Convert the numpy array back to tensor
                params_tensor = torch.as_tensor(params_np, device=self.device)
                # Calculate the loss and gradients
                loss, grads = self._calibrate_loss_grads(
                    params_tensor, obs, dt, t1, log_transform, params_scale)
                # Convert the gradients to numpy array
                grads_np = grads.detach().cpu().numpy()
                return loss.item(), grads_np

            # Minimize the loss function using LBFGS-B
            result = minimize(loss_func, params_np, method='L-BFGS-B',
                              jac=True, bounds=params_bound,
                              options={'maxiter': max_iter, 'disp': verbose})

            # Convert the optimized parameters back to tensor
            params = torch.as_tensor(result.x, device=self.device)

        # Transform the parameters back to the original scale
        params_new = self._transform_parameters(
            params, "back", log_transform, params_scale)

        # renew model parameters
        self._update_parameters(params_new[0], params_new[1], params_new[2:])

        # format an output
        result = {
            "D": params_new[0], "rho": params_new[1],
            "x0": params_new[2:], "loss_history": loss_history
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
            *[torch.mean(resid * eta_t[:, i, ...]) for i in range(self.dim)]
        )
        return torch.cat([g.view(-1) for g in grads])

    def solve_with_grad_multiscan(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            D: Optional[FloatLike] = None, rho: Optional[FloatLike] = None, x0: Optional[NDArrayLikeFloat] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # settings
        D, rho, x0 = self._get_parameters(D, rho, x0)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = self._spec_dt(dt, t_span, self.dx, D)

        nscan = obs.shape[0]
        assert nscan > 1
        mses = [float("Inf") for _ in range(nscan-1)]
        minse_t_scan = [t0 for _ in range(nscan-1)]
        minse_t_id = [0 for _ in range(nscan-1)]

        # initial state
        u = self.init_density(
            self.x_mesh, x0, **self.init_density_params)
        phi = torch.zeros(self.nx, device=self.device)  # phi = d u / d D
        psi = torch.zeros(self.nx, device=self.device)  # phi = d u / d rho
        eta = self.init_density_deriv(
            self.x_mesh, x0, **self.init_density_params, to_x0=True)  # eta = d u / d x0
        u_t = torch.zeros((nscan, *self.nx), device=self.device)
        phi_t = torch.zeros((nscan, *self.nx), device=self.device)
        psi_t = torch.zeros((nscan, *self.nx), device=self.device)
        eta_t = torch.zeros((nscan, self.dim, *self.nx), device=self.device)

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
                    for j in range(self.dim):
                        eta_t[k, j] = eta[j].clone()

        u_t[-1] = u.clone()
        phi_t[-1] = phi.clone()
        psi_t[-1] = psi.clone()
        for j in range(self.dim):
            eta_t[-1, j] = eta[j].clone()
        minse_t_scan.append(t1)

        # sort the scans, make sure t_scan in ascending order
        sorted_indices = sorted(range(
            len(minse_t_scan)), key=lambda i: minse_t_scan[i])
        minse_t_scan = [minse_t_scan[i] for i in sorted_indices]
        u_t = u_t[sorted_indices]
        phi_t = phi_t[sorted_indices]
        psi_t = psi_t[sorted_indices]
        eta_t = eta_t[sorted_indices]
        # grad and loss
        loss = 0.
        for k in range(nscan):
            loss += self._loss_func(u_t[k], obs[k]).item()
        grads = self._sensitivities_to_grads_multiscan(
            u_t, phi_t, psi_t, eta_t, obs)

        return u_t, minse_t_scan, loss, grads

    def _calibrate_loss_grads_multiscan(
            self, params: Tensor, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            log_transform: bool = False, params_scale: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        params_original = self._transform_parameters(
            params, 'back', log_transform, params_scale)
        _, t_scan, loss, grads = self.solve_with_grad_multiscan(
            obs, dt, t1, params_original[0], params_original[1], params_original[2:])
        if log_transform:
            grads[:2] *= params_original[:2]
        if params_scale is not None:
            grads[:2+self.dim] *= params_scale
        return t_scan, loss, grads

    def calibrate_model_multiscan(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            max_iter: int = 100, prepare_stage: bool = True, max_iter_prepare: int = 100,
            log_transform: bool = False, rescale_params: bool = False,
            verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

        # preprocess the data
        # obs shape: (nscan, *nx)
        if obs.ndim == self.dim:
            obs = obs.unsqueeze(0)
        assert obs.ndim == self.dim + 1
        nscan = obs.shape[0]

        # Stage 1: fit the last scan
        if nscan == 1 or prepare_stage:
            result = self.calibrate_model(
                obs[-1], dt, t1, max_iter_prepare, 'scipy',
                log_transform, rescale_params, verbose, message_period)
            if nscan == 1:
                return result

        # Stage 2: fit all scans
        params_scale, params_bound = self._get_param_scale_and_bound(
            log_transform, rescale_params)
        print(f"Parameter scaling: {params_scale}, Log-transform: {log_transform}")

        params = self.parameters(log_transform, params_scale)
        params_np = params.clone().detach().cpu().numpy()
        loss_history = {'data': []}  # To store the loss value

        def loss_func(params_np):
            params_tensor = torch.as_tensor(params_np, device=self.device)
            _, loss, grads = self._calibrate_loss_grads_multiscan(
                params_tensor, obs, dt, t1, log_transform, params_scale)
            return loss, grads.cpu().numpy()

        result = minimize(loss_func, params_np, method='L-BFGS-B',
                          jac=True, bounds=params_bound,
                          options={'maxiter': max_iter, 'disp': verbose})

        # Convert the optimized parameters back to tensor
        params = torch.as_tensor(result.x, device=self.device)
        t_scan, _, _ = self._calibrate_loss_grads_multiscan(
            params, obs, dt, t1, log_transform, params_scale)

        # Transform the parameters back to the original scale
        params_new = self._transform_parameters(
            params, "back", log_transform, params_scale)

        # renew model parameters
        self._update_parameters(params_new[0], params_new[1], params_new[2:])

        # format an output
        result = {
            "D": params_new[0], "rho": params_new[1],
            "x0": params_new[2:], "loss_history": loss_history,
            "t_scan": t_scan
        }

        return result


# class TumorGrowthODiL(TumorGrowthGridModel):

#     def __init__(self, domain: GeometryXTime, dt=0.001, x0=None, D=1., rho=1.,
#                  init_density_func=None, init_density_params=None, device='cpu'):

#         super().__init__(domain, x0, D, rho, True,
#                          init_density_func, init_density_params, device)

#         # specifying the time grid
#         dt, nt, t1, _ = self._spec_dt(
#             dt, self.t0, self.t_span, self.dx, self._D)
#         self.nt = nt
#         self.dt = dt
#         self.t1 = t1
#         # NOTE: nt will not change if D has been updated,
#         # which can be a problem if D is too large

#         # discrete pde values
#         self._u = nn.Parameter(torch.rand(
#             self.nt-1, *self.nx, device=self.device))
#         self.mesh_weights = torch.zeros((self.nt-1, *self.nx), device=device)

#     @property
#     def u(self):
#         with torch.no_grad():
#             u0 = self.init_density(
#                 self.x_mesh, self.x0, self.init_density_params)
#             u = torch.cat((u0.unsqueeze(0), self._u), dim=0)
#         return u

#     def _pde_loss(self, D, rho, x0):

#         _, _, c_sl = self._slice_positions
#         t_p1 = [slice(1, None)]
#         t_m1 = [slice(None, -1)]

#         u0 = self.init_density(
#             self.x_mesh, x0, self.init_density_params)
#         u = torch.cat((u0.unsqueeze(0), self._u), dim=0)  # cat along time

#         nab_d_nab_u = self._nabla_d_nabla_f(u)  # (nt-1, *(nx-2))

#         diffussion = D * nab_d_nab_u
#         proliferation = rho * u[t_m1 + c_sl] * (1 - u[t_m1 + c_sl])

#         du = u[t_p1 + c_sl] - u[t_m1 + c_sl]
#         pde_residual = du - self.dt * (diffussion + proliferation)

#         return torch.square(pde_residual)

#     def _boundary_loss(self):
#         # not quite useful, leave it blank for now
#         pass

#     def _data_loss(self, obs, i=-1):
#         """
#         obs: observed data
#         i: the time indices that match the obs, can be any array-like
#             objects, an int, or a slice object 
#         """
#         return torch.square(self._u[i] - obs)

#     def calibrate_model(self, obs, i=-1, method="adam", opt_args={}, max_epoch=1000,
#                         log_transform=False, rescale_params=False, loss_weights=[1., 1.],
#                         verbose=False, message_period=10):

#         params_scale, params_bound = self._get_param_scale_and_bound(
#             log_transform, rescale_params)
#         print(
#             f"Parameter scaling: {params_scale}, Log-transform: {log_transform}")

#         params = self.parameters(log_transform, params_scale)
#         loss_history = {'pde': [], 'data': [], 'total': []}

#         if method == "adam":

#             optimizer = optim.Adam([params, self._u], **opt_args)

#             for epoch in tqdm(range(max_epoch), desc="Optimization", disable=not verbose):
#                 optimizer.zero_grad()

#                 params_original = self._transform_parameters(
#                     params, 'back', log_transform, params_scale)

#                 # Calculate PDE residuals
#                 pde_losses = self._pde_loss(
#                     params_original[0], params_original[1], params_original[2:])
#                 pde_loss = torch.mean(pde_losses**2)

#                 # Calculate data residuals
#                 data_losses = self._data_loss(obs, i)
#                 data_loss = torch.mean(data_losses**2)

#                 # Total loss
#                 total_loss = loss_weights[0] * \
#                     data_loss + loss_weights[1] * pde_loss

#                 if epoch % message_period == 0:
#                     loss_history['pde'].append(pde_loss.item())
#                     loss_history['data'].append(data_loss.item())
#                     loss_history['total'].append(total_loss.item())

#                 # Backpropagation and optimization
#                 total_loss.backward()
#                 optimizer.step()

#                 if verbose and epoch % message_period == 0:
#                     print(f"Epoch {epoch}: Total Loss={total_loss.item():.4f}",
#                           f"Data Loss={data_loss.item():.4f}, PDE Loss={pde_loss.item():.4f}")
#                     print(f"> Parameters: D={params_original.detach()[0]:.3f}",
#                           f"rho={params_original.detach()[1]:.3f}",
#                           f"x0={_round_tensor_tolist(params_original.detach()[2:], 3)}")

#         elif method == "lbfgs":

#             params_bound_min = torch.as_tensor(
#                 [p[0] for p in params_bound], device=self.device)
#             params_bound_max = torch.as_tensor(
#                 [p[1] for p in params_bound], device=self.device)
#             loss_pde_tmp = [float("Inf")]
#             loss_data_tmp = [float("Inf")]
#             loss_total_tmp = [float("Inf")]

#             optimizer = optim.LBFGS([params, self._u], **opt_args)

#             def closure():
#                 optimizer.zero_grad()

#                 params_original = self._transform_parameters(
#                     params, 'back', log_transform, params_scale)

#                 # Calculate PDE residuals
#                 pde_losses = self._pde_loss(
#                     params_original[0], params_original[1], params_original[2:])
#                 pde_loss = torch.mean(pde_losses**2)

#                 # Calculate data residuals
#                 data_losses = self._data_loss(obs, i)
#                 data_loss = torch.mean(data_losses**2)

#                 # Total loss
#                 total_loss = loss_weights[0] * \
#                     data_loss + loss_weights[1] * pde_loss

#                 loss_pde_tmp[0] = pde_loss.item()
#                 loss_data_tmp[0] = data_loss.item()
#                 loss_total_tmp[0] = total_loss.item()

#                 # Backpropagation and optimization
#                 total_loss.backward()
#                 return total_loss

#             for epoch in tqdm(range(max_epoch), desc="Optimization",
#                               disable=not verbose):
#                 optimizer.step(closure)
#                 with torch.no_grad():
#                     if params_bound is not None:
#                         params.data.clamp_(
#                             min=params_bound_min, max=params_bound_max)
#                         # params.data = torch.clamp(
#                         #     params, min=params_bound_min, max=params_bound_max)

#                 if epoch % message_period == 0:
#                     loss_history['pde'].append(loss_pde_tmp[0])
#                     loss_history['data'].append(loss_data_tmp[0])
#                     loss_history['total'].append(loss_total_tmp[0])
#                     if verbose:
#                         with torch.no_grad():
#                             params_original = self._transform_parameters(
#                                 params, 'back', log_transform, params_scale)

#                         print(f"Epoch {epoch}: Total Loss={loss_history['total'][-1]:.4f}",
#                               f"Data Loss={loss_history['data'][-1]:.4f}, PDE Loss={loss_history['pde'][-1]:.4f}")
#                         print(f"> Parameters: D={params_original.detach()[0].item():.3f}",
#                               f"rho={params_original.detach()[1].item():.3f}",
#                               f"x0={_round_tensor_tolist(params_original.detach()[2:], 3)}")

#         # Transform the parameters back to the original scale
#         with torch.no_grad():
#             params_new = self._transform_parameters(
#                 params, "back", log_transform, params_scale)

#         # renew model parameters
#         self._update_parameters(params_new[0], params_new[1], params_new[2:])

#         # format an output
#         result = {
#             "D": params_new[0], "rho": params_new[1],
#             "x0": params_new[2:], "loss_history": loss_history
#         }

#         return result
