from typing import Optional, Callable, Dict, Any, Literal, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from scipy.optimize import minimize
from tqdm import tqdm
import nibabel as nib

from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike
from tumorpde.volume_domain import VolumeDomain
from tumorpde.models._base import TumorFixedFieldBase
from tumorpde.models.comp_utils import _get_slicing_positions, _nabla_d_nabla_f, _spec_dt

# TODO:
# - allow a user-specified initial density func with learnable params defined inside
# - remove built-in support for learning x0
# - user needs to provide a list of external learnable parameters

class TumorInfiltraFD(TumorFixedFieldBase):

    def __init__(self, domain: VolumeDomain,
                 D: TensorLikeFloat = 1.,
                 alpha: TensorLikeFloat = 1.,
                 init_density_func: Optional[Callable] = None,
                 init_learnable_params: Optional[TensorLikeFloat] = None,
                 init_other_params: Optional[Dict] = None,
                 init_density_deriv: Optional[Callable] = None,
                 init_param_min: Optional[TensorLikeFloat] = None,
                 init_param_max: Optional[TensorLikeFloat] = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        """
        Args:

        domain:
            VolumeDomain, the domain of the tumor growth model.
        D: float, diffusion factor
        alpha: float, proliferation factor
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
                         init_density_func, init_learnable_params,
                         init_other_params, init_density_deriv,
                         init_param_min, init_param_max,
                         False, dtype, device)

        # Parameters
        self.nparam_main = 2
        self.nparam = self.nparam_main + self.nparam_init

        D = torch.as_tensor(D, **self.factory_args)
        alpha = torch.as_tensor(alpha, **self.factory_args)
        init_params = self.init_learnable_params.clone()
        self.parameters = torch.cat(
            [D.view(-1), alpha.view(-1), init_params.view(-1)], dim=0)

        # Settings for the training process
        # parameters with proper resizing are called working parameters
        self.wkparams_min = torch.cat([
            torch.tensor([1e-4, 1e-4], **self.factory_args),
            self.init_param_min.view(-1).clone()])
        self.wkparams_max = torch.cat([
            1e3 * D.view(-1).clone(),
            1e3 * alpha.view(-1).clone(),
            self.init_param_max.view(-1).clone()])
        self.wkparams_scale = torch.cat([
            D.view(-1).clone(),
            alpha.view(-1).clone(),
            (self.init_param_max - self.init_param_min).view(-1).clone()])

    @property
    def D(self):
        return self._D(self.parameters)

    def _D(self, params):
        return params[0]

    @property
    def alpha(self):
        return self._alpha(self.parameters)

    def _alpha(self, params):
        return params[1]

    @property
    def init_params(self):
        return self._init_params(self.parameters)

    def _init_params(self, params):
        return params[2:]

    def _format_parameters(self, D: Optional[TensorLikeFloat] = None,
                           alpha: Optional[TensorLikeFloat] = None,
                           init_params: Optional[TensorLikeFloat] = None) -> Tensor:
        """ Helper function to format a new set parameters.
        If all None, return the current parameters. """
        D = self.D if D is None else torch.as_tensor(D, **self.factory_args)
        alpha = self.alpha if alpha is None else torch.as_tensor(alpha, **self.factory_args)
        init_params = self.init_params if init_params is None else torch.as_tensor(init_params, **self.factory_args)
        params = torch.cat(
            [D.view(-1), alpha.view(-1), init_params.view(-1)], dim=0)
        assert params.numel() == self.nparam
        return params

    # def _update_parameters(self, params: Tensor) -> None:
    #     """ Update the parameters. """
    #     assert params.numel() == self.nparam
    #     self.parameters = params.view(-1)

    # def _to_working_params(self, params: Tensor) -> Tensor:
    #     """ Convert params from original scale to the working scale"""
    #     wkparams = (params - self.wkparams_min) / self.wkparams_scale
    #     return wkparams

    # def _to_original_params(self, wkparams: Tensor) -> Tensor:
    #     """ Convert working parameters to their original scale"""
    #     params = wkparams * self.wkparams_scale + \
    #         self.wkparams_min
    #     return params

    def solve(self, dt: float = 0.01, t1: float = 1.,
              D: Optional[TensorLikeFloat] = None, alpha: Optional[TensorLikeFloat] = None,
              init_params: Optional[TensorLikeFloat] = None,
              verbose: bool = True, save_all: bool = False,
              plot_func: Optional[Callable] = None,
              plot_period: int = 10,
              plot_args: Optional[dict] = None,
              save_period: int = 50,
              save_dir: Optional[str] = None,
              save_args: Optional[dict] = None
        ) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        dt = float(dt)
        t1 = float(t1)
        params = self._format_parameters(D, alpha, init_params)
        D = self._D(params)
        alpha = self._alpha(params)
        init_params = self._init_params(params)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, nt, t1 = _spec_dt(dt, t_span, self.dx.tolist(), D.item())

        # initial state
        u = self.init_density(self.x_mesh, init_params, **self.init_other_params)
        if save_all:
            u_hist = torch.zeros((nt, *self.nx), **self.factory_args)
            u_hist[0] = u.clone()
        else:
            u_hist = torch.empty(0)

        if plot_args is None:
            plot_args = {}
        assert isinstance(plot_args, dict)
        if save_args is None:
            save_args = {}
        assert isinstance(save_args, dict)

        # finite difference forward
        ti = t0
        for i in tqdm(range(nt-1), desc="Forward Simulation", disable=not verbose):
            self._forward_update(u, dt, D, alpha)
            self._clip_state(u)
            u.mul_(self.domain_mask)

            ti += dt
            if save_all:
                u_hist[i+1] = u.clone()

            if plot_func is not None and i % plot_period == 0:
                curr_t = round(ti, ndigits=4)
                plot_func(u, i, curr_t, **plot_args)

            if save_dir is not None and i % save_period == 0:
                nifti_data = nib.Nifti1Image(
                    u.detach().cpu().numpy(), save_args['affine'], save_args['header'])
                nib.save(nifti_data, f"{save_dir}/{save_args['patient']}-i{i}.nii.gz")

        return u, torch.linspace(t0, t1, nt), u_hist

    def _forward_update(self, u: Tensor, dt: float, D: Tensor, alpha: Tensor) -> None:

        _, _, c_sl = _get_slicing_positions(self.dim)
        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, self.diff_map_aux)

        du = dt * (D * nab_d_nab_u + alpha * u[c_sl] * (1 - u[c_sl]))
        u[c_sl] += du
        # TODO: enforce boundary conditions

    def solve_with_grad(self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
                        D: Optional[TensorLikeFloat] = None, alpha: Optional[TensorLikeFloat] = None,
                        init_params: Optional[TensorLikeFloat] = None) -> Tuple[Tensor, Tensor, Tensor]:

        # settings
        params = self._format_parameters(D, alpha, init_params)
        D = self._D(params)
        alpha = self._alpha(params)
        init_params = self._init_params(params)
        t0 = 0.
        dt = float(dt)
        t1 = float(t1)
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        # dt, nt, t1 = _spec_dt(dt, t_span, self.dx, D)

        return self._solve_with_grad(obs, dt, t1, D, alpha, init_params)

    def _solve_with_grad(self, obs: Tensor, dt: float, t1: float,
                         D: Tensor, alpha: Tensor, init_params: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        # initial state
        u = self.init_density(self.x_mesh, init_params, **self.init_other_params)
        phi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d D
        psi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d alpha
        eta = None if self.nparam_init == 0 else self.init_density_deriv(
            self.x_mesh, init_params, **self.init_other_params)  # eta = d u / d init_params

        # finite difference forward
        t0 = 0.
        nt = int((t1 - t0) / dt) + 1
        for _ in range(nt-1):
            self._forward_with_sensitivities(u, phi, psi, eta, dt, D, alpha)
            self._clip_state(u)
            u.mul_(self.domain_mask)
            phi.mul_(self.domain_mask)
            psi.mul_(self.domain_mask)
            if eta is not None:
                eta.mul_(self.domain_mask)

        # gradients to parameters
        grads = self._sensitivities_to_grads(u, phi, psi, eta, obs)
        return u, torch.linspace(t0, t1, nt), grads

    def _forward_with_sensitivities(
        self, u: Tensor, phi: Tensor, psi: Tensor, eta: Optional[Tensor],
        dt: float, D: Tensor, alpha: Tensor) -> None:

        _, _, c_sl = _get_slicing_positions(self.dim)

        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, self.diff_map_aux)

        du = dt * (D * nab_d_nab_u + alpha * u[c_sl] * (1 - u[c_sl]))

        dphi = dt * (
            nab_d_nab_u + D * _nabla_d_nabla_f(phi, self.idx2, self.diff_map_aux) +
            alpha * (1 - 2 * u[c_sl]) * phi[c_sl])
        phi[c_sl] += dphi

        dpsi = dt * (
            D * _nabla_d_nabla_f(psi, self.idx2, self.diff_map_aux) +
            u[c_sl] * (1 - u[c_sl]) +
            alpha * (1 - 2 * u[c_sl]) * psi[c_sl]
        )
        psi[c_sl] += dpsi

        if eta is not None:
            deta = dt * (
                D * _nabla_d_nabla_f(eta, self.idx2, self.diff_map_aux) +
                alpha * (1 - 2 * u[c_sl]) * eta[[slice(None, None)] + c_sl]
            )
            eta[[slice(None, None)] + c_sl] += deta

        u[c_sl] += du
        # TODO: enforce boundary conditions

    def _sensitivities_to_grads(
            self, u: Tensor, phi: Tensor, psi: Tensor,
            eta: Optional[Tensor], obs: Tensor) -> Tensor:
        resid = u - obs
        if eta is not None:
            grads = (
                torch.mean(resid * phi), torch.mean(resid * psi),
                torch.mean(resid * eta, dim=tuple(range(1, self.dim+1)))
            )
        else:
            grads = (
                torch.mean(resid * phi), torch.mean(resid * psi)
            )
        return torch.cat([g.view(-1) for g in grads]).to(**self.factory_args)

    def _calibrate_loss_grads(
            self, wkparams: Tensor, obs: Tensor,
            dt: float = 0.01, t1: float = 1.) -> Tuple[Tensor, Tensor]:
        params = self._to_original_params(wkparams)
        u, _, grads = self.solve_with_grad(
            obs, dt, t1, self._D(params), self._alpha(params), self._init_params(params))
        grads *= self.wkparams_scale
        return self._loss_func(u, obs), grads

    def check_gradients(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            D: Optional[TensorLikeFloat] = None,
            alpha: Optional[TensorLikeFloat] = None,
            init_params: Optional[TensorLikeFloat] = None,
            epsilon: float = 1e-6) -> None:

        params = self._format_parameters(D, alpha, init_params)
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

    def calibrate_model(
            self, obs: Tensor, dt: float = 0.01,
            t1: float = 1., max_iter: int = 100,
            method: Literal['L-BFGS-B', 'Nelder-Mead'] = 'L-BFGS-B',
            verbose: bool = True, message_period: int = 10) -> Dict[str, Any]:

        # check inputs
        assert isinstance(obs, Tensor)
        obs = obs.to(**self.factory_args)

        # formulate working parameters
        wkparams = self._to_working_params(self.parameters)
        wkparams_np = wkparams.clone().detach().cpu().numpy()

        loss_history = {'data': []}

        use_grad = True
        # assume methods that accept gradients

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
            "D": self._D(params), "alpha": self._alpha(params),
            "init_params": self._init_params(params), "loss_history": loss_history
        }

        return result

    ###########################################################################
    #########              Multiscan Calibration                ###############
    ###########################################################################

    def _sensitivities_to_grads_multiscan(
            self, u_t: Tensor, phi_t: Tensor, psi_t: Tensor,
            eta_t: Optional[Tensor], obs: Tensor) -> Tensor:
        resid = u_t - obs
        if eta_t is not None:
            grads = (
                torch.mean(resid * phi_t), torch.mean(resid * psi_t),
                torch.mean(resid * eta_t, dim=tuple(range(1, self.dim+2)))
            )
        else:
            grads = (
                torch.mean(resid * phi_t), torch.mean(resid * psi_t)
            )
        return torch.cat([g.view(-1) for g in grads]).to(**self.factory_args)

    def solve_with_grad_multiscan(
            self, obs: Tensor, dt: float = 0.01, t1: float = 1.,
            D: Optional[FloatLike] = None, alpha: Optional[FloatLike] = None,
            init_params: Optional[NDArrayLikeFloat] = None) -> Tuple[Tensor, List[float], Tensor, Tensor]:

        # settings
        params = self._format_parameters(D, alpha, init_params)
        D = self._D(params)
        alpha = self._alpha(params)
        init_params = self._init_params(params)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # temporal grid for the discretized pde
        dt, _, t1 = _spec_dt(dt, t_span, self.dx.tolist(), D.item())

        nscan = obs.shape[0]
        assert nscan > 1

        return self._solve_with_grad_multiscan(obs, dt, t1, D, alpha, init_params)

    def _solve_with_grad_multiscan(
            self, obs: Tensor, dt: float, t1: float,
            D: Tensor, alpha: Tensor, init_params: Tensor) -> Tuple[Tensor, List[float], Tensor, Tensor]:

        t0 = 0.
        nt = int((t1 - t0) / dt) + 1

        nscan = obs.shape[0]
        mses = [float("Inf") for _ in range(nscan-1)]
        minse_t_scan = [t0 for _ in range(nscan-1)]
        minse_t_id = [0 for _ in range(nscan-1)]

        # initial state
        u = self.init_density(
            self.x_mesh, init_params, **self.init_other_params)
        phi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d D
        psi = torch.zeros(self.nx, **self.factory_args)  # phi = d u / d alpha
        # eta = d u / d init_params
        eta = None if self.nparam_init==0 else self.init_density_deriv(
            self.x_mesh, init_params, **self.init_other_params)
        u_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        phi_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        psi_t = torch.zeros((nscan, *self.nx), **self.factory_args)
        if eta is not None:
            eta_t = torch.zeros((self.dim, nscan, *self.nx), **self.factory_args)
        else:
            eta_t = None

        # finite difference forward
        ti = t0
        for i in range(nt-1):
            self._forward_with_sensitivities(
                u, phi, psi, eta, dt, D, alpha)
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
                    if eta is not None and eta_t is not None:
                        eta_t[:, k, ...] = eta.clone()

        u_t[-1] = u.clone()
        phi_t[-1] = phi.clone()
        psi_t[-1] = psi.clone()
        if eta is not None and eta_t is not None:
            eta_t[:, -1, ...] = eta.clone()
        minse_t_scan.append(t1)

        # sort the scans, make sure t_scan in ascending order
        sorted_indices = sorted(range(
            len(minse_t_scan)), key=lambda i: minse_t_scan[i])
        minse_t_scan = [minse_t_scan[i] for i in sorted_indices]
        u_t = u_t[sorted_indices]
        phi_t = phi_t[sorted_indices]
        psi_t = psi_t[sorted_indices]
        if eta_t is not None:
            eta_t = eta_t[:, sorted_indices, ...]

        # grad and loss
        loss = torch.tensor(0., **self.factory_args)
        for k in range(nscan):
            loss += self._loss_func(u_t[k], obs[k])
        grads = self._sensitivities_to_grads_multiscan(
            u_t, phi_t, psi_t, eta_t, obs)

        return u_t, minse_t_scan, loss, grads

    def _calibrate_loss_grads_multiscan(
            self, wkparams: Tensor, obs: Tensor,
            dt: float = 0.01, t1: float = 1.) -> Tuple[List[float], Tensor, Tensor]:
        params = self._to_original_params(wkparams)
        _, t_scan, loss, grads = self.solve_with_grad_multiscan(
            obs, dt, t1, self._D(params), self._alpha(params), self._init_params(params))
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
        wkparams = self._to_working_params(self.parameters)
        wkparams_np = wkparams.clone().detach().cpu().numpy()

        loss_history = {'data': []}  # To store the loss value

        # assume methods that accept gradients
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
            "D": self._D(params), "alpha": self._alpha(params),
            "init_params": self._init_params(params), "loss_history": loss_history,
            "t_scan": t_scan
        }

        return result
