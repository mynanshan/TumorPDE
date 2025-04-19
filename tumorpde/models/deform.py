from typing import Optional, Callable, Dict, Any, Literal, Tuple, List
import math
import numpy as np
import torch
from torch import Tensor
from scipy.optimize import minimize
from tqdm import tqdm
import nibabel as nib

from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike
from tumorpde.volume_domain import VolumeDomain
from tumorpde.models._base import TumorVarFieldBase
from tumorpde.models.comp_utils import _get_slicing_positions
from tumorpde.models.comp_utils import _nabla_d_nabla_f, _v_dot_nabla_u
from tumorpde.models.comp_utils import _fd_auxiliaries
from tumorpde.calc.linalg import CG

"""
TODO: critical rethinking: is the current model realistic enough?
- Lipkova's tumor is not like a tissue: it does not directly interact, push, or squeeze other tissues.
Instead, its growth creates a pressure field that pushes the surrounding tissues.
"""


class TumorBrainDeformState:

    def __init__(self,
                 domain_mask: Tensor,
                 time: float,
                 tumor_density: Tensor,
                 brain_density: Tensor,
                 deform_velocity: Tensor,
                 pressure_field: Tensor,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cpu')):
        # TODO: probably add an argument validation
        self.domain_mask = domain_mask
        self.dim = domain_mask.ndim
        self.time = time
        factor_args = {"dtype": dtype, "device": device}
        self.factory_args = factor_args
        self.tumor_density = torch.as_tensor(tumor_density, **factor_args)
        self.brain_density = torch.as_tensor(brain_density, **factor_args)
        self.deform_velocity = torch.as_tensor(deform_velocity, **factor_args)
        self.pressure_field = torch.as_tensor(pressure_field, **factor_args)

        # check whether brain matters fill the whole domain
        if torch.sum(torch.logical_and(
            self.domain_mask,
            torch.logical_not(self.matter_mask)
        )) > 0:
            raise ValueError("Brain matters do not fill the whole domain.")

    @property
    def brain_proportion(self):
        brain_sum = torch.sum(self.brain_density, dim=0)
        prop = torch.zeros((3, *self.domain_mask.shape), **self.factory_args)
        prop[:, self.domain_mask] = self.brain_density[:, self.domain_mask] / \
            brain_sum[self.domain_mask].unsqueeze(0)
        return prop

    @property
    def tissue_mask(self):
        return torch.logical_or(self.brain_density[0] > 0.,
                                self.brain_density[1] > 0.)

    @property
    def matter_mask(self):
        return torch.sum(self.brain_density, dim=0) > 0.

    def to(self, dtype: torch.dtype | None = None,
           device: torch.device | None = None):
        self.factory_args = {"dtype": dtype, "device": device}
        self.tumor_density = self.tumor_density.to(**self.factory_args)
        self.brain_density = self.brain_density.to(**self.factory_args)
        self.deform_velocity = self.deform_velocity.to(**self.factory_args)
        self.pressure_field = self.pressure_field.to(**self.factory_args)


class TumorDeformFD(TumorVarFieldBase):

    def __init__(self,
                 domain: VolumeDomain,
                 matters: Tensor,
                 D: TensorLikeFloat = 1.,
                 alpha: TensorLikeFloat = 1.,
                 M: TensorLikeFloat = 1.,
                 kappa: TensorLikeFloat = 1.,
                 D_ratio: TensorLikeFloat = 0.1,
                 kappa_ratios: Optional[TensorLikeFloat] = [0.01, 0.1],
                 estimate_ratios: bool = False,
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
        M: float, hydraulic conductivity factor
        kappa: pressure relaxation factor
        D_ratio: diffusion rate ratio, gm / wm
        kappa_ratios: pressure relaxation ratio, (gm, wm) / csf
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

        super().__init__(domain, matters,
                         init_density_func, init_learnable_params,
                         init_other_params, init_density_deriv,
                         init_param_min, init_param_max,
                         False, dtype, device)

        # Parameters
        self.nparam_main = 4 + 1 + 2  # D, alpha, M, kappa, and ratios
        self.nparam = self.nparam_main + self.nparam_init

        D = torch.as_tensor(D, **self.factory_args)
        alpha = torch.as_tensor(alpha, **self.factory_args)
        M = torch.as_tensor(M, **self.factory_args)
        kappa = torch.as_tensor(kappa, **self.factory_args)
        D_ratio = torch.as_tensor(D_ratio, **self.factory_args)
        kappa_ratios = torch.as_tensor(kappa_ratios, **self.factory_args)
        init_params = self.init_learnable_params.clone()

        # parameters indexes:
        # 0 - D
        # 1 - alpha
        # 2 - M
        # 3 - kappa
        # 4:4+nparam_init - init_params
        # 4+nparam_init:4+nparam_init+1 - D_ratio
        # 4+nparam_init+1:4+nparam_init+3 - kappa_ratios
        parameters = self._format_parameters(
            D, alpha, M, kappa,
            D_ratio, kappa_ratios,
            init_params
        )
        self._set_parameters(parameters)

        # Whether to estimate the ratios
        self.estimate_ratios = estimate_ratios
    
    def _set_parameters(self, parameters: Tensor):

        self.parameters = parameters.flatten()
        assert self.parameters.numel() == self.nparam

        # Settings for the training process
        # parameters with proper resizing are called working parameters
        self.rescale_params = False
        self.wkparams_min = torch.cat([
            torch.tensor([1e-4] * 4, **self.factory_args),
            self.init_param_min.view(-1).clone(),
            torch.tensor([1e-4] * 3, **self.factory_args)])
        self.wkparams_max = torch.cat([
            1e3 * self.D.view(-1).clone(),
            1e3 * self.alpha.view(-1).clone(),
            1e3 * self.M.view(-1).clone(),
            1e3 * self.kappa.view(-1).clone(),
            self.init_param_max.view(-1).clone(),
            torch.tensor([1.] * 3, **self.factory_args)])
        self.wkparams_scale = torch.cat([
            self.D.view(-1).clone(),
            self.alpha.view(-1).clone(),
            self.M.view(-1).clone(),
            self.kappa.view(-1).clone(),
            (self.init_param_max - self.init_param_min).view(-1).clone(),
            torch.tensor([1.] * 3, **self.factory_args)])
        # NOTE: there's no need to use the name "wkparams" for working parametrs.
        # Should be just params. Consider renaming it in the future.

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
    def M(self):
        return self._M(self.parameters)

    def _M(self, params):
        return params[2]

    @property
    def kappa(self):
        return self._kappa(self.parameters)

    def _kappa(self, params):
        return params[3]

    @property
    def init_params(self):
        return self._init_params(self.parameters)

    def _init_params(self, params):
        return params[4:4+self.nparam_init]

    @property
    def D_ratio(self):
        return self._D_ratio(self.parameters)

    def _D_ratio(self, params):
        return params[4+self.nparam_init:4+self.nparam_init+1]

    @property
    def kappa_ratios(self):
        return self._kappa_ratios(self.parameters)

    def _kappa_ratios(self, params):
        return params[4+self.nparam_init+1:4+self.nparam_init+3]

    def _format_parameters(self,
                           D: Optional[Tensor] = None,
                           alpha: Optional[Tensor] = None,
                           M: Optional[Tensor] = None,
                           kappa: Optional[Tensor] = None,
                           D_ratio: Optional[Tensor] = None,
                           kappa_ratios: Optional[Tensor] = None,
                           init_params: Optional[Tensor] = None):
        D = D if D is not None else self.D
        alpha = alpha if alpha is not None else self.alpha
        M = M if M is not None else self.M
        kappa = kappa if kappa is not None else self.kappa
        D_ratio = D_ratio if D_ratio is not None else self.D_ratio
        kappa_ratios = kappa_ratios if kappa_ratios is not None else self.kappa_ratios
        init_params = init_params if init_params is not None else self.init_params
        parameters = torch.cat([
            D.view(-1), alpha.view(-1),
            M.view(-1), kappa.view(-1),
            init_params.view(-1),
            D_ratio.view(-1), kappa_ratios.view(-1)
        ], dim=0)
        return parameters

    def _spec_dt(self, dt: float, t_span: float,
                 dx: List[float], D: float, vmax: float) -> float:

        dt = float(dt)
        t_span = float(t_span)
        D = float(D)
        vmax = float(vmax)

        # check if dt is small enough
        cfl = 0.25  # I chose this randomly
        ref_dt = cfl * min(min(dx)**2 / (D + 1e-4), min(dx) / (vmax + 1e-4))
        if dt > ref_dt:
            dt = ref_dt
            print(f"Change dt to {dt}.")

        return dt

    def solve(self, state: TumorBrainDeformState | None = None,
              dt: float = 0.001, t1: float = 1.,
              D: Optional[Tensor] = None, alpha: Optional[Tensor] = None,
              M: Optional[Tensor] = None, kappa: Optional[Tensor] = None,
              D_ratio: Optional[Tensor] = None,
              kappa_ratios: Optional[Tensor] = None,
              init_params: Optional[Tensor] = None,
              verbose=True):

        # parameters
        parameters = self._format_parameters(
            D, alpha, M, kappa,
            D_ratio, kappa_ratios,
            init_params)

        # settings
        dt0 = float(dt)
        t1 = float(t1)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0

        # initialization
        if state is None:
            state = self.init_state(self._init_params(parameters))
        assert isinstance(state, TumorBrainDeformState)

        # forward
        ti = t0
        t = [t0]
        # TODO: adapt dt dynamically
        while (ti < t1):
            dt = self._spec_dt(dt, t_span, self.dx.tolist(),
                               self._D(parameters).item(),
                               torch.max(torch.abs(state.deform_velocity)).item())
            if dt < dt0 / 100:
                dt = dt0 / 100
                print(
                    f"Fix dt to {dt0} / 100, but smaller dt may be necessary.")
            ti += dt
            t.append(ti)

            self.forward_update(state, dt, parameters)

        return state, t

    def init_state(self, init_params: Tensor) -> TumorBrainDeformState:

        # initial tumor density
        u = self.init_density(self.x_mesh, init_params,
                              **self.init_other_params)

        # initial brain matter densities
        rho = self.matters.clone()

        # inital relative velocity and pressure field
        v = torch.zeros((self.dim, *self.nx), **self.factory_args)
        p = torch.zeros(*self.nx, **self.factory_args)

        return TumorBrainDeformState(self.domain_mask, 0., u, rho,
                                     v, p, **self.factory_args)

    def forward_update(self,
                       state: TumorBrainDeformState,
                       dt: float,
                       parameters: Tensor | None) -> None:
        if parameters is not None:
            parameters = self.parameters
        self._forward_update(dt,
                             state.tumor_density,
                             state.brain_density,
                             state.brain_proportion,
                             state.deform_velocity,
                             state.pressure_field,
                             self._D(parameters),
                             self._alpha(parameters),
                             self._M(parameters),
                             self._kappa(parameters),
                             self._D_ratio(parameters),
                             self._kappa_ratios(parameters))

    def _forward_update(self, dt: float,
                        u: Tensor, rho: Tensor,
                        omega: Tensor, v: Tensor, p: Tensor,
                        D: Tensor, alpha: Tensor,
                        M: Tensor, kappa: Tensor,
                        D_ratio: Tensor, kappa_ratios: Tensor,
                        weno: bool = False) -> None:
        # TODO: add support for weno

        # rho: gm, wm, csf
        # in-place update of u, rho, p, v.

        p_sl, m_sl, c_sl = _get_slicing_positions(self.dim)

        # pressure field
        # the const term F is alpha * u * (1 - u)
        kappa_field = kappa * (
            omega[0] * kappa_ratios[0] +   # gm
            omega[1] * kappa_ratios[1] +   # wm
            omega[2] * 1.                  # csf
        )
        p[self.domain_mask] = self._pressure_field_interior(
            self.M, kappa_field, self.alpha * u * (1 - u))
        p.mul_(self.domain_mask)
        self.pad_boundary(p)

        # deformation velocity field
        for i in range(self.dim):
            v[i][c_sl] = - M * (p[p_sl[i]] - p[m_sl[i]]) / (2 * self.dx[i])
            v[i].mul_(self.domain_mask)
            self.pad_boundary(v[i])

        # diffusion rate field and its finite-difference aux
        diff_field = omega[0] * D_ratio + omega[1]
        diff_field_aux = _fd_auxiliaries(diff_field)

        # update tumor density
        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, diff_field_aux)
        du_dt = D * nab_d_nab_u + alpha * u[c_sl] * (1 - u[c_sl]) - \
            _v_dot_nabla_u(v, u, self.dx)[c_sl]
        u[c_sl] += dt * du_dt
        u.mul_(self.domain_mask)
        self._clip_state(u)
        self.pad_boundary(u)

        # update brain tissue density
        # TODO: change to WENO5 in the future, currently using central differences
        # TODO: warp advection computation into help functions
        drho_dt = torch.zeros((3, *self.nx), **self.factory_args)
        for s in range(3):
            for i in range(self.dim):

                # Upwind scheme for advection term
                v_pos = torch.where(v[i][c_sl] > 0, 1, 0)
                v_neg = 1 - v_pos
                
                # Advection term: -v * ∂rho/∂x
                drho_dx = (v_pos * (rho[s][c_sl] - rho[s][m_sl[i]]) / self.dx[i] +
                        v_neg * (rho[s][p_sl[i]] - rho[s][c_sl]) / self.dx[i])
        
                # Convection term: -rho * ∂v/∂x
                dv_dx = (v[i][p_sl[i]] - v[i][m_sl[i]]) / (2 * self.dx[i])
                
                drho_dt[s][c_sl] -= v[i][c_sl] * drho_dx + rho[s][c_sl] * dv_dx

            # update brain tissue 
            rho[s][c_sl] += dt * drho_dt[s][c_sl]
            rho[s].mul_(self.domain_mask)

            # After updating rho[s]:
            rho_positive = torch.clamp(rho[s], min=0.0)
            mass_loss = torch.sum(rho[s] - rho_positive)
            rho[s] = rho_positive + \
                mass_loss / torch.sum(self.domain_mask) * self.domain_mask  # Redistribute lost mass
            
            # boundary conditions
            self.pad_boundary(rho[s])

    def _pressure_field_interior(self, M: Tensor, Kappa: Tensor, F: Tensor) -> Tensor:

        # p_sl, m_sl: slicing the plus-one/minus-one grids along each dim
        p_sl, m_sl, _ = _get_slicing_positions(self.dim, no_boundary=False)

        phi = self.phase_field
        phi_hp = self.phase_aux  # phi on half points
        sum_idx2 = torch.sum(self.idx2)

        N = self.n_in_points
        ind_interior = self.interior_indices  # in-domain grid point indexes
        mask = self.domain_mask

        rows, cols, vals = [], [], []

        # --- Interior Points ---
        # Diagonal terms: phi*(kappa + 2*M*(1/dx^2 + 1/dy^2 + 1/dz^2))
        diag = phi * (Kappa + 2 * M * sum_idx2)
        rows.append(ind_interior[mask].flatten())
        cols.append(ind_interior[mask].flatten())
        vals.append(diag[mask].flatten())

        # Off-diagonal terms: -phi_{i+1/2,j,k} M / dx^2, ...
        for i in range(self.dim):
            is_p1sl_in_domain = mask[p_sl[i]]
            is_m1sl_in_domain = mask[m_sl[i]]

            # both minus-1 and plus-1 point are in domain
            avail_mask = torch.logical_and(
                is_p1sl_in_domain, is_m1sl_in_domain)

            p_term = ind_interior[m_sl[i]][avail_mask], \
                ind_interior[p_sl[i]][avail_mask], \
                - M/self.dx[i]**2 * phi_hp[i][avail_mask]
            m_term = ind_interior[p_sl[i]][avail_mask], \
                ind_interior[m_sl[i]][avail_mask], \
                - M/self.dx[i]**2 * phi_hp[i][avail_mask]
            for r, c, v in [p_term, m_term]:
                rows.append(r.flatten())
                cols.append(c.flatten())
                vals.append(v.flatten())

        # Concatenate all entries
        rows = torch.cat(rows)
        cols = torch.cat(cols)
        vals = torch.cat(vals)

        # Create sparse matrix
        A = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals,
            (N, N), **self.factory_args
        ).to_sparse_csr()

        # Right-hand side: b = phi * F for interior points
        b = (phi[mask] * F[mask]).flatten()

        # sparse symmetric linear system by conjugate gradient
        sol, _ = CG(A, b)

        return sol.to(**self.factory_args)
