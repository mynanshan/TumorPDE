from typing import Optional, Callable, Dict, Any, Literal, Tuple, List
import math
import numpy as np
import torch
from torch import Tensor
from torch.sparse import spdiags
from scipy.optimize import minimize
from tqdm import tqdm
import nibabel as nib

from tumorpde._typing import TensorLikeFloat, NDArrayLikeFloat, FloatLike
from tumorpde.volume_domain import VolumeDomain
from tumorpde.models._base import TumorVarFieldBase
from tumorpde.models.comp_utils import _get_slicing_positions
from tumorpde.models.comp_utils import _nabla_d_nabla_f, _fd_auxiliaries
from tumorpde.models.comp_utils import _spec_dt
from tumorpde.models.comp_utils import _neumann_rect
from tumorpde.calc.linalg import CG


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
        kappa_ratio: pressure relaxation ratio, (gm, wm) / csf
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
        self.parameters = torch.cat(
            [D.view(-1), alpha.view(-1), M, kappa, init_params.view(-1)], dim=0)
        if estimate_ratios:
            self.parameters = torch.cat(
                [self.parameters, D_ratio.view(-1), kappa_ratios.view(-1)], dim=0)
        # parameters indexes:
        # 0 - D
        # 1 - alpha
        # 2 - M
        # 3 - kappa
        # 4:4+nparam_init - init_params
        # 4+nparam_init:4+nparam_init+1 - D_ratio
        # 4+nparam_init+1:4+nparam_init+3 - kappa_ratio

        # Settings for the training process
        # parameters with proper resizing are called working parameters
        self.rescale_params = False
        self.wkparams_min = torch.cat([
            torch.tensor([1e-4] * 4, **self.factory_args),
            self.init_param_min.view(-1).clone(),
            torch.tensor([1e-4] * 3, **self.factory_args)])
        self.wkparams_max = torch.cat([
            1e3 * D.view(-1).clone(),
            1e3 * alpha.view(-1).clone(),
            1e3 * M.view(-1).clone(),
            1e3 * kappa.view(-1).clone(),
            self.init_param_max.view(-1).clone(),
            torch.tensor([1.] * 3, **self.factory_args)])
        self.wkparams_scale = torch.cat([
            D.view(-1).clone(),
            alpha.view(-1).clone(),
            M.view(-1).clone(),
            kappa.view(-1).clone(),
            (self.init_param_max - self.init_param_min).view(-1).clone(),
            torch.tensor([1.] * 3, **self.factory_args)])

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
        return params[3]

    @property
    def kappa(self):
        return self._kappa(self.parameters)

    def _kappa(self, params):
        return params[4]

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
        return params[4+self.nparam_init:4+self.nparam_init+1]

    def solve(self, dt: float = 0.001, t1: float = 1.,
              D: Optional[Tensor] = None, alpha: Optional[Tensor] = None,
              M: Optional[Tensor] = None, kappa: Optional[Tensor] = None,
              D_ratio: Optional[Tensor] = None, kappa_ratio: Optional[Tensor] = None,
              init_params: Optional[Tensor] = None,
              verbose=True):

        # parameters
        D = D if D is not None else self.D
        alpha = alpha if alpha is not None else self.alpha
        M = M if M is not None else self.M
        kappa = kappa if kappa is not None else self.kappa
        D_ratio = D_ratio if D_ratio is not None else self.D_ratio
        kappa_ratio = kappa_ratio if kappa_ratio is not None else self.kappa_ratios
        init_params = init_params if init_params is not None else self.init_params

        # settings
        dt = float(dt)
        t1 = float(t1)
        t0 = 0.
        assert t1 > t0
        t_span = t1 - t0
        dt, nt, t1 = _spec_dt(dt, t_span, self.dx.tolist(), D.item())

        # initial tumor density
        u = self.init_density(self.x_mesh, init_params,
                              **self.init_other_params)

        # initial brain matter densities
        rho = self.matters.clone()

        # compute brain tissue ratios
        rho_sum = torch.sum(rho, dim=0)
        omega = rho / rho_sum.unsqueeze(0)

        # forward
        ti = t0
        # TODO: adapt dt dynamically
        for i in tqdm(range(nt-1), desc="Forward Simulation", disable=not verbose):
            self._forward_update(
                dt, u, rho, omega,
                D, alpha, M, kappa, D_ratio, kappa_ratio
            )
            ti += dt

    def _forward_update(self, dt: float, u: Tensor,
                        rho: Tensor, omega: Tensor,
                        D: Tensor, alpha: Tensor,
                        M: Tensor, kappa: Tensor,
                        D_ratio: Tensor, kappa_ratio: Tensor,
                        weno: bool = False) -> None:
        # TODO: add support for weno

        # rho: gm, wm, csf
        # in-place update of u, rho and omega.

        p_sl, m_sl, c_sl = _get_slicing_positions(self.dim)

        # TODO: double check boundary conditions for p, v, u, rho

        # pressure field
        # the const term F is alpha * u * (1 - u)
        kappa_field = kappa * (
            omega[0] * kappa_ratio[0] +   # gm
            omega[1] * kappa_ratio[1] +   # wm
            omega[2] * 1.                 # csf
        )
        p = self._pressure_field(self.M, kappa_field, self.alpha * u * (1 - u))
        p.mul_(self.domain_mask)

        # deformation velocity field
        v = torch.zeros((self.dim, *self.nx), **self.factory_args)
        for i in range(self.dim):
            v[i][c_sl] = -0.5 * M / self.dx[i] * (p[p_sl[i]] - p[m_sl[i]])

        # diffusion rate field and its finite-difference aux
        diff_map = omega[0] * D_ratio + omega[1]
        diff_map_aux = _fd_auxiliaries(diff_map)

        # update tumor density
        nab_d_nab_u = _nabla_d_nabla_f(u, self.idx2, diff_map_aux)
        du = dt * (D * nab_d_nab_u + alpha * u[c_sl] * (1 - u[c_sl]))
        u[c_sl] += du
        # _neumann_rect(u)
        u.mul_(self.domain_mask)
        self._clip_state(u)

        # update brain tissue density
        # TODO: change to WENO5 in the future, currently using central differences
        drho_dt = torch.zeros((3, *self.nx), **self.factory_args)
        for s in range(3):
            for i in range(self.dim):
                drho_dt[s][c_sl] -= v[i][c_sl] * (rho[s][p_sl[i]] - rho[s][m_sl[i]]) / 2 + \
                    rho[s][c_sl] * (v[i][p_sl[i]] - v[i][m_sl[i]]) / 2
        for s in range(3):
            rho[s][c_sl] += dt * drho_dt[s][c_sl]
            # _neumann_rect(rho[s])
            rho[s].mul_(self.domain_mask)

        # update brain tissue ratios
        rho_sum = torch.sum(rho, dim=0)
        omega[...] = rho / rho_sum.unsqueeze(0)

    def _pressure_field(self, M: Tensor, Kappa: Tensor, F: Tensor) -> Tensor:

        # c_sl: slicing the interior
        # p2_sl, m2_sl: plus-1/minus-1 slice wrt the interior
        p2_sl, m2_sl, c_sl = _get_slicing_positions(self.dim)

        # NOTE: naming is not consistent, p_sl or p2_sl
        # p_sl, m_sl: slicing the plus-one/minus-one grids along each dim
        p_sl, m_sl, _ = _get_slicing_positions(self.dim, no_boundary=False)

        N = math.prod(list(self.nx))
        phi = self.phase_field
        phi_hp = self.phase_aux  # phi on half points
        sum_idx2 = torch.sum(self.idx2)

        # interior = self.cube_interior
        ind = self.indices

        rows, cols, vals = [], [], []

        # --- Interior Points ---
        # Diagonal terms: phi*(kappa + 2*M*(1/dx^2 + 1/dy^2 + 1/dz^2))
        diag = phi[c_sl] * (Kappa[c_sl] + 2 * M * sum_idx2)
        rows.append(ind[c_sl].flatten())
        cols.append(ind[c_sl].flatten())
        vals.append(diag.flatten())

        for i in range(self.dim):
            # Off-diagonal terms: -phi_{i+1/2,j,k} M / dx^2, ...
            p_term = ind[c_sl], ind[p2_sl[i]], - \
                M/self.dx[i]**2 * phi_hp[i][p_sl]
            m_term = ind[c_sl], ind[m2_sl[i]], - \
                M/self.dx[i]**2 * phi_hp[i][m_sl]
            for r, c, v in [p_term, m_term]:
                rows.append(r.flatten())
                cols.append(c.flatten())
                vals.append(v.flatten())

        # --- Boundary Conditions (Neumann) ---
        # Each boundary point equation: p = neighbor_p
        for i in range(self.dim):
            for j in [0, self.nx[i] - 1]:
                # boundary indices
                bnd_sl = [slice(None, None)] * self.dim
                bnd_sl[i] = slice(j, j+1)
                bnd = ind[bnd_sl].flatten()

                # neighbour indices
                k = 1 if j == 0 else self.nx[i]-2
                neb_sl = [slice(None, None)] * self.dim
                neb_sl[i] = slice(k, k+1)
                neb = ind[neb_sl].flatten()

                # add indices
                rows.append(bnd)
                cols.append(bnd)
                vals.append(torch.ones_like(bnd, dtype=self.dtype))

                rows.append(bnd)
                cols.append(neb)
                vals.append(-torch.ones_like(bnd, dtype=self.dtype))

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
        b = torch.zeros(N, **self.factory_args)
        b[c_sl] = (phi[c_sl] * F[c_sl])
        b = b.flatten()

        sol, _ = CG(A, b)

        return sol.reshape(self.nx).to(**self.factory_args)
