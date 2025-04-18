from typing import List, Tuple
from numpy.typing import NDArray

import torch
from torch import Tensor
from distmap import euclidean_signed_transform


def focus_slice(x: Tensor,
                focus: int,
                focus_start: int | None = None,
                focus_end: int | None = None,
                focus_by: int | None = None,
                other_start: int | None = None,
                other_end: int | None = None, 
                other_by: int | None = None) -> Tensor:
    """
    Slice an array with a list of slices.
    Use one slice for the focused dimension and another slice for the other dimensions.
    """
    # TODO: slicing in the current codebase is quite messy, need to refactor
    sl = [slice(other_start, other_end, other_by)] * x.ndim
    sl[focus] = slice(focus_start, focus_end, focus_by)
    return x[sl]


def _get_slicing_positions(dim: int, no_boundary: bool = True) -> \
    Tuple[List[List[slice]], List[List[slice]], List[slice]]:

    if no_boundary:

        p1_slices = []  # plus-one slices for all dims
        m1_slices = []  # minus-one slices for all dims
        c_slice = [slice(1, -1)] * dim  # central slice

        for i in range(dim):

            # right/left-shifted spatial slice
            s_m1 = [slice(1, -1)] * dim
            s_p1 = [slice(1, -1)] * dim
            s_m1[i] = slice(None, -2)
            s_p1[i] = slice(2, None)

            p1_slices.append(s_p1)
            m1_slices.append(s_m1)
    
    else:
    
        p1_slices = []  # plus-one slices for all dims
        m1_slices = []  # minus-one slices for all dims
        c_slice = [slice(None, None)] * dim  # central slice

        for i in range(dim):

            # right/left-shifted spatial slice
            s_m1 = [slice(None, None)] * dim
            s_p1 = [slice(None, None)] * dim
            s_m1[i] = slice(None, -1)
            s_p1[i] = slice(1, None)

            p1_slices.append(s_p1)
            m1_slices.append(s_m1)
    
    return p1_slices, m1_slices, c_slice


def _fd_auxiliaries(d: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
    r"""
    Auxiliaries for numerical approximation of the diffusion term on a grid
    Input: d(x), diffusivity field on the grid, shape (n_1, ..., n_d)
    Output:
        The following values on the central grid:
        dp_d = d(x + 0.5\delta x) + d(x)
        dm_d = d(x - 0.5\delta x) + d(x)
        Shape (n_1 - 2, ..., n_d - 2)
    """
    # FIXME: this should be replaced by _half_points_eval

    dim = d.ndim

    dp_d = []  # d^{+} + d
    dm_d = []  # d^{-} + d

    p1_sl, m1_sl, c_sl = _get_slicing_positions(dim)

    for i in range(dim):

        dp_d.append(0.5 * (d[p1_sl[i]] + d[c_sl]))
        dm_d.append(0.5 * (d[m1_sl[i]] + d[c_sl]))

    return dp_d, dm_d


def _half_points_eval(u: Tensor) -> List[Tensor]:
    r"""
    Auxiliaries for numerical approximation of a field on half-point grids
    Input: u(x_1, ..., x_d), field on the grid, shape (n_1, ..., n_d)
    Output:
        The following values on the central grid:
        d(x + 0.5\delta x, ...)
        Shape (n_1 - 1, n_2, ..., n_d)
    """

    u_hp = []  # u^{+1/2}

    p1_sl, m1_sl, _ = _get_slicing_positions(u.ndim, no_boundary=False)

    for i in range(u.ndim):

        u_hp.append(0.5 * (u[p1_sl[i]] + u[m1_sl[i]]))

    return u_hp


def _nabla_d_nabla_f(f: Tensor, idx2: Tensor, d_aux: Tuple[List[Tensor], List[Tensor]]) -> Tensor:
    r"""
    Numerical approximation of \nabla \cdot (d \nabla f) grid points
    Input: f(x) values on the grid, shape (n_1, n_2, ..., n_d)
    Output: \nabla \cdot (d \nabla f) on the central grid, shape (n_1 - 2, ..., n_d - 2)
    """

    dp_d, dm_d = d_aux

    # detect settings from dp_d[0]
    dim = dp_d[0].ndim
    nx_center = dp_d[0].shape
    factory_args = {"dtype": dp_d[0].dtype, "device": dp_d[0].device}

    p1_sl, m1_sl, c_sl = _get_slicing_positions(dim)

    if dim == f.ndim:
        if torch.any(torch.as_tensor(f.shape) - 2 !=
                     torch.as_tensor(nx_center)):
            raise ValueError("Incorrect shape of f")

        nab_d_nab_f = torch.zeros(
            *nx_center, **factory_args)

        for i in range(dim):
            nab_d_nab_f += idx2[i] * (
                dp_d[i] * (f[p1_sl[i]] - f[c_sl]) +
                dm_d[i] * (f[m1_sl[i]] - f[c_sl])
            )

    elif dim == f.ndim - 1:
        if torch.any(torch.as_tensor(f.shape[1:]) - 2 !=
                     torch.as_tensor(nx_center)):
            raise ValueError("Incorrect shape of f")

        no_sl = [slice(None, None)]
        nab_d_nab_f = torch.zeros(
            f.shape[0], *nx_center, **factory_args)

        for i in range(dim):
            # broadcasting will appropriately expand the shape (*nx-2) to (m, *nx-2)
            nab_d_nab_f += idx2[i] * (
                dp_d[i] * (f[no_sl + p1_sl[i]] - f[no_sl + c_sl]) +
                dm_d[i] * (f[no_sl + m1_sl[i]] - f[no_sl + c_sl])
            )
    else:
        raise ValueError("Dimension mismatch between 'f' and 'dim'")

    return nab_d_nab_f

# def _v_dot_nabla_u(v: Tensor, u: Tensor, dx: Tensor) -> Tensor:
#     r"""
#     Numerical approximation of v \cdot \nabla u on a grid
#     Input: v(x), velocity field on the grid, shape (dim, n_1, n_2, ..., n_d)
#            u(x), field on the grid, shape (n_1, n_2, ..., n_d)
#     Output: v \cdot \nabla u on the central grid, shape (n_1 - 2, ..., n_d - 2)
#     """
#     # TODO: change to WENO5 in the future, currently using central differences

#     dim = u.ndim

#     p2_sl, m2_sl, c_sl = _get_slicing_positions(dim)

#     nab_v_u = torch.zeros_like(u, dtype=u.dtype, device=u.device)

#     for i in range(dim):
#         nab_v_u[c_sl] += v[i][c_sl] * (u[p2_sl[i]] - u[m2_sl[i]]) / (2 * dx[i])

#     return nab_v_u[c_sl]

def _v_dot_nabla_u(v: Tensor, u: Tensor, dx: Tensor) -> Tensor:
    nab_v_u = torch.zeros_like(u)
    dim = v.shape[0]
    p_sl, m_sl, c_sl = _get_slicing_positions(dim)
    
    for i in range(dim):
        # Upwind scheme
        v_pos = torch.where(v[i][c_sl] > 0, 1, 0)
        v_neg = 1 - v_pos
        
        # Backward difference for positive velocity, forward for negative
        grad_u_i = (v_pos * (u[c_sl] - u[m_sl[i]]) / dx[i] +
                    v_neg * (u[p_sl[i]] - u[c_sl]) / dx[i])
        
        nab_v_u[c_sl] += v[i][c_sl] * grad_u_i
        
    return nab_v_u

def _phase_field(mask: Tensor | NDArray, margin: float = 1.):

    mask = torch.as_tensor(mask)
    signed_dist = euclidean_signed_transform(mask)
    phase = 0.5 * (1. + torch.tanh(3 * signed_dist / margin))

    return phase


def _spec_dt(dt: float, t_span: float,
             dx: List[float], D: float) -> Tuple[float, int, float]:

    dt = float(dt)
    t_span = float(t_span)
    D = float(D)

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


def _neumann_rect(x: Tensor) -> None:
    """Enforce Neumann boundary condition for a field defined
    on a rectangular grid."""
    xdims = x.shape
    ndim = x.ndim
    for i in range(ndim):
        for j in [0, xdims[i] - 1]:
            # boundary indices
            bnd_sl = [slice(None, None)] * ndim
            bnd_sl[i] = slice(j, j+1)

            # neighbour indices
            k = 1 if j ==0 else xdims[i]-2
            neb_sl = [slice(None, None)] * ndim
            neb_sl[i] = slice(k, k+1)

            x[bnd_sl] = x[neb_sl]

