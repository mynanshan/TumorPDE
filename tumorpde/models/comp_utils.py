from typing import List, Tuple

import torch
from torch import Tensor

def _get_slicing_positions(dim: int) -> Tuple[List[slice], List[slice], List[slice]]:

    p1_slices = []
    m1_slices = []
    c_slice = [slice(1, -1)] * dim
    # c_slice = tuple([slice(1, -1)] * dim)

    for i in range(dim):

        # right/left-shifted spatial slice
        s_m1 = [slice(1, -1)] * dim
        s_p1 = [slice(1, -1)] * dim
        s_m1[i] = slice(None, -2)
        s_p1[i] = slice(2, None)
        # s_m1 = tuple(s_m1)
        # s_p1 = tuple(s_p1)

        p1_slices.append(s_p1)
        m1_slices.append(s_m1)

    return p1_slices, m1_slices, c_slice

def _diff_map_auxiliaries(d: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
    r"""
    Auxiliaries for numerical approximation of the diffusion term on a grid
    Input: d(x), diffusivity field on the grid, shape (n_1, ..., n_d)
    Output:
        The following values on the central grid:
        dp_d = d(x + 0.5\delta x) + d(x)
        dm_d = d(x - 0.5\delta x) + d(x)
        Shape (n_1 - 2, ..., n_d - 2)
    """

    dim = d.ndim

    dp_d = []  # d^{+} + d
    dm_d = []  # d^{-} + d

    # slicing helpers
    p1_sl, m1_sl, c_sl = _get_slicing_positions(dim)

    for i in range(dim):

        dp_d.append(d[p1_sl[i]] + d[c_sl])
        dm_d.append(d[m1_sl[i]] + d[c_sl])

    return dp_d, dm_d

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
            nab_d_nab_f += 0.5 * idx2[i] * (
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
            nab_d_nab_f += 0.5 * idx2[i] * (
                dp_d[i] * (f[no_sl + p1_sl[i]] - f[no_sl + c_sl]) +
                dm_d[i] * (f[no_sl + m1_sl[i]] - f[no_sl + c_sl])
            )
    else:
        raise ValueError("Dimension mismatch between 'f' and 'dim'")

    return nab_d_nab_f

