import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple
from numpy.typing import NDArray

def signed_distance_and_nearest_index(
    mask: NDArray[np.bool_]
) -> Tuple[NDArray[np.float64], NDArray[np.intp]]:
    """
    Compute signed Euclidean distance transform and nearest‑boundary indices.

    Parameters
    ----------
    mask : np.ndarray of bool
        Input mask: True=inside, False=outside.

    Returns
    -------
    signed_d : np.ndarray of float64
        Signed distance map: negative inside, positive outside.
    idx      : np.ndarray of intp
        C‑order linear index of the nearest boundary (mask==False) voxel.
    """
    # 1) Ensure boolean
    m: NDArray[np.bool_] = mask.astype(bool)

    # 2) Outside distances + nearest‑zero indices
    res_out = distance_transform_edt(
        ~m,
        return_distances=True,
        return_indices=True
    )  # type: ignore[call-arg]
    d_out, inds_out = res_out  # type: ignore[assignment]

    # 3) Inside distances + nearest‑zero indices
    res_in = distance_transform_edt(
        m,
        return_distances=True,
        return_indices=True
    )  # type: ignore[call-arg]
    d_in, inds_in = res_in  # type: ignore[assignment]

    # 4) Signed distance
    signed_d: NDArray[np.float64] = d_out - d_in

    # 5) Stack per‑axis index arrays → shape (ndim, *mask.shape)
    inds_out_arr = np.stack(inds_out, axis=0)
    inds_in_arr  = np.stack(inds_in,  axis=0)

    # 6) Select inside vs outside indices
    sel = m[None, ...]  # broadcastable to (ndim, *shape)
    inds_sel = np.where(sel, inds_in_arr, inds_out_arr)

    # 7) Convert multi‑dim indices → linear C‑order index array
    linear_idx = np.ravel_multi_index(
        tuple(inds_sel),
        dims=mask.shape
    )  # type: ignore[assignment]

    idx_arr: NDArray[np.intp] = linear_idx  # type: ignore[assignment]

    return signed_d, idx_arr
