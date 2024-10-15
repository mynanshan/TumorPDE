from torch import Tensor

from typing import Tuple
from tumorpde._typing import FloatLike, TensorLikeFloat


def _spec_dt(dt: FloatLike, t_span: FloatLike,
             dx: TensorLikeFloat, D: FloatLike) -> Tuple[float, int, float]:

    dt = float(dt)
    t_span = float(t_span)
    if isinstance(dx, Tensor):
        dx = dx.tolist()
    else:
        dx = list(dx)
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
