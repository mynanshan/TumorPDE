import torch
from torch import Tensor

# loss functions
# Arguments:
#   x: should be a density tensor taking values in [0, 1]
#   y: same shape as x, binary labels or continuous labels in [0,1]
#   loss_params: learnable parameters in the loss function
#   ...: other parameters

def empty_loss_deriv(x: Tensor) -> Tensor:
    """
    Args:
        x: Tensor, the input tensor.
    Returns:
        Tensor, an empty tensor of shape (0, *voxel_shape).
    """
    return torch.empty((0, *x.shape), device=x.device, dtype=x.dtype)

def mse_loss(x: Tensor, y: Tensor,
    loss_params: Tensor | None = None) -> Tensor:
    return 0.5 * torch.mean((x - y)**2)

def deriv_mse_loss(
    x: Tensor, y: Tensor,
    loss_params: Tensor | None = None) -> Tensor:
    return (x - y) / x.numel()

def logistic_loss(x: Tensor, y: Tensor,
    loss_params: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    return -torch.mean(y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps))

def deriv_logistic_loss(x: Tensor, y: Tensor,
    loss_params: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    return -(y / (x + eps) - (1 - y) / (1 - x + eps)) / x.numel()

def concentrate_density(
    u: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    ) -> Tensor:
    # "concentrate" a density curve
    assert torch.all(u >= 0) and torch.all(u <= capacity)
    assert order > 1.
    density = torch.where(
        u <= thresh,
        thresh * (u / thresh)**order,
        capacity - (capacity - thresh) * ((capacity - u) / (capacity - thresh))**order,
    )
    return density

def deriv_concentrate_density(
    u: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    ) -> Tensor:
    density = torch.where(
        u <= thresh,
        order * (u / thresh)**(order-1),
        order * ((capacity - u) / (capacity - thresh))**(order-1),
    )
    return density

def concentrated_mse_loss(
    x: Tensor, y: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    ) -> Tensor:
    x_trans = concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    return mse_loss(x_trans, y)

def deriv_concentrated_mse_loss(
    x: Tensor, y: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    ) -> Tensor:
    x_trans = concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    d_mse_loss = deriv_mse_loss(x_trans, y)
    d_trans = deriv_concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    return d_mse_loss * d_trans

def concentrated_logistic_loss(
    x: Tensor, y: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    eps: float = 1e-8) -> Tensor:
    x_trans = concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    return logistic_loss(x_trans, y, eps=eps)

def deriv_concentrated_logistic_loss(
    x: Tensor, y: Tensor,
    thresh: float | Tensor = 0.5,
    capacity: float | Tensor = 1.,
    order: float = 2.,
    eps: float = 1e-8) -> Tensor:
    x_trans = concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    d_logi_loss = deriv_logistic_loss(x_trans, y, eps=eps)
    d_trans = deriv_concentrate_density(x, thresh=thresh, capacity=capacity, order=order)
    return d_logi_loss * d_trans