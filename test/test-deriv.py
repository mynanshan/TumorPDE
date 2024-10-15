import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the default_init_density and default_init_density_deriv functions
def default_init_density(x: torch.Tensor, x0: torch.Tensor, w: float, h: float, rmax: float = 1.,
                         log_transform: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape (dim, N), the input tensor.
        x0: Tensor, shape (dim,), the initial position of the tumor.
        w: float, the width of the Gaussian.
        h: float, the height of the Gaussian.
        rmax: float, the maximum radius of the Gaussian.
        log_transform: bool, whether to apply a logit transform to the density.
        eps: float, the epsilon value for the logit transform.
    Returns:
        Tensor, the density value, shape (N,).
    """
    dim = x0.numel()
    assert x.shape[0] == dim
    diff = x - x0.view(-1, 1)  # Shape: (dim, N)
    r2 = torch.sum(diff ** 2, dim=0)  # Shape: (N,)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= rmax**2)
    if log_transform:
        density_value = (1 - eps) * density_value + 0.5 * eps
        density_value = torch.logit(density_value)
    return density_value

def default_init_density_deriv(
        x: torch.Tensor, x0: torch.Tensor, w: float, h: float, rmax: float = 3.,
        to_x0: bool = False) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape (dim, N), the input tensor.
        x0: Tensor, shape (dim,), the initial position of the tumor.
        w: float, the width of the Gaussian.
        h: float, the height of the Gaussian.
        rmax: float, the maximum radius of the Gaussian.
        to_x0: bool, whether to return the derivative with respect to x0.
    Returns:
        Tensor, the derivative of the density with respect to x0, shape (dim, N).
    """
    dim = x0.numel()
    assert x.shape[0] == dim
    diff = x - x0.view(-1, 1)  # Shape: (dim, N)
    r2 = torch.sum(diff ** 2, dim=0)  # Shape: (N,)
    density_value = torch.exp(-r2 / w**2) * h * (r2 <= rmax**2)
    deriv = (-2 / w**2) * diff * density_value  # Shape: (dim, N)
    if to_x0:
        deriv = -deriv  # derivative w.r.t. x0
    return deriv

# Experiment parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

dim = 1  # 1D case for simplicity
N = 1000  # Number of spatial points
x_min, x_max = 0.0, 10.0
x = torch.linspace(x_min, x_max, N, device=device).unsqueeze(0)  # Shape: (1, N)

# Parameters for the density function
w = 0.5  # Width of the Gaussian
h = 1.0  # Height of the Gaussian
rmax = 3.0  # Maximum radius
x0 = torch.tensor([5.0], device=device)  # Initial position

# Compute the density at x
density = default_init_density(x, x0, w, h, rmax)

# Compute the analytical derivative w.r.t. x0
density_deriv_analytical = default_init_density_deriv(x, x0, w, h, rmax, to_x0=True)

# Compute the numerical derivative using finite differences
delta = 1e-5  # Small perturbation
x0_plus = x0 + delta
x0_minus = x0 - delta

density_plus = default_init_density(x, x0_plus, w, h, rmax)
density_minus = default_init_density(x, x0_minus, w, h, rmax)

density_deriv_numerical = (density_plus - density_minus) / (2 * delta)  # Shape: (N,)

# Compute the difference between analytical and numerical derivatives
error = density_deriv_analytical.squeeze(0) - density_deriv_numerical  # Shape: (N,)

# Compute maximum and average errors
max_error = torch.max(torch.abs(error)).item()
mean_error = torch.mean(torch.abs(error)).item()

print(f"Maximum error between analytical and numerical derivatives: {max_error:.6e}")
print(f"Mean error between analytical and numerical derivatives: {mean_error:.6e}")

# Plot the derivatives
x_np = x.squeeze(0).cpu().numpy()
density_deriv_analytical_np = density_deriv_analytical.squeeze(0).cpu().numpy()
density_deriv_numerical_np = density_deriv_numerical.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(x_np, density_deriv_analytical_np, label='Analytical Derivative')
plt.plot(x_np, density_deriv_numerical_np, '--', label='Numerical Derivative')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Comparison of Analytical and Numerical Derivatives')
plt.legend()
plt.grid(True)
plt.show()

# Plot the error
plt.figure(figsize=(10, 6))
plt.plot(x_np, error.cpu().numpy())
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error between Analytical and Numerical Derivatives')
plt.grid(True)
plt.show()
