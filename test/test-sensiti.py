import torch
import numpy as np
import matplotlib.pyplot as plt
from tumorpde.models.growth import TumorGrowthFD
from tumorpde.volume_domain import VolumeDomain

# Ensure the proper device is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Define the spatial grid
dx = 0.1
x_range = (0., 10.)
nx = int((x_range[1] - x_range[0]) / dx + 1)
x_grid = np.linspace(x_range[0], x_range[1], nx)

# Define the diffusivity field d(x)
def matter_density(x):
    in_interval1 = np.logical_and(x >= 2., x <= 9.)
    in_interval2 = np.logical_and(x >= 4., x <= 6.)
    return np.zeros_like(x) + 0.1 * in_interval1 + 0.9 * in_interval2

vox = matter_density(x_grid)
geom = VolumeDomain(vox, voxel_widths=[dx])

# Model parameters
rho = 0.6 + 0.1 * (2 * torch.rand(1).item() - 1)  # Adding small randomization to rho
D = 0.4 + 0.1 * (2 * torch.rand(1).item() - 1)  # Adding small randomization to D
x0 = [6. + 1. * (2 * torch.rand(1).item() - 1)]  # Initial tumor position
init_peak_height = 0.01
init_peak_width = 0.1
init_density_params = {'w': init_peak_width, 'h': init_peak_height, 'rmax': 3.*dx}

# Create the PDE model
fd_pde = TumorGrowthFD(
    geom, x0, D, rho,
    init_density_params=init_density_params,
    device=device
)

# Modify the model to include the new method
def solve_with_sensitivities(self, dt: float = 0.01, t1: float = 1.,
                             D: float = None, rho: float = None, x0: list = None):
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
    psi = torch.zeros(self.nx, device=self.device)  # psi = d u / d rho
    eta = self.init_density_deriv(
        self.x_mesh, torch.tensor(x0, device=self.device), **self.init_density_params, to_x0=True)  # eta = d u / d x0

    # finite difference forward
    for _ in range(nt-1):
        self._forward_with_sensitivities(u, phi, psi, eta, dt, D, rho)
        self._clip_state(u)
        u.mul_(self.domain_mask)

    return u, phi, psi, eta

# Add the new method to the instance
import types
fd_pde.solve_with_sensitivities = types.MethodType(solve_with_sensitivities, fd_pde)

# Time parameters
dt = 0.001
t1 = 1.0  # Shorter time for testing
t_span = (0., t1)

# Solve the PDE and compute sensitivities
u, phi, psi, eta = fd_pde.solve_with_sensitivities(dt=dt, t1=t1, D=D, rho=rho, x0=x0)
u = u.detach()
# Visualize u as the PDE solution
plt.figure(figsize=(10, 6))
plt.plot(x_grid, u.cpu().numpy(), label='PDE Solution u')
plt.xlabel('x')
plt.ylabel('u')
plt.title('PDE Solution u over Spatial Domain')
plt.legend()
plt.show()


phi = phi.detach()
psi = psi.detach()
eta = eta.detach()

# Compute numerical derivatives
delta = 1e-3
delta_D = delta
delta_rho = delta
delta_x0 = [delta] * fd_pde.dim

# Perturb D
u_D_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D + delta_D, rho=rho, x0=x0)
u_D_plus = u_D_plus.detach()
numerical_phi = (u_D_plus - u) / delta_D

# Perturb rho
u_rho_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho + delta_rho, x0=x0)
u_rho_plus = u_rho_plus.detach()
numerical_psi = (u_rho_plus - u) / delta_rho

# Perturb x0
x0_plus = [x0_i + delta_x0_i for x0_i, delta_x0_i in zip(x0, delta_x0)]
u_x0_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho, x0=x0_plus)
u_x0_plus = u_x0_plus.detach()
numerical_eta = (u_x0_plus - u) / delta_x0[0]  # Since x0 is scalar in 1D

# Compare phi
error_phi = phi - numerical_phi
max_error_phi = torch.max(torch.abs(error_phi)).item()
mean_error_phi = torch.mean(torch.abs(error_phi)).item()
print(f"Phi (∂u/∂D) - Max Error: {max_error_phi:.6e}, Mean Error: {mean_error_phi:.6e}")

# Compare psi
error_psi = psi - numerical_psi
max_error_psi = torch.max(torch.abs(error_psi)).item()
mean_error_psi = torch.mean(torch.abs(error_psi)).item()
print(f"Psi (∂u/∂rho) - Max Error: {max_error_psi:.6e}, Mean Error: {mean_error_psi:.6e}")

# Compare eta
error_eta = eta.squeeze(0) - numerical_eta
max_error_eta = torch.max(torch.abs(error_eta)).item()
mean_error_eta = torch.mean(torch.abs(error_eta)).item()
print(f"Eta (∂u/∂x0) - Max Error: {max_error_eta:.6e}, Mean Error: {mean_error_eta:.6e}")

# Plotting the results
x_np = x_grid

# Plot phi comparison
plt.figure(figsize=(10, 6))
plt.plot(x_np, phi.cpu().numpy(), label='Analytical φ (∂u/∂D)')
plt.plot(x_np, numerical_phi.cpu().numpy(), '--', label='Numerical φ (∂u/∂D)')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Comparison of Analytical and Numerical φ (∂u/∂D)')
plt.legend()
plt.grid(True)
plt.show()

# Plot psi comparison
plt.figure(figsize=(10, 6))
plt.plot(x_np, psi.cpu().numpy(), label='Analytical ψ (∂u/∂rho)')
plt.plot(x_np, numerical_psi.cpu().numpy(), '--', label='Numerical ψ (∂u/∂rho)')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Comparison of Analytical and Numerical ψ (∂u/∂rho)')
plt.legend()
plt.grid(True)
plt.show()

# Plot eta comparison
plt.figure(figsize=(10, 6))
plt.plot(x_np, eta.squeeze(0).cpu().numpy(), label='Analytical η (∂u/∂x0)')
plt.plot(x_np, numerical_eta.cpu().numpy(), '--', label='Numerical η (∂u/∂x0)')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Comparison of Analytical and Numerical η (∂u/∂x0)')
plt.legend()
plt.grid(True)
plt.show()
