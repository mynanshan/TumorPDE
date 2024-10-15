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

# Model parameters with small randomization
rho_true = 10.0 + 0.1 * (2 * torch.rand(1).item() - 1)  # Randomize rho slightly
D_true = 1.2 + 0.1 * (2 * torch.rand(1).item() - 1)    # Randomize D slightly
x0_true = [6. + 0.5 * (2 * torch.rand(1).item() - 1)]   # Randomize x0 slightly

init_peak_height = 0.01
init_peak_width = 0.1
init_density_params = {'w': init_peak_width, 'h': init_peak_height, 'rmax': 3.*dx}

# Create the PDE model
fd_pde = TumorGrowthFD(
    geom, x0_true, D_true, rho_true,
    init_density_params=init_density_params,
    device=device
)

# Modify the model to include the new method that computes loss and gradients
def solve_with_sensitivities(self, dt: float = 0.01, t1: float = 1.,
                             D: float = None, rho: float = None, x0: list = None, obs: torch.Tensor = None):
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

    # Compute the loss and gradients if observations are provided
    if obs is not None:
        resid = u - obs
        loss = 0.5 * torch.mean(resid ** 2)
        # Compute gradients using sensitivities
        grads = torch.tensor([
            torch.mean(resid * phi).item(),
            torch.mean(resid * psi).item(),
            torch.mean(resid * eta, dim=tuple(range(1, self.dim+1))).item()
        ])
    else:
        loss = None
        grads = None

    return u, phi, psi, eta, loss, grads

# Add the new method to the instance
import types
fd_pde.solve_with_sensitivities = types.MethodType(solve_with_sensitivities, fd_pde)

# Time parameters
dt = 0.001
t1 = 1.0  # Simulation time
t_span = (0., t1)

# Observed data: use the solution u as the observation
obs_u, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D_true, rho=rho_true, x0=x0_true)
obs_u = obs_u.detach()


# Visualize the observed data obs_u
plt.figure(figsize=(10, 6))
plt.plot(x_grid, obs_u.cpu().numpy(), label='Observed Data obs_u')
plt.xlabel('x')
plt.ylabel('obs_u')
plt.title('Observed Data obs_u over Spatial Domain')
plt.legend()
plt.grid(True)
plt.show()


# Model parameters with small randomization
rho = rho_true + 0.8 * (2 * torch.rand(1).item() - 1)  # Randomize rho slightly
D = D_true + 0.3 * (2 * torch.rand(1).item() - 1)    # Randomize D slightly
x0 = [x0_true[0] + 1.5 * (2 * torch.rand(1).item() - 1)]   # Randomize x0 slightly

# Solve the PDE and compute sensitivities, loss, and analytical gradients
u, phi, psi, eta, loss, grads = fd_pde.solve_with_sensitivities(dt=dt, t1=t1, D=D, rho=rho, x0=x0, obs=obs_u)
u = u.detach()
phi = phi.detach()
psi = psi.detach()
eta = eta.detach()
loss = loss.item()
grads = grads.detach()

print(f"Loss at current parameters: {loss:.6e}")
print(f"Analytical gradients:")
print(f"  ∂Loss/∂D:   {grads[0]:.6e}")
print(f"  ∂Loss/∂rho: {grads[1]:.6e}")
print(f"  ∂Loss/∂x0:  {grads[2]:.6e}")

# Compute numerical gradients
epsilon = 1e-5  # Small perturbation

# Perturb D
D_plus = D + epsilon
u_D_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D_plus, rho=rho, x0=x0)
loss_D_plus = 0.5 * torch.mean((u_D_plus - obs_u) ** 2).item()

D_minus = D - epsilon
u_D_minus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D_minus, rho=rho, x0=x0)
loss_D_minus = 0.5 * torch.mean((u_D_minus - obs_u) ** 2).item()

grad_D_numerical = (loss_D_plus - loss_D_minus) / (2 * epsilon)

# Perturb rho
rho_plus = rho + epsilon
u_rho_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho_plus, x0=x0)
loss_rho_plus = 0.5 * torch.mean((u_rho_plus - obs_u) ** 2).item()

rho_minus = rho - epsilon
u_rho_minus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho_minus, x0=x0)
loss_rho_minus = 0.5 * torch.mean((u_rho_minus - obs_u) ** 2).item()

grad_rho_numerical = (loss_rho_plus - loss_rho_minus) / (2 * epsilon)

# Perturb x0
x0_plus = [x0[0] + epsilon]
u_x0_plus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho, x0=x0_plus)
loss_x0_plus = 0.5 * torch.mean((u_x0_plus - obs_u) ** 2).item()

x0_minus = [x0[0] - epsilon]
u_x0_minus, _, _ = fd_pde.solve(dt=dt, t1=t1, D=D, rho=rho, x0=x0_minus)
loss_x0_minus = 0.5 * torch.mean((u_x0_minus - obs_u) ** 2).item()

grad_x0_numerical = (loss_x0_plus - loss_x0_minus) / (2 * epsilon)

# Print numerical gradients
print(f"Numerical gradients:")
print(f"  ∂Loss/∂D:   {grad_D_numerical:.6e}")
print(f"  ∂Loss/∂rho: {grad_rho_numerical:.6e}")
print(f"  ∂Loss/∂x0:  {grad_x0_numerical:.6e}")

# Compare analytical and numerical gradients
diff_D = abs(grads[0] - grad_D_numerical)
diff_rho = abs(grads[1] - grad_rho_numerical)
diff_x0 = abs(grads[2] - grad_x0_numerical)

print(f"D gradient difference:   {diff_D:.6e}")
print(f"rho gradient difference: {diff_rho:.6e}")
print(f"x0 gradient difference:  {diff_x0:.6e}")
