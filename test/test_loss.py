import torch
import numpy as np
from tumorpde.calc.loss import concentrate_density, deriv_concentrate_density, \
    mse_loss, deriv_mse_loss, logistic_loss, deriv_logistic_loss, concentrated_logistic_loss, deriv_concentrated_logistic_loss

def test_concentrate_density_derivative():

    print(">>> Start testing concentrate_density")

    # Test parameters
    u = torch.tensor([0.1, 0.3, 0.7, 0.9])
    thresh = 0.5
    capacity = 1.0
    order = 2.0
    
    # Calculate analytical derivative
    deriv_analytical = deriv_concentrate_density(u, thresh, capacity, order)
    
    # Calculate numerical derivative
    h = 1e-4
    
    # Derivative w.r.t. u
    u_plus = u + h
    u_minus = u - h
    deriv_numerical = (concentrate_density(u_plus, thresh, capacity, order) -
                       concentrate_density(u_minus, thresh, capacity, order)) / (2 * h)
    
    print("- Test derivative of u")
    print("Analytical derivative:", deriv_analytical)
    print("Numerical derivative:", deriv_numerical)
    
    # Check if analytical and numerical derivatives are close
    assert torch.allclose(deriv_analytical, deriv_numerical, atol=1e-3)
    

def test_mse_loss_derivative():

    print(">>> Start testing mse_loss")

    x = torch.tensor([0.2, 0.5, 0.8])
    y = torch.tensor([0., 0., 1.])
    
    # Calculate analytical derivative
    deriv_analytical = deriv_mse_loss(x, y)
    
    # Calculate numerical derivative
    h = 1e-4
    deriv_numerical = []
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus[i] += h
        x_minus = x.clone()
        x_minus[i] -= h
        deriv_numerical.append((mse_loss(x_plus, y) - mse_loss(x_minus, y)) / (2 * h))
    deriv_numerical = torch.cat([deriv.view(-1) for deriv in deriv_numerical], dim=0)
    
    print("- Test derivative of u")
    print("Analytical derivative:", deriv_analytical)
    print("Numerical derivative:", deriv_numerical)
    
    assert torch.allclose(deriv_analytical, deriv_numerical, atol=1e-3)

def test_logistic_loss_derivative():

    print(">>> Start testing logistic_loss")

    x = torch.tensor([0.2, 0.5, 0.8])
    y = torch.tensor([0., 0., 1.])
    
    # Calculate analytical derivative
    deriv_analytical = deriv_logistic_loss(x, y)
    
    # Calculate numerical derivative
    h = 1e-4
    deriv_numerical = []
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus[i] += h
        x_minus = x.clone()
        x_minus[i] -= h
        deriv_numerical.append((logistic_loss(x_plus, y) - logistic_loss(x_minus, y)) / (2 * h))
    deriv_numerical = torch.cat([deriv.view(-1) for deriv in deriv_numerical], dim=0)
    
    print("- Test derivative of u")
    print("Analytical derivative:", deriv_analytical)
    print("Numerical derivative:", deriv_numerical)
    
    assert torch.allclose(deriv_analytical, deriv_numerical, atol=1e-3)

def test_contrasted_logistic_loss():

    print(">>> Start testing contrasted_logistic_loss")

    x = torch.tensor([0.2, 0.5, 0.8])
    y = torch.tensor([0., 0., 1.])
    thresh = 0.5
    capacity = 1.
    order = 2.

    # Calculate analytical derivative
    deriv_analytical = deriv_concentrated_logistic_loss(x, y, thresh, capacity, order)
    
    # Calculate numerical gradient
    h = 1e-4
    deriv_numerical = []
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus[i] += h
        x_minus = x.clone()
        x_minus[i] -= h
        deriv_numerical.append((concentrated_logistic_loss(x_plus, y, thresh, capacity, order) - 
                               concentrated_logistic_loss(x_minus, y, thresh, capacity, order)) / (2 * h))
    deriv_numerical = torch.cat([deriv.view(-1) for deriv in deriv_numerical], dim=0)
    
    print("- Test derivative of u")
    print("Analytical derivative:", deriv_analytical)
    print("Numerical derivative:", deriv_numerical)
    
    assert torch.allclose(deriv_analytical, deriv_numerical, atol=1e-3)

if __name__ == "__main__":
    test_concentrate_density_derivative()
    test_mse_loss_derivative()
    test_logistic_loss_derivative()
    test_contrasted_logistic_loss()
    print("All tests passed!")
