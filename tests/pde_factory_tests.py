"""
PDE Factory Test Suite.

This module provides comprehensive testing functionality for the PDE factory system,
which dynamically constructs PDE classes from user-provided functions.

The tests validate that the factory can create valid PDE classes with various
configurations and that the resulting instances work correctly with both
PyTorch and NumPy backends.

Functions tested:
    - construct_PDE_class with full specification
    - construct_PDE_class with minimal specification
    - Backend compatibility for all configurations
"""

from typing import Callable, Optional

import torch
import numpy as np

from hmpinn.PDEs.PDE_factory import construct_PDE_class
from hmpinn.PDEs.utils import ensure_backend, check_backend, stack
from tests.pdes_tests import test_outputs, test_attributes


def create_source_term() -> Callable:
    """
    Create a piecewise source term function for testing.
    
    Returns:
        Callable: Source term function f(x, backend) that supports both
            PyTorch and NumPy backends with piecewise definition.
    """
    def f(x, backend=torch):
        """
        Piecewise source term function.
        
        Args:
            x: Input coordinates of shape (batch_size, 2).
            backend: Backend library (torch or numpy).
            
        Returns:
            Source term values at input coordinates.
        """
        if check_backend(backend):
            x = ensure_backend(x, backend)
        
        x_val, y_val = x[:, 0], x[:, 1]
        
        result = backend.where(
            x_val <= 0.5,
            15 * x_val**4 + 3 * x_val**2 * y_val + 33*x_val + y_val + 3*x_val**2 - 6 * y_val**2 + 2 * x_val * y_val - 14,
            -(12 * x_val**3 + 6 * x_val * y_val - 12 * x_val - 2 * y_val**2 + 3 * x_val**3 - 8 * y_val**3 + 3 * y_val**2 * x_val - 6)
        )
        return result
    return f


def create_analytical_solution() -> Callable:
    """
    Create an analytical solution function for testing.
    
    Returns:
        Callable: Solution function u(x, backend) that supports both
            PyTorch and NumPy backends.
    """
    def u(x, backend=torch):
        """
        Cubic polynomial analytical solution.
        
        Args:
            x: Input coordinates of shape (batch_size, 2).
            backend: Backend library (torch or numpy).
            
        Returns:
            Solution values at input coordinates.
        """
        if check_backend(backend):
            x = ensure_backend(x, backend)
        return x[:, 0]**3 - x[:, 1]**2 + x[:, 0] * x[:, 1]
    return u


def create_boundary_condition() -> Callable:
    """
    Create a boundary condition function for testing.
    
    Returns:
        Callable: Boundary condition function that matches the analytical solution.
    """
    def boundary_condition(x, backend=torch):
        """
        Dirichlet boundary condition matching analytical solution.
        
        Args:
            x: Input coordinates of shape (batch_size, 2).
            backend: Backend library (torch or numpy).
            
        Returns:
            Boundary values at input coordinates.
        """
        if check_backend(backend):
            x = ensure_backend(x, backend)
        return x[:, 0]**3 - x[:, 1]**2 + x[:, 0] * x[:, 1]
    return boundary_condition


def create_diffusion_matrix() -> Callable:
    """
    Create a piecewise diffusion matrix function for testing.
    
    Returns:
        Callable: Diffusion matrix function k(x, backend) that returns
            a tensor of shape (batch_size, 2, 2).
    """
    def diffusion_matrix(x, backend=torch):
        """
        Piecewise anisotropic diffusion matrix.
        
        Args:
            x: Input coordinates of shape (batch_size, 2).
            backend: Backend library (torch or numpy).
            
        Returns:
            Diffusion matrix of shape (batch_size, 2, 2).
        """
        if check_backend(backend):
            x = ensure_backend(x, backend)
        
        batch_size = x.shape[0]
        
        def k1(x):
            """First diffusion matrix for x <= 0.5."""
            diffusion = backend.empty((batch_size, 2, 2))
            diffusion[:, 0, 0] = x[:, 0]**3 + 5
            diffusion[:, 0, 1] = x[:, 0] + x[:, 1]
            diffusion[:, 1, 0] = x[:, 1] + x[:, 0]
            diffusion[:, 1, 1] = x[:, 1]**2 + 7
            return diffusion
        
        def k2(x):
            """Second diffusion matrix for x > 0.5."""
            diffusion = backend.empty((batch_size, 2, 2))
            diffusion[:, 0, 0] = -x[:, 0]**2 + 2
            diffusion[:, 0, 1] = -x[:, 0] * x[:, 1]
            diffusion[:, 1, 0] = -x[:, 0] * x[:, 1]
            diffusion[:, 1, 1] = -x[:, 1]**3 - 3
            return diffusion
        
        condition = (x[:, 0] <= 0.5)[:, None, None]
        return backend.where(condition, k1(x), k2(x))
    return diffusion_matrix


def create_gradient_function() -> Callable:
    """
    Create a gradient function corresponding to the analytical solution.
    
    Returns:
        Callable: Gradient function grad_u(x, backend) that returns
            the gradient of the analytical solution.
    """
    def grad_u(x, backend=torch):
        """
        Gradient of the analytical solution.
        
        Args:
            x: Input coordinates of shape (batch_size, 2).
            backend: Backend library (torch or numpy).
            
        Returns:
            Gradient vector of shape (batch_size, 2).
        """
        if check_backend(backend):
            x = ensure_backend(x, backend)
        
        grad_x = 3 * x[:, 0]**2 + x[:, 1]
        grad_y = -2 * x[:, 1] + x[:, 0]
        return stack([grad_x, grad_y], dim=1, backend=backend)
    return grad_u


def test_pde_configuration(is_divergence_form: bool, 
                          f: Callable,
                          diffusion_matrix: Optional[Callable] = None,
                          boundary_condition: Optional[Callable] = None,
                          u: Optional[Callable] = None,
                          grad_u: Optional[Callable] = None,
                          backend_module = torch) -> None:
    """
    Test a specific PDE configuration created by the factory.
    
    Creates a PDE class using the factory with the given configuration
    and validates that it passes all standard PDE tests.
    
    Args:
        is_divergence_form: Whether the PDE is in divergence form.
        f: Source term function.
        diffusion_matrix: Optional diffusion matrix function.
        boundary_condition: Optional boundary condition function.
        u: Optional analytical solution function.
        grad_u: Optional gradient function.
        backend_module: Backend to test with (torch or numpy).
        
    Raises:
        AssertionError: If the created PDE fails any tests.
    """
    pde_class = construct_PDE_class(
        is_divergence_form, f, diffusion_matrix, boundary_condition, u, grad_u
    )
    
    pde_instance = pde_class(backend=backend_module)
    test_outputs(pde_instance)
    test_attributes(pde_instance)


def run_factory_tests() -> None:
    """
    Run comprehensive PDE factory tests.
    
    Tests the factory with various configurations including full specification
    with all components and minimal specification with only source term.
    Each configuration is tested with both PyTorch and NumPy backends.
    
    Raises:
        AssertionError: If any factory tests fail.
    """
    # Create all required functions
    f = create_source_term()
    u = create_analytical_solution()
    boundary_condition = create_boundary_condition()
    diffusion_matrix = create_diffusion_matrix()
    grad_u = create_gradient_function()
    
    test_cases = [
        # Full PDE with all components - divergence form
        {"is_divergence_form": True, "f": f, "diffusion_matrix": diffusion_matrix, 
         "boundary_condition": boundary_condition, "u": u, "grad_u": grad_u},
        # Full PDE with all components - non-divergence form
        {"is_divergence_form": False, "f": f, "diffusion_matrix": diffusion_matrix, 
         "boundary_condition": boundary_condition, "u": u, "grad_u": grad_u},
        
        # Minimal PDE (source term only) - divergence form
        {"is_divergence_form": True, "f": f},
        # Minimal PDE (source term only) - non-divergence form
        {"is_divergence_form": False, "f": f},
    ]
    
    backends = [torch, np]
    
    failed_tests = []
    for i, test_case in enumerate(test_cases):
        for backend in backends:
            try:
                backend_name = backend.__name__
                print(f"Testing configuration {i+1} with {backend_name} backend...")
                test_pde_configuration(backend_module=backend, **test_case)
            except Exception as e:
                failed_tests.append(f"Config {i+1} ({backend_name}): {e}")
    
    if failed_tests:
        for failure in failed_tests:
            print(f"FAILED: {failure}")
        raise AssertionError(f"{len(failed_tests)} PDE factory tests failed")
    else:
        print("All PDE factory tests passed!")


if __name__ == "__main__":
    run_factory_tests()