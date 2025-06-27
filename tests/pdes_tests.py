"""
PDE Classes Test Suite.

This module provides comprehensive testing functionality for all PDE classes,
validating output shapes, required attributes, and backend compatibility.

The tests ensure that all PDE implementations conform to the expected interface
and produce outputs of correct dimensions for both PyTorch and NumPy backends.

Classes tested:
    - All PDE classes from PDE_NAME_TO_CLASS except harmonic maps
    - Both torch and numpy backend implementations
"""

import copy
from typing import List, Tuple

import torch
import numpy as np

from hmpinn.PDEs import PDE_NAME_TO_CLASS


# Remove harmonic maps from test classes (tested separately)
PDE_CLASSES = copy.deepcopy(PDE_NAME_TO_CLASS)
HARMONIC_MAPS = ["quarter_annulus_hm", "L_bend_hm", "sin_boundaries_hm", "poly_boundaries_hm"]
for hm in HARMONIC_MAPS:
    PDE_CLASSES.pop(hm, None)


def create_test_pdes() -> Tuple[List, List]:
    """
    Create lists of PDE instances for both supported backends.
    
    Returns:
        Tuple[List, List]: A tuple containing (torch_pdes, numpy_pdes) where
            each list contains instances of all PDE classes with the respective backend.
    """
    torch_pdes = [pde_class() for pde_class in PDE_CLASSES.values()]
    numpy_pdes = [pde_class(backend=np) for pde_class in PDE_CLASSES.values()]
    return torch_pdes, numpy_pdes


def test_torch_outputs(pde, x: torch.Tensor) -> None:
    """
    Test output shapes for PyTorch backend PDE implementations.
    
    Validates that all PDE methods return tensors of the correct shape
    when using PyTorch backend.
    
    Args:
        pde: PDE instance with torch backend.
        x: Input tensor of shape (batch_size, input_dim) with requires_grad=True.
        
    Raises:
        AssertionError: If any output shape is incorrect.
    """
    batch_size = x.shape[0]
    
    # Test source term output
    f_out = pde.f(x)
    assert f_out.shape == torch.Size([batch_size]), \
        f"f(x) shape mismatch for {pde.__class__.__name__}: expected {[batch_size]}, got {f_out.shape}"
    
    # Test analytical solution output (if available)
    u_out = pde.u(x)
    if u_out is not None:
        assert u_out.shape == torch.Size([batch_size]), \
            f"u(x) shape mismatch for {pde.__class__.__name__}: expected {[batch_size]}, got {u_out.shape}"
    
    # Test diffusion matrix output
    diff_out = pde.diffusion_matrix(x)
    assert diff_out.shape == torch.Size([batch_size, 2, 2]), \
        f"diffusion_matrix(x) shape mismatch for {pde.__class__.__name__}: expected {[batch_size, 2, 2]}, got {diff_out.shape}"
    
    # Test gradient output (if available)
    grad_out = pde.grad_u(x)
    if grad_out is not None:
        assert grad_out.shape == torch.Size([batch_size, 2]), \
            f"grad_u(x) shape mismatch for {pde.__class__.__name__}: expected {[batch_size, 2]}, got {grad_out.shape}"
    
    # Test boundary conditions output
    bc_out = pde.BC(x)
    if isinstance(bc_out, torch.Tensor):
        assert bc_out.shape == torch.Size([batch_size]), \
            f"BC(x) shape mismatch for {pde.__class__.__name__}: expected {[batch_size]}, got {bc_out.shape}"


def test_numpy_outputs(pde, x: np.ndarray) -> None:
    """
    Test output shapes for NumPy backend PDE implementations.
    
    Validates that all PDE methods return arrays of the correct shape
    when using NumPy backend.
    
    Args:
        pde: PDE instance with numpy backend.
        x: Input array of shape (batch_size, input_dim).
        
    Raises:
        AssertionError: If any output shape is incorrect.
    """
    batch_size = x.shape[0]
    
    # Test source term output
    f_out = pde.f(x)
    assert f_out.shape == (batch_size,), \
        f"f(x) shape mismatch for {pde.__class__.__name__}: expected ({batch_size},), got {f_out.shape}"
    
    # Test analytical solution output (if available)
    u_out = pde.u(x)
    if u_out is not None:
        assert u_out.shape == (batch_size,), \
            f"u(x) shape mismatch for {pde.__class__.__name__}: expected ({batch_size},), got {u_out.shape}"
    
    # Test diffusion matrix output
    diff_out = pde.diffusion_matrix(x)
    assert diff_out.shape == (batch_size, 2, 2), \
        f"diffusion_matrix(x) shape mismatch for {pde.__class__.__name__}: expected ({batch_size}, 2, 2), got {diff_out.shape}"
    
    # Test gradient output (if available)
    grad_out = pde.grad_u(x)
    if grad_out is not None:
        assert grad_out.shape == (batch_size, 2), \
            f"grad_u(x) shape mismatch for {pde.__class__.__name__}: expected ({batch_size}, 2), got {grad_out.shape}"
    
    # Test boundary conditions output
    bc_out = pde.BC(x)
    if isinstance(bc_out, np.ndarray):
        assert bc_out.shape == (batch_size,), \
            f"BC(x) shape mismatch for {pde.__class__.__name__}: expected ({batch_size},), got {bc_out.shape}"


def test_outputs(pde) -> None:
    """
    Test PDE outputs based on the configured backend.
    
    Automatically selects the appropriate test function based on the PDE's
    backend configuration and generates suitable test data.
    
    Args:
        pde: PDE instance to test.
    """
    if pde.backend == torch:
        x = torch.rand(100, 2, requires_grad=True)
        test_torch_outputs(pde, x)
    else:
        x = np.random.rand(100, 2).astype(np.float32)
        test_numpy_outputs(pde, x)


def test_attributes(pde) -> None:
    """
    Test that PDE instances have all required attributes.
    
    Validates the presence of all mandatory attributes and methods
    that define the PDE interface, including solution-specific attributes
    when applicable.
    
    Args:
        pde: PDE instance to test.
        
    Raises:
        AssertionError: If any required attribute is missing or invalid.
    """
    required_attrs = [
        'f', 'u', 'diffusion_matrix', 'grad_u', 'BC', 'is_in_divergence_form',
        'has_solution', 'compute_residual', 'type_BC', 'differential_operator', 'backend'
    ]
    
    pde_name = pde.__class__.__name__
    for attr in required_attrs:
        assert hasattr(pde, attr), f"Missing attribute '{attr}' for {pde_name}"
    
    # Test solution-specific attributes for PDEs with analytical solutions
    if pde.has_solution:
        if pde.backend == torch:
            x = torch.rand(100, 2, requires_grad=True)
        else:
            x = np.random.rand(100, 2).astype(np.float32)
        
        assert pde.u(x) is not None, f"Missing solution u(x) for {pde_name}"
        assert pde.grad_u(x) is not None, f"Missing gradient grad_u(x) for {pde_name}"


def run_pde_tests() -> None:
    """
    Run comprehensive tests for all PDE classes.
    
    Tests all PDE implementations with both PyTorch and NumPy backends,
    validating output shapes and required attributes.
    
    Raises:
        AssertionError: If any tests fail, with detailed failure information.
    """
    torch_pdes, numpy_pdes = create_test_pdes()
    all_pdes = torch_pdes + numpy_pdes
    
    failed_tests = []
    for pde in all_pdes:
        try:
            print(f"Testing {pde.__class__.__name__} with {pde.backend.__name__} backend")
            test_outputs(pde)
            test_attributes(pde)
        except Exception as e:
            failed_tests.append(f"{pde.__class__.__name__} ({pde.backend.__name__}): {e}")
    
    if failed_tests:
        for failure in failed_tests:
            print(f"FAILED: {failure}")
        raise AssertionError(f"{len(failed_tests)} tests failed")
    else:
        print("All PDE tests passed!")


if __name__ == "__main__":
    run_pde_tests()
