import torch
import numpy as np
from hmpinn.PDEs import *

list_of_poisson_problems_torch = [cl() for cl in PDE_NAME_TO_CLASS.values()]

list_of_poisson_problems_np = [cl(backend=np) for cl in PDE_NAME_TO_CLASS.values()]

def test_outputs_torch(PDE):
    """
    Test the outputs of the PDEs to ensure they are correct.
    """
    x = torch.rand(100, 2, requires_grad=True)
    assert PDE.f(x).shape == torch.Size([100]), f"Output shape for f(x) is incorrect for {PDE.__class__.__name__} with values {PDE.f(x).shape}"
    assert PDE.u(x) is None or PDE.u(x).shape == torch.Size([100]), f"Output shape for u(x) is incorrect for {PDE.__class__.__name__} with values {PDE.u(x).shape}"
    assert PDE.diffusion_matrix(x).shape == torch.Size([100, 2, 2]), f"Output shape for diffusion_matrix(x) is incorrect for {PDE.__class__.__name__} with values {PDE.diffusion_matrix(x).shape}"
    assert PDE.grad_u(x) is None or PDE.grad_u(x).shape == torch.Size([100, 2]), f"Output shape for grad_u(x) is incorrect for {PDE.__class__.__name__} with values {PDE.grad_u(x).shape}"
    if isinstance(PDE.BC(x), torch.Tensor):
        assert PDE.BC(x).shape == torch.Size([100,]), f"Output shape for bc(x) is incorrect for {PDE.__class__.__name__} with values {PDE.BC(x).shape}"

def test_outputs_np(PDE):
    """
    Test the outputs of the PDEs to ensure they are correct.
    """
    x = np.random.rand(100, 2).astype(np.float32)
    assert PDE.f(x).shape == (100,), f"Output shape for f(x) is incorrect for {PDE.__class__.__name__} with values {PDE.f(x).shape}"
    assert PDE.u(x) is None or PDE.u(x).shape == (100,), f"Output shape for u(x) is incorrect for {PDE.__class__.__name__} with values {PDE.u(x).shape}"
    assert PDE.diffusion_matrix(x).shape == (100, 2, 2), f"Output shape for diffusion_matrix(x) is incorrect for {PDE.__class__.__name__} with values {PDE.diffusion_matrix(x).shape}"
    assert PDE.grad_u(x) is None or PDE.grad_u(x).shape == (100, 2), f"Output shape for grad_u(x) is incorrect for {PDE.__class__.__name__} with values {PDE.grad_u(x).shape}"
    if isinstance(PDE.BC(x), np.ndarray):
        assert PDE.BC(x).shape == (100,), f"Output shape for bc(x) is incorrect for {PDE.__class__.__name__} with values {PDE.BC(x).shape}"

def test_outputs(PDE):
    """
    Test the outputs of the PDEs to ensure they are correct.
    """
    if PDE.backend == torch:
        test_outputs_torch(PDE)
    else:
        test_outputs_np(PDE)

def test_attributes(PDE):
    """
    Test the attributes of the PDEs to ensure they are correct.
    """
    assert hasattr(PDE, 'f'), f"Missing attribute f for {PDE.__class__.__name__}"
    assert hasattr(PDE, 'u'), f"Missing attribute u for {PDE.__class__.__name__}"
    assert hasattr(PDE, 'diffusion_matrix'), f"Missing attribute diffusion_matrix for {PDE.__class__.__name__}"
    assert hasattr(PDE, 'grad_u'), f"Missing attribute grad_u for {PDE.__class__.__name__}"
    assert hasattr(PDE, 'BC'), f"Missing attribute BC for {PDE.__class__.__name__}"
    assert hasattr(PDE, 'is_in_divergence_form'), f"Missing attribute is_in_divergence_form for {PDE.__class__.__name__}"
    assert hasattr(PDE, "has_solution"), f"Missing attribute has_solution for {PDE.__class__.__name__}"
    if PDE.has_solution:
        if PDE.backend == torch:
            x = torch.rand(100, 2, requires_grad=True)
        else:
            x = np.random.rand(100, 2).astype(np.float32)
        assert PDE.u(x) is not None, f"Missing attribute u for {PDE.__class__.__name__}"
        assert PDE.grad_u(x) is not None, f"Missing attribute grad_u for {PDE.__class__.__name__}"
    assert hasattr(PDE, "compute_residual"), f"Missing attribute compute_residual for {PDE.__class__.__name__}"
    assert hasattr(PDE, "type_BC"), f"Missing attribute type_BC for {PDE.__class__.__name__}"
    assert hasattr(PDE, "differential_operator"), f"Missing attribute differential_operator for {PDE.__class__.__name__}"
    assert hasattr(PDE, "backend"), f"Missing attribute backend for {PDE.__class__.__name__}"
if __name__ == "__main__":
    for PDE in list_of_poisson_problems_torch + list_of_poisson_problems_np:
        print(f"Testing {PDE}")
        test_outputs(PDE)
        test_attributes(PDE)
    print("All Poisson equations are correct")
