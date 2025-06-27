"""
Differential Operators Test Suite.

This module provides comprehensive testing functionality for all differential operators
used in Physics-Informed Neural Networks (PINNs), including partial derivatives,
gradients, Hessian matrices, Laplacians, divergence, and Jacobians.

The tests validate mathematical correctness by comparing computed derivatives
against analytical solutions for known test functions, ensuring that the automatic
differentiation implementations produce accurate results.

Classes:
    TestFunctions: Collection of test functions with known analytical derivatives
    VectorTestFunction: Test vector function for Jacobian testing
    SimpleVectorFunction: Simple vector function for divergence testing

Functions:
    Various test functions for each differential operator
    run_operator_tests: Comprehensive test runner
"""

import torch
import torch.nn as nn

from hmpinn.differential_operators import (
    Laplacian, Divergence, Gradient, Jacobian, PartialDerivative, Hessian
)


class TestFunctions:
    """
    Collection of test functions with known analytical derivatives.
    
    This class provides static methods for various mathematical functions
    along with their analytical derivatives, used to validate the correctness
    of automatic differentiation implementations.
    """
    
    @staticmethod
    def quadratic_function(x: torch.Tensor) -> torch.Tensor:
        """
        Quadratic test function: g(x₁,x₂) = x₁² × x₂².
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Function values of shape (batch_size,).
        """
        return x[:, 0]**2 * x[:, 1]**2
    
    @staticmethod
    def quadratic_gradient(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical gradient of quadratic function: ∇g = (2x₁x₂², 2x₁²x₂).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Gradient vector of shape (batch_size, 2).
        """
        return torch.stack([2*x[:, 0]*x[:, 1]**2, 2*x[:, 0]**2*x[:, 1]], dim=1)
    
    @staticmethod
    def quadratic_partial_x(x: torch.Tensor) -> torch.Tensor:
        """
        Partial derivative with respect to x₁: ∂g/∂x₁ = 2x₁x₂².
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Partial derivative values of shape (batch_size,).
        """
        return 2*x[:, 0]*x[:, 1]**2
    
    @staticmethod
    def quadratic_partial_y(x: torch.Tensor) -> torch.Tensor:
        """
        Partial derivative with respect to x₂: ∂g/∂x₂ = 2x₁²x₂.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Partial derivative values of shape (batch_size,).
        """
        return 2*x[:, 0]**2*x[:, 1]
    
    @staticmethod
    def quadratic_hessian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Hessian matrix of quadratic function.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Hessian matrix of shape (batch_size, 2, 2).
        """
        batch_size = x.shape[0]
        hessian = torch.zeros(batch_size, 2, 2)
        hessian[:, 0, 0] = 2 * x[:, 1]**2  # ∂²g/∂x₁²
        hessian[:, 0, 1] = 4 * x[:, 0] * x[:, 1]  # ∂²g/∂x₁∂x₂
        hessian[:, 1, 0] = 4 * x[:, 0] * x[:, 1]  # ∂²g/∂x₂∂x₁
        hessian[:, 1, 1] = 2 * x[:, 0]**2  # ∂²g/∂x₂²
        return hessian
    
    @staticmethod
    def quadratic_laplacian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Laplacian of quadratic function: Δg = 2x₂² + 2x₁².
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Laplacian values of shape (batch_size,).
        """
        return 2*x[:, 1]**2 + 2*x[:, 0]**2
    
    @staticmethod
    def cubic_function(x: torch.Tensor) -> torch.Tensor:
        """
        Cubic test function: f(x₁,x₂) = x₁³ + x₂³ - 3x₁x₂.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Function values of shape (batch_size,).
        """
        return x[:, 0]**3 + x[:, 1]**3 - 3*x[:, 0]*x[:, 1]
    
    @staticmethod
    def cubic_gradient(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical gradient of cubic function: ∇f = (3x₁² - 3x₂, 3x₂² - 3x₁).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Gradient vector of shape (batch_size, 2).
        """
        return torch.stack([3*x[:, 0]**2 - 3*x[:, 1], 3*x[:, 1]**2 - 3*x[:, 0]], dim=1)
    
    @staticmethod
    def cubic_partial_x(x: torch.Tensor) -> torch.Tensor:
        """
        Partial derivative of cubic function with respect to x₁: ∂f/∂x₁ = 3x₁² - 3x₂.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Partial derivative values of shape (batch_size,).
        """
        return 3*x[:, 0]**2 - 3*x[:, 1]
    
    @staticmethod 
    def cubic_partial_y(x: torch.Tensor) -> torch.Tensor:
        """
        Partial derivative of cubic function with respect to x₂: ∂f/∂x₂ = 3x₂² - 3x₁.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Partial derivative values of shape (batch_size,).
        """
        return 3*x[:, 1]**2 - 3*x[:, 0]
    
    @staticmethod
    def cubic_hessian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Hessian matrix of cubic function.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Hessian matrix of shape (batch_size, 2, 2).
        """
        batch_size = x.shape[0]
        hessian = torch.zeros(batch_size, 2, 2)
        hessian[:, 0, 0] = 6 * x[:, 0]  # ∂²f/∂x₁²
        hessian[:, 0, 1] = -3  # ∂²f/∂x₁∂x₂
        hessian[:, 1, 0] = -3  # ∂²f/∂x₂∂x₁
        hessian[:, 1, 1] = 6 * x[:, 1]  # ∂²f/∂x₂²
        return hessian
    
    @staticmethod
    def cubic_laplacian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Laplacian of cubic function: Δf = 6x₁ + 6x₂.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Laplacian values of shape (batch_size,).
        """
        return 6*x[:, 0] + 6*x[:, 1]
    
    @staticmethod
    def log_function(x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic harmonic function: h(x₁,x₂) = 0.5 × ln(x₁² + x₂²).
        
        This is a harmonic function (Laplacian = 0) useful for testing.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Function values of shape (batch_size,).
        """
        return 0.5 * torch.log(x[:, 0]**2 + x[:, 1]**2)
    
    @staticmethod
    def log_gradient(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical gradient of logarithmic function: ∇h = (x₁/(x₁²+x₂²), x₂/(x₁²+x₂²)).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Gradient vector of shape (batch_size, 2).
        """
        r_squared = x[:, 0]**2 + x[:, 1]**2
        return torch.stack([x[:, 0]/r_squared, x[:, 1]/r_squared], dim=1)
    
    @staticmethod
    def log_laplacian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Laplacian of logarithmic function: Δh = 0 (harmonic function).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Laplacian values of shape (batch_size,) (all zeros).
        """
        return torch.zeros_like(x[:, 0])
    
    @staticmethod
    def create_diffusion_matrix(x: torch.Tensor, model: nn.Module = None) -> torch.Tensor:
        """
        Create a test diffusion matrix for anisotropic diffusion testing.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            model: Neural network model (unused but required for interface).
            
        Returns:
            torch.Tensor: Diffusion matrix of shape (batch_size, 2, 2).
        """
        batch_size = x.shape[0]
        diffusion = torch.zeros(batch_size, 2, 2)
        diffusion[:, 0, 0] = x[:, 0]**2
        diffusion[:, 0, 1] = x[:, 0]
        diffusion[:, 1, 0] = x[:, 1]**2
        diffusion[:, 1, 1] = x[:, 1]
        return diffusion.requires_grad_()
    
    @staticmethod
    def quadratic_laplacian_with_diffusion(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Laplacian of quadratic function with anisotropic diffusion.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Diffusive Laplacian values of shape (batch_size,).
        """
        return 6 * x[:, 0]**2 * x[:, 1]**2 + 10 * x[:, 1] * x[:, 0]**2 + 8 * x[:, 0] * x[:, 1]**3


class VectorTestFunction(nn.Module):
    """
    Test vector function for Jacobian testing.
    
    Implements a complex vector-valued function with known analytical Jacobian
    for validating the Jacobian operator implementation.
    """
    
    def __init__(self):
        """Initialize the vector test function."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vector function: F(x₁,x₂) = (x₁²x₂⁵, x₂⁷x₁⁴).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Vector output of shape (batch_size, 2).
        """
        return torch.stack([
            x[:, 0]**2 * x[:, 1]**5, 
            x[:, 1]**7 * x[:, 0]**4
        ], dim=1)
    
    @staticmethod
    def true_jacobian(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical Jacobian of the vector function.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Jacobian matrix of shape (batch_size, 2, 2).
        """
        return torch.stack([
            torch.stack([2 * x[:, 0] * x[:, 1]**5, 5 * x[:, 0]**2 * x[:, 1]**4], dim=1),
            torch.stack([4 * x[:, 1]**7 * x[:, 0]**3, 7 * x[:, 1]**6 * x[:, 0]**4], dim=1)
        ], dim=1)


class SimpleVectorFunction(nn.Module):
    """
    Simple vector function for divergence testing.
    
    Implements a basic vector function with easily computed divergence
    for validating the divergence operator.
    """
    
    def __init__(self):
        """Initialize the simple vector function."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple vector function: G(x₁,x₂) = (x₁², x₂²).
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Vector output of shape (batch_size, 2).
        """
        return torch.stack([x[:, 0]**2, x[:, 1]**2], dim=1)
    
    @staticmethod
    def true_divergence(x: torch.Tensor) -> torch.Tensor:
        """
        Analytical divergence of the simple vector function: ∇·G = 2x₁ + 2x₂.
        
        Args:
            x: Input tensor of shape (batch_size, 2).
            
        Returns:
            torch.Tensor: Divergence values of shape (batch_size,).
        """
        return 2*x[:, 0] + 2*x[:, 1]


def test_partial_derivative() -> None:
    """
    Test PartialDerivative operator with multiple test functions.
    
    Validates that partial derivatives are computed correctly by comparing
    against analytical solutions for both quadratic and cubic functions.
    
    Raises:
        AssertionError: If any partial derivative computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    partial_op = PartialDerivative()
    
    # Test partial derivatives of quadratic function
    computed_x = partial_op(TestFunctions.quadratic_function, x, 0)
    expected_x = TestFunctions.quadratic_partial_x(x)
    assert torch.allclose(computed_x, expected_x), \
        "Partial derivative w.r.t. x failed for quadratic function"
    
    computed_y = partial_op(TestFunctions.quadratic_function, x, 1)
    expected_y = TestFunctions.quadratic_partial_y(x)
    assert torch.allclose(computed_y, expected_y), \
        "Partial derivative w.r.t. y failed for quadratic function"
    
    # Test partial derivatives of cubic function
    computed_x_cubic = partial_op(TestFunctions.cubic_function, x, 0)
    expected_x_cubic = TestFunctions.cubic_partial_x(x)
    assert torch.allclose(computed_x_cubic, expected_x_cubic), \
        "Partial derivative w.r.t. x failed for cubic function"
    
    computed_y_cubic = partial_op(TestFunctions.cubic_function, x, 1)
    expected_y_cubic = TestFunctions.cubic_partial_y(x)
    assert torch.allclose(computed_y_cubic, expected_y_cubic), \
        "Partial derivative w.r.t. y failed for cubic function"


def test_gradient() -> None:
    """
    Test Gradient operator with multiple test functions.
    
    Validates that gradients are computed correctly by comparing against
    analytical solutions for quadratic, cubic, and logarithmic functions.
    
    Raises:
        AssertionError: If any gradient computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    gradient_op = Gradient()
    
    # Test gradient of quadratic function
    computed = gradient_op(TestFunctions.quadratic_function, x)
    expected = TestFunctions.quadratic_gradient(x)
    assert torch.allclose(computed, expected), \
        "Gradient failed for quadratic function"
    
    # Test gradient of cubic function
    computed_cubic = gradient_op(TestFunctions.cubic_function, x)
    expected_cubic = TestFunctions.cubic_gradient(x)
    assert torch.allclose(computed_cubic, expected_cubic), \
        "Gradient failed for cubic function"
    
    # Test gradient of logarithmic function
    computed_log = gradient_op(TestFunctions.log_function, x)
    expected_log = TestFunctions.log_gradient(x)
    assert torch.allclose(computed_log, expected_log, atol=1e-5), \
        "Gradient failed for log function"


def test_hessian() -> None:
    """
    Test Hessian operator with validation of symmetry and trace properties.
    
    Validates Hessian matrix computation, symmetry property for smooth functions,
    and the relationship between Hessian trace and Laplacian.
    
    Raises:
        AssertionError: If any Hessian computation or property is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    hessian_op = Hessian()
    
    # Test Hessian of quadratic function
    computed = hessian_op(TestFunctions.quadratic_function, x)
    expected = TestFunctions.quadratic_hessian(x)
    assert torch.allclose(computed, expected, atol=1e-5), \
        "Hessian failed for quadratic function"
    
    # Test Hessian of cubic function
    computed_cubic = hessian_op(TestFunctions.cubic_function, x)
    expected_cubic = TestFunctions.cubic_hessian(x)
    assert torch.allclose(computed_cubic, expected_cubic, atol=1e-5), \
        "Hessian failed for cubic function"
    
    # Test Hessian symmetry property
    assert torch.allclose(computed[:, 0, 1], computed[:, 1, 0], atol=1e-5), \
        "Hessian is not symmetric for quadratic function"
    assert torch.allclose(computed_cubic[:, 0, 1], computed_cubic[:, 1, 0], atol=1e-5), \
        "Hessian is not symmetric for cubic function"
    
    # Test relationship: trace(Hessian) = Laplacian
    laplacian_op = Laplacian()
    computed_laplacian = laplacian_op(TestFunctions.quadratic_function, x)
    hessian_trace = torch.diagonal(computed, dim1=1, dim2=2).sum(dim=1)
    assert torch.allclose(computed_laplacian, hessian_trace, atol=1e-5), \
        "Trace of Hessian ≠ Laplacian for quadratic function"


def test_divergence() -> None:
    """
    Test Divergence operator and its relationship with Laplacian.
    
    Validates divergence computation and the fundamental relationship
    that divergence of gradient equals Laplacian.
    
    Raises:
        AssertionError: If divergence computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    divergence_op = Divergence()
    
    # Test divergence of simple vector function
    vector_func = SimpleVectorFunction()
    computed = divergence_op(vector_func, x)
    expected = SimpleVectorFunction.true_divergence(x)
    assert torch.allclose(computed, expected), \
        "Divergence failed for simple vector function"
    
    # Test fundamental relationship: div(grad(f)) = Laplacian(f)
    gradient_op = Gradient()
    laplacian_op = Laplacian()
    
    grad_result = gradient_op(TestFunctions.quadratic_function, x)
    div_grad = divergence_op(grad_result, x)
    laplacian_result = laplacian_op(TestFunctions.quadratic_function, x)
    assert torch.allclose(div_grad, laplacian_result), \
        "Divergence of gradient ≠ Laplacian"


def test_laplacian_basic() -> None:
    """
    Test basic Laplacian operator functionality.
    
    Validates Laplacian computation for quadratic, cubic, and harmonic functions,
    including the important property that harmonic functions have zero Laplacian.
    
    Raises:
        AssertionError: If any Laplacian computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    laplacian_op = Laplacian()
    
    # Test Laplacian of quadratic function
    computed = laplacian_op(TestFunctions.quadratic_function, x)
    expected = TestFunctions.quadratic_laplacian(x)
    assert torch.allclose(computed, expected, atol=1e-4), \
        "Laplacian failed for quadratic function g(x₁,x₂) = x₁²x₂²"
    
    # Test Laplacian of cubic function
    computed_cubic = laplacian_op(TestFunctions.cubic_function, x)
    expected_cubic = TestFunctions.cubic_laplacian(x)
    assert torch.allclose(computed_cubic, expected_cubic, atol=1e-4), \
        "Laplacian failed for cubic function"
    
    # Test Laplacian of harmonic function
    computed_log = laplacian_op(TestFunctions.log_function, x)
    expected_log = TestFunctions.log_laplacian(x)
    assert torch.allclose(computed_log, expected_log, atol=1e-4), \
        "Laplacian failed for harmonic function h(x₁,x₂) = 0.5×ln(x₁²+x₂²)"


def test_laplacian_with_diffusion() -> None:
    """
    Test Laplacian operator with anisotropic diffusion matrix.
    
    Validates the computation of the generalized Laplacian with diffusion
    for anisotropic diffusion problems.
    
    Raises:
        AssertionError: If diffusive Laplacian computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    laplacian_op = Laplacian()
    
    # Test with function values
    y = TestFunctions.quadratic_function(x)
    computed = laplacian_op(y, x, TestFunctions.create_diffusion_matrix)
    expected = TestFunctions.quadratic_laplacian_with_diffusion(x)
    assert torch.allclose(computed, expected), \
        "Laplacian with diffusion failed for quadratic function"
    
    # Test with function directly
    computed = laplacian_op(TestFunctions.quadratic_function, x, TestFunctions.create_diffusion_matrix)
    assert torch.allclose(computed, expected), \
        "Laplacian with diffusion failed when passing function directly"


def test_laplacian_divergence_equivalence() -> None:
    """
    Test fundamental equivalence: Laplacian = divergence(gradient).
    
    Validates this crucial mathematical relationship for multiple test functions.
    
    Raises:
        AssertionError: If the equivalence does not hold.
    """
    x = torch.rand(100, 2, requires_grad=True)
    laplacian_op = Laplacian()
    divergence_op = Divergence()
    gradient_op = Gradient()
    
    # Test equivalence for quadratic function
    lap_result = laplacian_op(TestFunctions.quadratic_function, x)
    div_grad_result = divergence_op(gradient_op(TestFunctions.quadratic_function, x), x)
    assert torch.allclose(lap_result, div_grad_result), \
        "Laplacian ≠ divergence of gradient for quadratic function"
    
    # Test equivalence for cubic function
    lap_result_cubic = laplacian_op(TestFunctions.cubic_function, x)
    div_grad_result_cubic = divergence_op(gradient_op(TestFunctions.cubic_function, x), x)
    assert torch.allclose(lap_result_cubic, div_grad_result_cubic), \
        "Laplacian ≠ divergence of gradient for cubic function"
    
    # Test equivalence for harmonic function
    lap_result_log = laplacian_op(TestFunctions.log_function, x)
    div_grad_result_log = divergence_op(gradient_op(TestFunctions.log_function, x), x)
    assert torch.allclose(lap_result_log, div_grad_result_log, atol=1e-4), \
        "Laplacian ≠ divergence of gradient for harmonic function"


def test_jacobian() -> None:
    """
    Test Jacobian operator for vector-valued functions.
    
    Validates Jacobian matrix computation against analytical solutions
    for both complex and simple vector functions.
    
    Raises:
        AssertionError: If Jacobian computation is incorrect.
    """
    x = torch.rand(100, 2, requires_grad=True)
    jacobian_op = Jacobian()
    
    # Test Jacobian of complex vector function
    vector_func = VectorTestFunction()
    computed = jacobian_op(vector_func, x)
    expected = VectorTestFunction.true_jacobian(x)
    assert torch.allclose(computed, expected), \
        "Jacobian computation failed for vector test function"
    
    # Test with simple vector function
    simple_func = SimpleVectorFunction()
    computed_simple = jacobian_op(simple_func, x)
    
    # Expected Jacobian for G(x₁,x₂) = (x₁², x₂²) is [[2x₁, 0], [0, 2x₂]]
    expected_simple = torch.zeros(x.shape[0], 2, 2)
    expected_simple[:, 0, 0] = 2 * x[:, 0]
    expected_simple[:, 1, 1] = 2 * x[:, 1]
    
    assert torch.allclose(computed_simple, expected_simple), \
        "Jacobian computation failed for simple vector function"


def test_operator_consistency() -> None:
    """
    Test mathematical consistency between different operators.
    
    Validates relationships like gradient components matching partial derivatives,
    Hessian diagonal elements, and trace-Laplacian equivalence.
    
    Raises:
        AssertionError: If any consistency check fails.
    """
    x = torch.rand(50, 2, requires_grad=True)
    
    partial_op = PartialDerivative()
    gradient_op = Gradient()
    hessian_op = Hessian()
    laplacian_op = Laplacian()
    
    # Test that gradient components match partial derivatives
    grad = gradient_op(TestFunctions.quadratic_function, x)
    partial_x = partial_op(TestFunctions.quadratic_function, x, 0)
    partial_y = partial_op(TestFunctions.quadratic_function, x, 1)
    
    assert torch.allclose(grad[:, 0], partial_x), \
        "Gradient x-component ≠ partial derivative w.r.t. x"
    assert torch.allclose(grad[:, 1], partial_y), \
        "Gradient y-component ≠ partial derivative w.r.t. y"
    
    # Test that Hessian diagonal equals second partial derivatives
    hess = hessian_op(TestFunctions.quadratic_function, x)
    expected_hess_00 = 2 * x[:, 1]**2  # ∂²g/∂x₁²
    expected_hess_11 = 2 * x[:, 0]**2  # ∂²g/∂x₂²
    
    assert torch.allclose(hess[:, 0, 0], expected_hess_00), \
        "Hessian diagonal (0,0) incorrect"
    assert torch.allclose(hess[:, 1, 1], expected_hess_11), \
        "Hessian diagonal (1,1) incorrect"
    
    # Test that trace of Hessian equals Laplacian
    laplacian = laplacian_op(TestFunctions.quadratic_function, x)
    hess_trace = hess[:, 0, 0] + hess[:, 1, 1]
    assert torch.allclose(laplacian, hess_trace), \
        "Laplacian ≠ trace of Hessian"


def test_edge_cases() -> None:
    """
    Test edge cases and numerical stability.
    
    Validates operator behavior with very small and large input values
    to ensure numerical stability and proper error handling.
    
    Raises:
        AssertionError: If any edge case test fails.
    """
    x = torch.rand(10, 2, requires_grad=True)
    
    # Test with very small inputs (near zero)
    x_small = torch.ones(10, 2, requires_grad=True) * 1e-6
    gradient_op = Gradient()
    
    # This should not fail for polynomial functions
    grad_small = gradient_op(TestFunctions.quadratic_function, x_small)
    assert not torch.isnan(grad_small).any(), "Gradient contains NaN for small inputs"
    
    # Test with larger inputs
    x_large = torch.ones(10, 2, requires_grad=True) * 100
    grad_large = gradient_op(TestFunctions.quadratic_function, x_large)
    assert not torch.isinf(grad_large).any(), "Gradient contains Inf for large inputs"


def run_operator_tests() -> None:
    """
    Run comprehensive differential operator tests.
    
    Executes all differential operator tests with detailed progress reporting
    and comprehensive error handling. Provides summary of test results.
    
    Raises:
        AssertionError: If any tests fail, with detailed failure information.
    """
    test_functions = [
        ("Partial derivative", test_partial_derivative),
        ("Gradient", test_gradient),
        ("Hessian", test_hessian),
        ("Divergence", test_divergence),
        ("Laplacian basic functionality", test_laplacian_basic),
        ("Laplacian with diffusion", test_laplacian_with_diffusion),
        ("Laplacian-divergence equivalence", test_laplacian_divergence_equivalence),
        ("Jacobian computation", test_jacobian),
        ("Operator consistency", test_operator_consistency),
        ("Edge cases", test_edge_cases),
    ]
    
    failed_tests = []
    for test_name, test_func in test_functions:
        try:
            print(f"Running {test_name}...")
            test_func()
            print(f"✓ {test_name} passed")
        except Exception as e:
            failed_tests.append(f"{test_name}: {e}")
            print(f"✗ {test_name} failed: {e}")
    
    if failed_tests:
        print(f"\n{len(failed_tests)} tests failed:")
        for failure in failed_tests:
            print(f"  - {failure}")
        raise AssertionError(f"{len(failed_tests)} differential operator tests failed")
    else:
        print(f"\nAll {len(test_functions)} differential operator tests passed!")


if __name__ == "__main__":
    run_operator_tests()