"""
PDE Utilities.

This module provides utility functions for PDE computations in the hmpinn library,
including backend management, tensor operations, and mathematical functions.

Functions:
    check_backend: Validate backend compatibility.
    ensure_backend: Ensure input matches backend type.
    backend_to_str: Convert backend to string representation.
    stack: Stack tensors along specified dimension.
    zeros: Create tensor of zeros.
    ones: Create tensor of ones.
    all: Check if all elements are True.
    norm: Compute tensor norm.
    frobenius_prod: Compute Frobenius product of matrices.
    relative_error: Compute relative error between tensors.
"""

import torch
import numpy as np
import nutils
import nutils.function as fn

def check_backend(backend):
    """
    Validate backend compatibility.

    Parameters:
        backend: torch or np
            The backend to validate.

    Returns:
        bool
            True if backend is valid.
            
    Raises:
        ValueError: If backend is not supported.
    """
    if backend not in [torch, np]:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")
    return True

def ensure_backend(input, backend):
    """
    Ensure input tensor matches the specified backend.

    Parameters:
        input: torch.Tensor or np.ndarray
            The input tensor to validate.
        backend: torch or np
            The expected backend.

    Returns:
        torch.Tensor or np.ndarray
            The input tensor if it matches the backend.
            
    Raises:
        ValueError: If input type doesn't match backend.
    """
    if backend == torch:
        if isinstance(input, torch.Tensor):
            return input
        else:
            raise ValueError(f"With torch backend, input must be a torch.Tensor not {type(input)}")
    elif backend == np:
        if isinstance(input, np.ndarray) or isinstance(input, nutils.function._Transpose):
            return input
        else:
            raise ValueError(f"With numpy backend, input must be a numpy.ndarray not {type(input)}")
    else:
        raise ValueError(f"Input {type(input)} is incompatible with the backend {backend}.")
    
def backend_to_str(backend):
    """
    Convert backend to string representation.

    Parameters:
        backend: torch or np
            The backend to convert.

    Returns:
        str
            String representation of the backend.
            
    Raises:
        ValueError: If backend is not supported.
    """
    if backend == torch:
        return 'torch'
    elif backend == np:
        return 'numpy'
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")

def stack(X, dim=1, backend=torch):
    """
    Stack tensors along specified dimension.

    Parameters:
        X: sequence of torch.Tensor or np.ndarray
            Sequence of tensors to stack.
        dim: int, optional
            Dimension along which to stack. Defaults to 1.
        backend: torch or np, optional
            Backend to use for stacking. Defaults to torch.

    Returns:
        torch.Tensor or np.ndarray
            Stacked tensor.
            
    Raises:
        ValueError: If backend is not supported.
    """
    if backend == torch:
        return torch.stack(X, dim=dim)
    elif backend == np:
        return np.stack(X, axis=dim)
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")

def zeros(x, backend=torch):
    """
    Create a tensor of zeros with specified dimensions and device.

    Parameters:
    x (torch.Tensor or np.array): The input tensor to get the shape from.
    backend (torch or np): The backend to use for creating the tensor.

    Returns:
    torch.Tensor or np.array: The tensor of zeros.
    """
    if isinstance(x, fn.Array):
        # Symbolic: create array of zeros with appropriate shape
        shape = (1,)
        return fn.zeros(shape)

    if backend == torch:
        return torch.zeros_like(x, device=x.device)
    elif backend == np:
        return np.zeros_like(x, dtype=np.float64)
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")
    
def ones(x, backend=torch, add_extra_dim=False):
    """
    Create a tensor of ones with specified dimensions and device.

    Parameters:
    x (torch.Tensor or np.array): The input tensor to get the shape from.
    backend (torch or np): The backend to use for creating the tensor.
    add_extra_dim (bool): Whether to add an extra dimension to the tensor of ones.

    Returns:
    torch.Tensor or np.array: The tensor of ones.
    """
    if isinstance(x, fn.Array):
        # Symbolic: create array of ones with appropriate shape
        shape = (1,)
        return fn.ones(shape)

    dim = (x.shape[0], 1) if add_extra_dim else (x.shape[0],)
    if backend == torch:
        return torch.ones(dim, device=x.device)
    elif backend == np:
        return np.ones(dim, dtype=np.float64)
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")
    
def all(x, dim=1, backend=torch):
    """
    Check if all elements in the tensor are True.

    Parameters:
    x (torch.Tensor or np.array): The input tensor.
    dim (int): The dimension along which to check for True values.

    Returns:
    bool: True if all elements are True, False otherwise.
    """
    
    if isinstance(x, fn.Array):
        # Symbolic: create array of ones with appropriate shape
        shape = (1,)
        return fn.ones(shape)

    if backend == torch:
        return torch.all(x, dim=dim)
    elif backend == np:
        return np.all(x, axis=dim)
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")



def norm(x, backend=torch):
    """
    Compute the norm of a tensor.

    Parameters:
    x (torch.Tensor or np.array): The input tensor. of size (batch_size, 2)
    backend (torch or np): The backend to use for computing the norm.

    Returns:
    torch.Tensor or np.array: The norm of the tensor.
    """
    if backend == torch:
        return torch.norm(x, dim=1)
    elif backend == np:
        return np.linalg.norm(x, axis=1)
    else:
        raise ValueError(f"Backend {backend} is not supported. Use torch or np.")

def frobenius_prod(a, b, backend=torch):
    """
    Compute the Frobenius product of two tensors.

    Parameters:
    a (torch.Tensor or np.array): The first tensor.
    b (torch.Tensor or np.array): The second tensor.
    backend (torch or np): The backend to use for computing the product.

    Returns:
    torch.Tensor or np.array: The Frobenius product of the tensors.
    """
    # Check that dimensions match
    if a.shape != b.shape:
        raise ValueError(f"Shapes of tensors do not match: {a.shape} and {b.shape}")
    assert len(a.shape) == 3, f"Expected 3D tensors, got {len(a.shape)}D tensors"
    assert a.shape[1] == a.shape[2], f"Expected square tensors, got {a.shape[1]}x{a.shape[2]} tensors"

    return a[:, 0, 0] * b[:, 0, 0] + a[:, 0, 1] * b[:, 0, 1] + a[:, 1, 0] * b[:, 1, 0] + a[:, 1, 1] * b[:, 1, 1]
    
#A function to compute the relative error
def relative_error(t1, t2):
    """
    Compute relative error between two tensors.
    
    Calculates the relative L2 error: ||t1 - t2|| / ||t2||

    Parameters:
        t1: torch.Tensor
            First tensor.
        t2: torch.Tensor
            Second tensor (reference).

    Returns:
        torch.Tensor
            Relative error between the tensors.
    """
    return (torch.norm(t1 - t2) / torch.norm(t2))



