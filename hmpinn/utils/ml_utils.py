"""
Machine Learning Training Utilities.

This module provides utility functions for training Physics-Informed Neural Networks,
including sampler initialization, optimizer setup, and the main training loop.

Functions:
    init_samplers: Initialize interior and boundary point samplers.
    init_optimizers: Initialize optimizers for training.
    construct_loss_fn: Construct PINN loss function.
    sample_domain_points: Sample training points from domain.
    train: Main training function for PINN models.
"""

import torch
import tqdm

from hmpinn.loss_function import PINNLoss
from hmpinn.samplers import *

def init_samplers(interior_sampler, boundary_sampler, seed, boundary_batch_ratio, default_batch_size):
    """
    Initialize interior and boundary point samplers for training.
    
    Sets up samplers with appropriate batch sizes and random seeds,
    creating default samplers if none are provided.

    Parameters:
        interior_sampler: InteriorSampler or None
            Sampler for interior domain points. If None, creates default sampler.
        boundary_sampler: BoundarySampler or None  
            Sampler for boundary points. If None, creates default sampler.
        seed: int or None
            Random seed for reproducibility. Overrides sampler seeds if provided.
        boundary_batch_ratio: float
            Ratio of boundary to interior points in each batch.
        default_batch_size: int
            Default batch size for interior points.

    Returns:
        tuple
            Tuple of (interior_sampler, boundary_sampler) ready for training.
            
    Raises:
        ValueError: If provided samplers are not of the correct type.
    """
    # Initialize interior sampler
    if interior_sampler is None:
        interior_sampler = InteriorSampler(seed=seed, default_batch_size=default_batch_size)
    elif isinstance(interior_sampler, InteriorSampler):
        # Override seed and batch size if provided
        interior_sampler.change_seed(seed)
        interior_sampler.change_default_batch_size(default_batch_size)
    else:
        raise ValueError("The interior_sampler must be an instance of InteriorSampler or None")
    
    # Initialize boundary sampler
    if boundary_sampler is None:
        boundary_sampler = BoundarySampler(seed=seed, default_batch_size=boundary_batch_ratio * default_batch_size)
    elif isinstance(boundary_sampler, BoundarySampler):
        # Override seed and batch size if provided
        boundary_sampler.change_seed(seed)
        boundary_sampler.change_default_batch_size(boundary_batch_ratio * default_batch_size)
    else:
        raise ValueError("The boundary_sampler must be an instance of BoundarySampler or None")
    
    return interior_sampler, boundary_sampler

def init_optimizers(model, optimizer):
    """
    Initialize optimizer for model training.
    
    Creates the specified optimizer with default learning rates and settings.

    Parameters:
        model: torch.nn.Module
            The neural network model to optimize.
        optimizer: str
            Optimizer type ("SGD" or "Adam").

    Returns:
        torch.optim.Optimizer
            Initialized optimizer ready for training.
            
    Raises:
        ValueError: If optimizer type is not recognized.
    """
    if optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=1e-3)
    elif optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError("The optimizer is not recognized")

def construct_loss_fn(model, loss_BC_weight):
    """
    Construct PINN loss function with appropriate boundary condition weighting.
    
    Automatically handles embedding layers by setting boundary condition weight to zero.

    Parameters:
        model: torch.nn.Module
            The neural network model being trained.
        loss_BC_weight: float
            Weight for boundary condition loss (ignored if model has embedding layer).

    Returns:
        PINNLoss
            Configured loss function ready for training.
    """
    # Disable boundary condition loss for models with embedding layers
    if model.has_embedding_layer:
        loss_BC_weight = 0
    
    return PINNLoss(model, weight=loss_BC_weight)

def sample_domain_points(interior_sampler, boundary_sampler, device):
    """
    Sample interior and boundary points for a training batch.

    Parameters:
        interior_sampler: InteriorSampler
            Sampler for interior domain points.
        boundary_sampler: BoundarySampler
            Sampler for boundary points.
        device: torch.device
            Device to move tensors to (CPU or CUDA).

    Returns:
        tuple
            Tuple of (X, X_boundary) containing sampled points on specified device.
    """
    # Sample interior points using default batch size
    X = interior_sampler.sample_batch()
    X = X.to(device)

    # Sample boundary points using default batch size
    X_boundary = boundary_sampler.sample_batch()
    X_boundary = X_boundary.to(device)

    return X, X_boundary

def train(model,
          batch_size=128, 
          n_epochs=12000, 
          optimizer="Adam",
          optimizer_threshold=7000,
          loss_BC_weight=20,
          save_BC_loss=True,
          boundary_batch_ratio=1,
          seed=None,
          interior_sampler=None,
          boundary_sampler=None):
    """
    Train a PINN model to satisfy PDE and boundary conditions.
    
    Uses a two-stage training approach: Adam optimizer initially, then LBFGS
    for fine-tuning. Automatically handles device selection and loss tracking.

    Parameters:
        model: torch.nn.Module
            The PINN model to train.
        batch_size: int, optional
            Training batch size. Overrides sampler defaults. Defaults to 128.
        n_epochs: int, optional
            Total number of training epochs. Defaults to 12000.
        optimizer: str, optional
            Initial optimizer type ("Adam" or "SGD"). Defaults to "Adam".
        optimizer_threshold: int, optional
            Epoch to switch from initial optimizer to LBFGS. Defaults to 7000.
        loss_BC_weight: float, optional
            Weight for boundary condition loss. Ignored if model has embedding layer. Defaults to 20.
        save_BC_loss: bool, optional
            Whether to save boundary condition loss history. Defaults to True.
        boundary_batch_ratio: float, optional
            Ratio of boundary to interior points per batch. Defaults to 1.
        seed: int, optional
            Random seed for reproducibility. Defaults to None.
        interior_sampler: InteriorSampler, optional
            Custom interior point sampler. Defaults to None (creates default).
        boundary_sampler: BoundarySampler, optional
            Custom boundary point sampler. Defaults to None (creates default).

    Returns:
        tuple
            Training history tuple. If save_BC_loss=True: (errors, grad_errors, losses, BC_losses).
            Otherwise: (errors, grad_errors, losses).
    """
    # Select and report device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Move model to selected device
    model.to(device)

    # Print PDE information
    print(f"Solving the PDE: {model.PDE}")

    # Initialize tracking lists for training statistics
    errors, grad_errors, losses, BC_losses = [], [], [], []

    # Initialize optimizer and loss function
    optimizer = init_optimizers(model, optimizer)
    loss_fn = construct_loss_fn(model, loss_BC_weight)

    # Initialize samplers with appropriate parameters
    interior_sampler, boundary_sampler = init_samplers(
        interior_sampler, boundary_sampler, seed, boundary_batch_ratio, default_batch_size=batch_size)

    # Phase 1: Train with initial optimizer (Adam/SGD)
    for epoch in tqdm.tqdm(range(optimizer_threshold)):
        # Sample training points
        X, X_boundary = sample_domain_points(interior_sampler, boundary_sampler, device)

        # Compute loss
        loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
        # Track training statistics
        errors.append(loss_fn.relative_residual_error_value)
        losses.append(loss_fn.loss_value)
        if loss_fn.boundary_loss_value is not None:
            BC_losses.append(loss_fn.boundary_loss_value)
        if loss_fn.relative_grad_error_value is not None:
            grad_errors.append(loss_fn.relative_grad_error_value)

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)      
        optimizer.step()

    # Phase 2: Train with LBFGS optimizer for fine-tuning
    for epoch in tqdm.tqdm(range(optimizer_threshold, n_epochs)):
        # Sample training points
        X, X_boundary = sample_domain_points(interior_sampler, boundary_sampler, device)
        
        # Create LBFGS optimizer for this epoch
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", lr=1e-5)

        # Compute loss for tracking
        loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
        # Track training statistics
        errors.append(loss_fn.relative_residual_error_value)
        losses.append(loss_fn.loss_value)
        if loss_fn.boundary_loss_value is not None:
            BC_losses.append(loss_fn.boundary_loss_value)
        if loss_fn.relative_grad_error_value is not None:
            grad_errors.append(loss_fn.relative_grad_error_value)

        # LBFGS requires closure function for line search
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
            loss.backward()
            return loss
        
        # Backward pass with gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(closure)

    # Return training history
    if save_BC_loss:
        return errors, grad_errors, losses, BC_losses
    else:
        return errors, grad_errors, losses