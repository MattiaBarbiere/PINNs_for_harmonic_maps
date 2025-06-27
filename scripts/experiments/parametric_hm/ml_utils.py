"""
Machine Learning Utilities for Parametric Harmonic Maps.

This module provides specialized training utilities for parametric harmonic map models,
adapting the standard hmpinn training functions to handle parameter-dependent PDEs.

Functions:
    train_parametric_hm: Main training function for parametric harmonic map models.
"""

import torch
import tqdm

from hmpinn.utils.ml_utils import init_samplers, init_optimizers, construct_loss_fn, sample_domain_points

def train_parametric_hm(model,
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
    Train a parametric harmonic map model using Physics-Informed Neural Networks.
    
    This function implements a two-stage training approach specifically designed
    for parametric harmonic maps, where PDE parameters are treated as additional
    network inputs, enabling learning across parameter spaces.

    Parameters:
        model: ParametricHmModel
            The parametric harmonic map model to train.
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

    # Print PDE class information
    print(f"Solving parametric PDE: {model.PDE_class.__name__}")

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

        # Compute loss (model automatically handles parameter sampling)
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
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        
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