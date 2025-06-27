"""
PINN Loss Function.

This module provides the loss function implementation for Physics-Informed Neural Networks,
combining residual loss and boundary condition loss with automatic weight management.

Classes:
    PINNLoss: Main loss function class for PINN training.
"""

import torch

class PINNLoss():
    """
    Loss function class for Physics-Informed Neural Networks.
    
    This class combines residual loss (from PDE satisfaction) and boundary condition loss
    with configurable weighting. It automatically handles embedding layers by disabling
    boundary condition loss when appropriate.
    
    Attributes:
        model: The neural network model being trained.
        BC_weight: Weight for boundary condition loss component.
        loss: Combined loss tensor (MSE).
        residual_loss: PDE residual loss component.
        boundary_loss: Boundary condition loss component.
        loss_value: RMSE loss value for tracking.
        boundary_loss_value: RMSE boundary loss value for tracking.
        relative_residual_error_value: Relative L2 residual error.
        relative_grad_error_value: Relative L2 gradient error.
    """

    def __init__(self, model, weight=0):
        """
        Initialize the PINN loss function.

        Parameters:
            model: torch.nn.Module
                The model to be used for loss computation. If the model has an embedding layer,
                the loss will not include the boundary condition and will override any value of weight.
            weight: float, optional
                The weight of the boundary condition on the loss function.
                If equal to 0, the loss function will not include the boundary condition.
                Defaults to 0.
                
        Raises:
            ValueError: If weight is negative.
        """
        # Validate weight parameter
        if weight < 0:
            raise ValueError("The weight must be strictly positive or equal to 0.")
        
        self.model = model
        self.BC_weight = weight

        # If the model has an embedding layer, we don't need to compute the boundary loss
        if self.model.has_embedding_layer:
            self.BC_weight = 0

        # Initialize loss components
        self.loss = None
        self.residual_loss = None
        self.boundary_loss = None
        
        # Initialize tracking statistics for plotting and analysis
        self.loss_value = None                      
        self.boundary_loss_value = None             
        self.relative_residual_error_value = None  
        self.relative_grad_error_value = None      
        
    def __call__(self, y, X, y_boundary=None, X_boundary=None):
        """
        Compute the combined PINN loss.
        
        Evaluates both residual loss (PDE satisfaction) and boundary condition loss
        (if applicable), combining them with the specified weight.

        Parameters:
            y: torch.Tensor
                The output of the model on interior points.
            X: torch.Tensor
                The input coordinates for interior points.
            y_boundary: torch.Tensor, optional
                The output of the model at boundary points.
            X_boundary: torch.Tensor, optional
                The input coordinates for boundary points.

        Returns:
            torch.Tensor
                The computed combined loss (MSE).
        """
        # Compute the residual loss
        self.residual_loss = self.model.PDE.compute_residual(y, X)

        # Access the relative residual error from the PDE's BaseResidual class
        self.relative_residual_error_value = self.model.PDE.relative_residual_error.item()

        # Compute the relative gradient error if analytical solution exists
        relative_grad_error = self.model.PDE.compute_relative_grad_error(self.model, X)
        if relative_grad_error is not None:
            self.relative_grad_error_value = relative_grad_error.item()

        # If weight is 0, return only the residual loss
        if self.BC_weight == 0:
            self.loss = self.residual_loss
            self.loss_value = torch.sqrt(self.loss).item()
            return self.loss
        
        # Compute the boundary loss if weight is not 0
        self.boundary_loss = self.model.PDE.compute_boundary_loss(y_boundary, X_boundary)

        # Compute the boundary loss RMSE for tracking
        self.boundary_loss_value = torch.sqrt(self.boundary_loss).item()

        # Combine residual and boundary losses with weighting
        self.loss = self.residual_loss + self.BC_weight * self.boundary_loss
        self.loss_value = torch.sqrt(self.loss).item()

        return self.loss