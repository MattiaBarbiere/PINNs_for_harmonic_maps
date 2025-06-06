import torch

### Utils for loss functions ###

# class PINN_Loss():
#     """Class for the loss function of the PINN."""

#     def __init__(self, model, weight=0):
#         """
#         Parameters:
#         model (callable): The model to be used for loss computation.
#         weight (float): The weight of the boundary condition on the loss function.
#                         If equal to 0, the loss function will not include the boundary condition.
#         """
#         # Check if weight is 0
#         if weight < 0:
#             raise ValueError("The weight must be strictly positive or equal to 0.")
        
#         # Initialize the model and parameters
#         self.model = model
#         self.f = model.poisson_equation.f
#         self.diffusion_matrix = model.poisson_equation.diffusion_matrix
#         self.BC = model.poisson_equation.boundary_condition
#         self.BC_weight = weight

#         # Initialize loss components
#         self.loss = None                            # The loss (torch.Tensor)
#         self.relative_residual_error = None         # The relative L^2 residual error of the loss (torch.Tensor)
#         self.relative_grad_error = None             # The relative L^2 error of the gradient (torch.Tensor)
#         self.residual_loss = None                   # The residual loss (== loss_value is BC_weight is 0) (torch.Tensor)   
#         self.boundary_loss = None                   # The boundary loss (== None if BC_weight is 0) (torch.Tensor)

#         # Initialize the statistics of the loss (for plotting)
#         self.loss_value = None                            # The loss value (float)
#         self.relative_residual_error_value = None         # The relative L^2 residual error of the loss (float)
#         self.relative_grad_error_value = None             # The relative L^2 error of the gradient (float)
#         self.residual_loss_value = None                   # RMSE of the the residual loss (== loss_value is BC_weight is 0) (torch.Tensor)   
#         self.boundary_loss_value = None                   # RMSE of the boundary loss (== None if BC_weight is 0) (torch.Tensor)

#     def compute_residual_loss(self, y, X):
#         """
#         Compute the residual loss and the relative residual loss.

#         Parameters:
#         y (torch.Tensor): The output of the model.
#         X (torch.Tensor): The input to the model.

#         Returns:
#         None (the values are stored in the class attributes)
#         """
#         # Compute the residual loss depending on the form of the Poisson equation
#         if self.model.poisson_equation.is_in_non_divergence_form:
#             # If the Poisson equation is in non-divergent form, we compute the Frobenius norm
#             hessian_of_model = hessian(y, X)
#             frobenius_prod = (self.diffusion_matrix(X) * hessian_of_model).sum(dim=(1,2))
#             real_source = self.f(X)

#             # Compute the relative residual error
#             self.relative_residual_error = relative_error(frobenius_prod, real_source)

#             # Compute the residual loss
#             self.residual_loss = nn.MSELoss()(frobenius_prod, real_source)

#         else:
#             # If the Poisson equation is in divergent form, we compute the Laplacian
#             laplacian_of_model = laplacian_with_diffusion(y, X, k=self.diffusion_matrix)
#             real_laplacian = self.f(X)

#             # Compute the relative residual error
#             self.relative_residual_error = relative_error(laplacian_of_model,real_laplacian)

#             # Compute the residual loss
#             self.residual_loss = nn.MSELoss()(laplacian_of_model, real_laplacian)
    
#     def compute_boundary_loss(self, y, X):
#         """
#         Compute the boundary loss and the relative boundary loss.

#         Parameters:
#         y (torch.Tensor): The output of the model.
#         X (torch.Tensor): The input to the model.

#         Returns:
#         None (the values are stored in the class attributes)
#         """
#         # Compute the boundary loss
#         self.boundary_loss = nn.MSELoss()(y, self.BC(X))

#     def compute_relative_grad_error(self, X):
#         """
#         Compute the relative gradient error if gradient of the true solution exists

#         Parameters:
#         X (torch.Tensor): The input to the model.

#         Returns:
#         None (the values are stored in the class attributes)
#         """
#         # Compute the true gradient if it exists
#         true_grad = self.model.poisson_equation.grad_u(X)
        
#         # If the true gradient does not exist, return None
#         if true_grad is None:
#             return None

#         # Compute the model gradient
#         self.model.zero_grad()
#         model_grad = gradient(self.model(X), X)

#         # Compute the relative gradient error
#         self.relative_grad_error = relative_error(model_grad,true_grad)

#     def __call__(self, y, X, y_boundary=None, X_boundary=None):
#         """
#         Compute the loss.

#         Parameters:
#         y (torch.Tensor): The output of the model.
#         X (torch.Tensor): The input to the model.
#         y_boundary (torch.Tensor): The output of the model at the boundary.
#         X_boundary (torch.Tensor): The input to the model at the boundary.

#         Returns:
#         torch.Tensor: The computed loss.
#         """
#         # Compute and store the residual loss
#         self.compute_residual_loss(y, X)
#         self.residual_loss_value = torch.sqrt(self.residual_loss).item()
#         self.relative_residual_error_value = self.relative_residual_error.item()

#         # Compute and store the relative gradient error
#         self.compute_relative_grad_error(X)
#         if self.relative_grad_error is not None:
#             self.relative_grad_error_value = self.relative_grad_error.item()

#         # If weight is 0, return only the residual loss
#         if self.BC_weight == 0:
#             self.loss = self.residual_loss
#             self.loss_value = torch.sqrt(self.loss).item()
#             return self.loss

#         # Compute and store the boundary loss if weight is not 0
#         self.compute_boundary_loss(y_boundary, X_boundary)
#         self.boundary_loss_value = torch.sqrt(self.boundary_loss).item()

#         # Save the loss
#         self.loss = self.residual_loss + self.BC_weight * self.boundary_loss
#         self.loss_value = torch.sqrt(self.loss).item()

#         return self.loss

class PINNLoss():
    """Class for the loss function of the PINN."""

    def __init__(self, model, weight=0):
        """
        Parameters:
        model (callable): The model to be used for loss computation. Note that if the model has an embedding layer, the loss will not include the boundary condition
                            and will override any value of weight.
        weight (float): The weight of the boundary condition on the loss function.
                        If equal to 0, the loss function will not include the boundary condition.
        """
        # Check if weight is 0
        if weight < 0:
            raise ValueError("The weight must be strictly positive or equal to 0.")
        self.model = model
        self.BC_weight = weight

        # If the model has an embedding layer, we don't need to compute the boundary loss
        if self.model.has_embedding_layer:
            self.BC_weight = 0

        # Initialize loss components
        self.loss = None            # The loss MSE (torch.Tensor)
        self.residual_loss = None  # The residual loss (== loss_value is BC_weight is 0) (torch.Tensor)
        self.boundary_loss = None  # The boundary loss (== None if BC_weight is 0) (torch.Tensor)
        
        
        # Initialize the statistics of the loss (for plotting)
        self.loss_value = None                      # The loss value RMSE (float)
        self.boundary_loss_value = None             # The boundary loss RMSE (== None if BC_weight is 0) (float)
        self.relative_residual_error_value = None  # The relative L^2 residual error of the loss (float)
        self.relative_grad_error_value = None      # The relative L^2 error of the gradient (float) (None if gradient of the true solution does not exist)
        
    def __call__(self, y, X, y_boundary=None, X_boundary=None):
        """
        Compute the loss.

        Parameters:
        y (torch.Tensor): The output of the model.
        X (torch.Tensor): The input to the model.
        y_boundary (torch.Tensor): The output of the model at the boundary.
        X_boundary (torch.Tensor): The input to the model at the boundary.

        Returns:
        torch.Tensor: The computed loss.
        """
        # Compute the residual loss
        self.residual_loss = self.model.PDE.compute_residual(y, X)

        # Access the attribute from the BaseResidual class
        self.relative_residual_error_value = self.model.PDE.relative_residual_error.item()

        # Compute the relative gradient error if gradient of the true solution exists
        if self.model.PDE.compute_relative_grad_error(self.model, X) is not None:
            self.relative_grad_error_value = self.model.PDE.compute_relative_grad_error(self.model, X).item()

        # If weight is 0, return only the residual loss
        if self.BC_weight == 0:
            self.loss = self.residual_loss
            self.loss_value = torch.sqrt(self.loss).item()
            return self.loss
        
        # Compute the boundary loss if weight is not 0
        self.boundary_loss = self.model.PDE.compute_boundary_loss(y_boundary, X_boundary)

        # Compute the boundary loss RMSE
        self.boundary_loss_value = torch.sqrt(self.boundary_loss).item()

        # Save the loss
        self.loss = self.residual_loss + self.BC_weight * self.boundary_loss
        self.loss_value = torch.sqrt(self.loss).item()

        return self.loss

        
        



        