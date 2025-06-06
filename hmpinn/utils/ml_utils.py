import torch
import tqdm
from hmpinn.loss_function import PINNLoss
from hmpinn.samplers import *


# Function to prepare the samplers for training
def init_samplers(interior_sampler, boundary_sampler, seed, boundary_batch_ratio, default_batch_size):
    # Initialize the samplers if not given
    if interior_sampler is None:
        interior_sampler = InteriorSampler(seed=seed, default_batch_size=default_batch_size)
    elif isinstance(interior_sampler, InteriorSampler):
        # If the sampler was given, then we change the seed to the argument "seed". 
        # If the seed argument is None, the seed of the sampler will not be changed
        interior_sampler.change_seed(seed)
        # If the sampler was given, then we override the batch_size value changing it to the value given to the train function.
        interior_sampler.change_default_batch_size(default_batch_size)
    else:
        raise ValueError("The interior_sampler must be an instance of InteriorSampler or None")
    
    if boundary_sampler is None:
        boundary_sampler = BoundarySampler(seed=seed, default_batch_size=boundary_batch_ratio * default_batch_size)
    elif isinstance(boundary_sampler, BoundarySampler):
        # If the sampler was given, then we change the seed to the argument "seed". 
        # If the seed argument is None, the seed of the sampler will not be changed
        boundary_sampler.change_seed(seed)
        # If the sampler was given, then we override the batch_size value changing it to the value given to the train function. 
        boundary_sampler.change_default_batch_size(boundary_batch_ratio * default_batch_size)
    else:
        raise ValueError("The boundary_sampler must be an instance of BoundarySampler or None")
    
    return interior_sampler, boundary_sampler

def init_optimizers(model, optimizer):
    """
    Prepare the optimizer for training
    """
    #The optimizer
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError("The optimizer is not recognized")
    return optimizer

def construct_loss_fn(model, loss_BC_weight):
    """
    Construct the loss function
    """
    # Construct the loss function
    if model.has_embedding_layer:
        loss_BC_weight = 0
    
    loss_fn = PINNLoss(model, weight=loss_BC_weight)
    return loss_fn

def sample_domain_points(interior_sampler, boundary_sampler, device):
    """
    Sample the interior and boundary points for training

    Parameters:
    interior_sampler (Interior_Point_Sampler): The sampler to use for the interior points
    boundary_sampler (Boundary_Point_Sampler): The sampler to use for the boundary points
    batch_size (int): The size of each batch
    boundary_batch_ratio (float): The ratio of the batch size for the boundary points to the batch size for the interior points
    device (torch.device): The device to use for training

    Returns:
    X (torch.Tensor): The interior points
    X_boundary (torch.Tensor): The boundary points
    """
    # Generate the interior data using default batch size
    X = interior_sampler.sample_batch()
    X = X.to(device)

    # Generate the boundary data (note that if the model has the embedding layer this will be ignored)
    X_boundary = boundary_sampler.sample_batch()
    X_boundary = X_boundary.to(device)

    return X, X_boundary


def train(model,
          batch_size=128, 
          n_epochs=12000, 
          optimizer="Adam",
          optimizer_threshold = 7000,
          loss_BC_weight = 20,
          save_BC_loss = True,
          boundary_batch_ratio = 1,
          seed=None,
          interior_sampler=None,
          boundary_sampler=None):
    """
    Train the model so that the laplacian of the model is close to the target function.

    Parameters:
    model (torch.nn.Module): The model to train
    batch_size (int): The size of each batch. If sampler objects are given this will override the default batch size in the samplers
    n_epochs (int): Number of epochs to train
    optimizer (torch.optim): The optimizer to use (default is Adam)
    optimizer_threshold (int): The epoch to switch to LBFGS
    loss_BC_weight (float): The weight of the boundary condition on the loss function.
                            If equal to 0, the loss function will not include the boundary condition.
                            If the model has an embedding layer, this will be ignored
    save_BC_loss (bool): If True, the boundary condition loss will be saved
    boundary_batch_ratio (float): The ratio of the batch size for the boundary points to the batch size for the interior points.
                        If the model has an embedding layer, this will be ignored
    seed (int): The seed to use for the random number generator. If not None this will override the seed in the samplers
    interior_sampler (Interior_Point_Sampler): The sampler to use for the interior points. If None the defualt sampler will be used
                                                Note: If the arguemnt "seed" in the function train is not None, the seed of the given 
                                                        sampler will be overriden
    boundary_sampler (Boundary_Point_Sampler): The sampler to use for the boundary points. If None the defualt sampler will be used
                                                Note: If the arguemnt "seed" in the function train is not None, the seed of the given 
                                                        sampler will be overriden

    Returns:
    None
    """
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    #Move the model to the device
    model.to(device)

    # Print the PDE that the model is solving
    print(f"Solving the PDE: {model.PDE}")

    #Initialise a lists for statistics
    errors, grad_errors, losses, BC_losses = [], [], [], []

    # Init the optimizer
    optimizer = init_optimizers(model, optimizer)
    
    # Construct the loss function
    loss_fn = construct_loss_fn(model, loss_BC_weight)

    # Initialize the samplers
    interior_sampler, boundary_sampler = init_samplers(interior_sampler, boundary_sampler, seed, boundary_batch_ratio, default_batch_size=batch_size)
    

    for epoch in tqdm.tqdm(range(optimizer_threshold)):

        #For each epoch we generate new data
        X, X_boundary = sample_domain_points(interior_sampler, boundary_sampler, device)

        # Compute the loss
        loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
        #Append values to the lists
        errors.append(loss_fn.relative_residual_error_value)        # Relative residual error
        losses.append(loss_fn.loss_value)                           # RMSE loss
        if loss_fn.boundary_loss_value is not None:
            BC_losses.append(loss_fn.boundary_loss_value)           # RMSE boundary condition loss
        if loss_fn.relative_grad_error_value is not None:
            grad_errors.append(loss_fn.relative_grad_error_value)   # Relative gradient error

        #Backwards pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)      
        optimizer.step()

    for epoch in tqdm.tqdm(range(optimizer_threshold, n_epochs)):

        #For each epoch we generate new data
        X, X_boundary = sample_domain_points(interior_sampler, boundary_sampler, device)
        print("Using lr=1e-5 for LBFGS optimizer")

        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe", lr=1e-5)

        # Compute the loss
        loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
        #Append values to the lists
        errors.append(loss_fn.relative_residual_error_value)        # Relative residual error
        losses.append(loss_fn.loss_value)                           # RMSE loss
        if loss_fn.boundary_loss_value is not None:
            BC_losses.append(loss_fn.boundary_loss_value)           # RMSE boundary condition loss
        if loss_fn.relative_grad_error_value is not None:
            grad_errors.append(loss_fn.relative_grad_error_value)   # Relative gradient error

        #Backward pass
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
            loss.backward()
            return loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(closure)

    if save_BC_loss:
        return errors, grad_errors, losses, BC_losses
    else:
        return errors, grad_errors, losses




# ## NOTE: The following function is only used for model v0
# #Train function for v0 model
# def train_v0(model, loss_fn=nn.MSELoss(),
#           batch_size=128,
#           numb_batchs=100, 
#           n_epochs=1, 
#           optimizer=None,  
#           lr = 1e-3, 
#           show_loss=False):
#     """
#     Train the model so that the laplacian of the model is close to the target function.

#     Parameters:
#     model (torch.nn.Module): The model to train
#     loss_fn (torch.nn.Module): The loss function to use
#     n_epochs (int): Number of epochs to train
#     optimizer (torch.optim): The optimizer to use (default is Adam)
#     lr (float): The learning rate

#     Returns:
#     None
#     """

#     #Move the model to the device
#     model.to(device)

#     #The default optimizer is Adam
#     if optimizer is None:
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     #Initialise a list to keep the root relative error
#     errors = []

#     #Initialise a list to keep the error of the gradient
#     grad_errors = []

#     #Initialise a list to keep the loss
#     losses = []

#     for epoch in tqdm.tqdm(range(n_epochs)):
#         #For each epoch we generate new data
#         data = torch.rand(numb_batchs, batch_size, 2, requires_grad=True)

#         for X in data:
#             X = X.to(device)

#             # Computing the loss
#             laplacian_of_model = laplacian_with_diffusion(model, X, k=model.poisson_equation.diffusion_matrix)
#             real_laplacian = model.poisson_equation.f(X)
#             loss = loss_fn(laplacian_of_model, real_laplacian)

#             #Adding the root relative error
#             errors.append(relative_error(laplacian_of_model,real_laplacian))

#             #Adding the error of the gradient
#             true_grad = model.poisson_equation.grad_u(X)
#             model.zero_grad()
#             model_grad = torch.autograd.grad(model(X), X, grad_outputs=torch.ones_like(model(X)), create_graph=True)[0]
#             grad_errors.append(relative_error(model_grad, true_grad))
        
#             # Adding the RMSE loss to the list
#             losses.append(torch.sqrt(loss).item())

#             #Backwards pass
#             optimizer.zero_grad()
#             loss.backward()       
#             optimizer.step()

#             if show_loss:
#                 print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}')

#     return errors, grad_errors, losses