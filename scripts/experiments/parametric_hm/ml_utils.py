'''
These are slight modifications of the hmpinn.utils.ml_utils module to fit the parametric harmonic maps model.
'''

import torch
import tqdm
from hmpinn.utils.ml_utils import init_samplers, init_optimizers, construct_loss_fn, sample_domain_points



def train_parametric_hm(model,
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
    print(f"Solving the PDE: {model.PDE_class.__name__}")

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

        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        
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