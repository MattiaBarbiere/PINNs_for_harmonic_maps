from hmpinn.PDEs import * 
from hmpinn.models import * 
from hmpinn import DEFAULT_CONFIG
from hmpinn.PDEs import PDE_NAME_TO_CLASS
import copy 

def get_PDE_class(poisson_equation: str):
    """
    Returns the class of the PDE given its name.

    Parameters:
    poisson_equation (str): The name of the PDE

    Returns:
    class: The class of the PDE
    """
    if poisson_equation not in PDE_NAME_TO_CLASS:
        raise ValueError(f"The PDE {poisson_equation} is not recognized")
    
    return PDE_NAME_TO_CLASS[poisson_equation]

# A function that given a string returns the corresponding PDE
def get_PDE_object(params: dict, backend) -> object:
    """
    Function that returns the PDE object given the parameters.

    Parameters:
    params (dict): The parameters of the PDE (see DEFAULT_CONFIG for the structure)
    backend (torch or np): The backend to use for the PDE. This can be either "numpy" or "torch".
    """
    # Copy the parameters
    params = params.copy()

    # Get the PDE parameters from the dict
    pde_params = params["PDE"]

    # Get the kwargs of the PDE from the dict
    pde_kwargs = pde_params["PDE_kwargs"]

    # Append the backend to the parameters
    pde_kwargs["backend"] = backend
    
    # Get the class of the PDE
    name = pde_params["name"]
    pde_class = get_PDE_class(name)

    return pde_class(**pde_kwargs)

def get_model_class(model_type: str):
    """
    Returns the class of the model given its name.

    Parameters:
    model_type (str): The name of the model

    Returns:
    class: The class of the model
    """
    if model_type == "v0":
        return ModelV0
    elif model_type == "v1":
        return ModelV1
    elif model_type == "v2":
        return ModelV2
    else:
        raise ValueError("The model type is not recognized")

def flatten_dict(d: dict) -> dict:
    """
    Flattens a nested dictionary.

    Parameters:
    d (dict): The dictionary to flatten

    Returns:
    dict: The flattened dictionary
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def organise_dict(d: dict) -> dict:
    """
    Organises the dictionary by grouping parameters into subdictionaries. This help to 
    standerdise the parameters and make them easier to read.
    Note: The code is ugly, but it was made to make sure older data from experiments are still compatible with new version of the package.

    Parameters:
    d (dict): The dictionary to be organised

    Returns:
    dict: The organised dictionary
    """    
    # Flatten the dict to make it easier to work with
    d = flatten_dict(d)

    # To store the organised dict, initially we use the default config in a deep copy
    res_dict = copy.deepcopy(DEFAULT_CONFIG)

    for key in d.keys():
        # Keys for the PDE
        if key == "name" or key == "poisson_equation":
            res_dict["PDE"]["name"] = d[key]
        
        elif key in ["a", "b", "amplitude"]:
            # These kwargs are only for the eigenfunction PDE
            if res_dict["PDE"]["name"] in ["eigenfunc", "eigenfunc_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the eigenfunction PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
        
        elif key == "const_value":
            # This kwargs is only for the constant source PDE
            if res_dict["PDE"]["name"] in ["const_source", "const_source_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key 'const_value' is only valid for the constant source PDE. Input was 'const_value': {d[key]} for {res_dict['PDE']['name']}")
        
        elif key in ["mu_x", "mu_y", "std_x", "std_y"]:
            # These kwargs are only for the gaussian bump PDE
            if res_dict["PDE"]["name"] in ["gaussian_bump_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the gaussian bump PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key == "K":
            # This kwargs is only for the convection dominated PDE
            if res_dict["PDE"]["name"] in ["convection_dominated", "convection_dominated_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"]["K"] = d[key]
            else:
                raise ValueError(f"The key 'K' is only valid for the convection dominated PDE. Input was 'K': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key in ["curvature", "frequency_x", "frequency_y"]:
            # These kwargs are only for the sin boundaries harmonic map
            if res_dict["PDE"]["name"] in ["sin_boundaries_hm"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the sin boundaries harmonic map PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key in ["a_left", "a_right", "b_top", "b_bottom", "degree"]:
            # These kwargs are only for the sin boundaries harmonic map
            if res_dict["PDE"]["name"] in ["poly_boundaries_hm"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the sin boundaries harmonic map PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")

        # Keys for the model
        elif key == "type":
            res_dict["model"]["type"] = d[key]
        elif key == "activation_function":
            res_dict["model"]["model_kwargs"]["activation_function"] = d[key]
        elif key == "hidden_layers" or key == "nodes_hidden_layers":
            res_dict["model"]["model_kwargs"]["nodes_hidden_layers"] = d[key]
        elif key == "embeddings_per_dim":
            res_dict["model"]["model_kwargs"]["embeddings_per_dim"] = d[key]
        elif key == "embedding_layer" or key == "has_embedding_layer":
            res_dict["model"]["model_kwargs"]["has_embedding_layer"] = d[key]
        elif key == "output_dim":
            res_dict["model"]["model_kwargs"]["output_dim"] = d[key]
        
        # Keys for training
        elif key == "n_epochs" or key == "epochs":
            res_dict["train"]["n_epochs"] = d[key]
        elif key == "batch_size":
            res_dict["train"]["batch_size"] = d[key]
        elif key == "optimizer":
            res_dict["train"]["optimizer"] = d[key]
        elif key == "optimizer_threshold":
            res_dict["train"]["optimizer_threshold"] = d[key]
        elif key == "loss_BC_weight":
            res_dict["train"]["loss_BC_weight"] = d[key]
        elif key == "boundary_batch_ratio":
            res_dict["train"]["boundary_batch_ratio"] = d[key]
        elif key == "seed":
            res_dict["train"]["seed"] = d[key]
        elif key == "interior_sampler":
            res_dict["train"]["interior_sampler"] = d[key]
        elif key == "boundary_sampler":
            res_dict["train"]["boundary_sampler"] = d[key]
        elif key == "save_BC_loss":
            res_dict["train"]["save_BC_loss"] = d[key]
        
        # Keys for the solver
        elif key == "nx":
            res_dict["solver"]["nx"] = d[key]
        elif key == "ny":
            res_dict["solver"]["ny"] = d[key]
        elif key == "p":
            res_dict["solver"]["p"] = d[key]
        
        # This is an old key that is not guaranteed to be in the dict
        elif key == "numb_batches":
            res_dict["train"]["numb_batches"] = d[key]
        else:
            # If the key is not recognised, raise an error
            raise ValueError(f"The key {key} was not added to the organised dict")
        
    return res_dict
        
        
        
        


    



# def train(model,
#           batch_size=128, 
#           n_epochs=12000, 
#           optimizer="SGD",
#           optimizer_threshold = 11999,
#           loss_BC_weight = 1,
#           save_BC_loss = False,
#           boundary_batch_ratio = 1,
#           seed=None,
#           interior_sampler=None,
#           boundary_sampler=None):
#     """
#     Train the model so that the laplacian of the model is close to the target function.

#     Parameters:
#     model (torch.nn.Module): The model to train
#     batch_size (int): The size of each batch
#     n_epochs (int): Number of epochs to train
#     optimizer (torch.optim): The optimizer to use (default is SGD)
#     optimizer_threshold (int): The epoch to switch to LBFGS
#     loss_BC_weight (float): The weight of the boundary condition on the loss function.
#                             If equal to 0, the loss function will not include the boundary condition.
#                             If the model has an embedding layer, this will be ignored
#     save_BC_loss (bool): If True, the boundary condition loss will be saved
#     boundary_batch_ratio (float): The ratio of the batch size for the boundary points to the batch size for the interior points.
#                         If the model has an embedding layer, this will be ignored
#     seed (int): The seed to use for the random number generator. If not None this will override the seed in the samplers
#     interior_sampler (Interior_Point_Sampler): The sampler to use for the interior points. If None the defualt sampler will be used
#                                                 Note: If the arguemnt "seed" in the function train is not None, the seed of the given 
#                                                         sampler will be overriden
#     boundary_sampler (Boundary_Point_Sampler): The sampler to use for the boundary points. If None the defualt sampler will be used
#                                                 Note: If the arguemnt "seed" in the function train is not None, the seed of the given 
#                                                         sampler will be overriden

#     Returns:
#     None
#     """
#     # Select device
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("Using Cuda")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")

#     #Move the model to the device
#     model.to(device)

#     #Initialise a list to keep the root relative error
#     errors = []

#     #Initialise a list to keep the error of the gradient
#     grad_errors = []

#     #Initialise a list to keep the loss
#     losses = []

#     # The list to keep the boundary condition loss
#     BC_losses = []

#     #The optimizer
#     if optimizer == "SGD":
#         optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     elif optimizer == "Adam":
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     else:
#         raise ValueError("The optimizer is not recognized")
    
#     # Construct the loss function
#     if model.has_embedding_layer:
#         loss_BC_weight = 0
    
#     loss_fn = PINNLoss(model, weight=loss_BC_weight)

#     # Initialize the samplers
#     interior_sampler, boundary_sampler = init_samplers(interior_sampler, boundary_sampler, seed)
    

#     for epoch in tqdm.tqdm(range(optimizer_threshold)):
#         #For each epoch we generate new data
#         X = interior_sampler.sample_batch(batch_size)
#         X = X.to(device)

#         # Generate the boundary data (note that if the model has the embedding layer this will be ignored)
#         X_boundary = boundary_sampler.sample_batch(boundary_batch_ratio*batch_size)
#         X_boundary = X_boundary.to(device)

#         # Compute the loss
#         loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
#         #Append values to the lists
#         errors.append(loss_fn.relative_residual_error_value)        # Relative residual error
#         losses.append(loss_fn.loss_value)                           # RMSE loss
#         if loss_fn.boundary_loss_value is not None:
#             BC_losses.append(loss_fn.boundary_loss_value)           # RMSE boundary condition loss
#         if loss_fn.relative_grad_error_value is not None:
#             grad_errors.append(loss_fn.relative_grad_error_value)   # Relative gradient error

#         #Backwards pass
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)      
#         optimizer.step()

#     for epoch in tqdm.tqdm(range(optimizer_threshold, n_epochs)):
#         #For each epoch we generate new data
#         X = interior_sampler.sample_batch(batch_size)
#         X = X.to(device)

#         # Generate the boundary data (note that if the model has the embedding layer this will be ignored)
#         X_boundary = boundary_sampler.sample_batch(boundary_batch_ratio*batch_size)
#         X_boundary = X_boundary.to(device)

#         optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
        
#         # Compute the loss
#         loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
        
#         #Append values to the lists
#         errors.append(loss_fn.relative_residual_error_value)        # Relative residual error
#         losses.append(loss_fn.loss_value)                           # RMSE loss
#         if loss_fn.boundary_loss_value is not None:
#             BC_losses.append(loss_fn.boundary_loss_value)           # RMSE boundary condition loss
#         if loss_fn.relative_grad_error_value is not None:
#             grad_errors.append(loss_fn.relative_grad_error_value)   # Relative gradient error

#         #Backward pass
#         def closure():
#             optimizer.zero_grad()
#             loss = loss_fn(model(X), X, model(X_boundary), X_boundary)
#             loss.backward()
#             return loss
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#         optimizer.step(closure)

#     if save_BC_loss:
#         return errors, grad_errors, losses, BC_losses
#     else:
#         return errors, grad_errors, losses




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