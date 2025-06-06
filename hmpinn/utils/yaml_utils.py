import yaml
import os
from hmpinn.models import *
import torch
from hmpinn.utils.utils import get_PDE_class, get_PDE_object, organise_dict
from hmpinn.PDEs.PDE_factory import construct_PDE_class

def read_yaml_file(path):
    """
    Function that opens the yaml file and returns the parameters
    """
    # Get the absolute path
    path = os.path.abspath(path)
    with open(os.path.join(path, ".hydra", "config.yaml"), 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Organise the dict of parameters
    return organise_dict(params)

# def get_PDE_name_from_params(params):
#     """
#     Function that gets the PDE name from the parameters
#     """
#     if 'PDE' in params:
#         PDE_name = params['PDE']["name"]
#     elif "name" in params:
#         PDE_name = params["name"]
#     elif 'poisson_equation' in params:
#         PDE_name = params['poisson_equation']
#     else:
#         warnings.warn("The poisson_equation parameter is not found in the yaml file. Using the default NonSymDiffusion")
#         PDE_name = "diff"
    
#     return PDE_name

def load_model_v0(path, backend=torch):
    """
    Function that loads the model and the errors for model v0

    Parameters:
    path (str): The path to the model directory.
    backend (torch or np): The backend to use for the model. Default is torch.

    Returns:
    model (ModelV0): The loaded model.
    errors (torch.Tensor): The loaded errors.
    grad_errors (torch.Tensor): The loaded gradient errors.
    loss (torch.Tensor): The loaded loss.
    BC_loss (torch.Tensor): The loaded boundary condition loss.
    """
    # Open the yaml file to read the parameters
    params = read_yaml_file(path)
    
    # Extract the parameters
    embeddings_per_dim = params["model"]["model_kwargs"]['embeddings_per_dim']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']
    has_embedding_layer = params["model"]["model_kwargs"]['has_embedding_layer']

    # try:
    #     embedding_layer = params['embedding_layer']
    # except:
    #     embedding_layer = params["has_embedding_layer"]

    PDE_obj = get_PDE_object(params, backend=backend)

    # Define and load the model
    model = ModelV0(PDE_obj, 
                     embedding_size_per_dim=embeddings_per_dim, 
                     has_embedding_layer=has_embedding_layer,
                     nodes_hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load errors
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Check is the file has BC loss
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss

def load_model_v1(path, backend=torch):
    """
    Function that loads the model and the errors for model v1
    """
    # Open the yaml file to read the parameters
    params = read_yaml_file(path)
    
    # Extract the parameters
    embeddings_per_dim = params["model"]["model_kwargs"]['embeddings_per_dim']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']

    PDE_obj = get_PDE_object(params, backend=backend)

    # Define and load the model
    model = ModelV1(PDE_obj, 
                     embedding_size_per_dim=embeddings_per_dim,
                     nodes_hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load errors
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Check is the file has BC loss
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss


def load_model_v2(path, backend=torch):
    """
    Function that loads the model and the errors for model v1
    """
    # Open the yaml file to read the parameters
    params = read_yaml_file(path)
    
    # Extract the parameters
    activation_function = params["model"]["model_kwargs"]['activation_function']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']
    output_dim = params["model"]["model_kwargs"]['output_dim']

    PDE_obj = get_PDE_object(params, backend=backend)

    # Define and load the model
    model = ModelV2(PDE_obj, 
                     nodes_hidden_layers=hidden_layers,
                     activation_function=activation_function,
                     output_dim=output_dim)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load errors
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Check is the file has BC loss
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss

# Function to load the model
def load_model(path, backend=torch):
    """
    Function that loads the model and the errors for any model
    """
    # Open the yaml file to read the parameters
    params = read_yaml_file(path)

    # Extract the model type
    model_type = params["model"]["type"]

    if model_type == "v0":
        return load_model_v0(path, backend=backend)
    elif model_type == "v1":
        return load_model_v1(path, backend=backend)
    elif model_type == "v2":
        return load_model_v2(path, backend=backend)
    else:
        ValueError(f"The model type '{model_type}' is not recognized. Supported types are: v0, v1, v2.")

def load_PDE_ymal(file_path: str):
    """
    Gets the PDE parameters from the YAML file.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    data = read_yaml_file(file_path)
    return data["PDE"]

def load_solver_ymal(file_path: str):
    """
    Gets the solver parameters from the YAML file.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    data = read_yaml_file(file_path)
    return data["solver"]

def load_train_ymal(file_path: str):
    """
    Gets the training parameters from the YAML file.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    data = read_yaml_file(file_path)
    return data["train"]

def load_model_ymal(file_path: str):
    """
    Gets the model parameters from the YAML file.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    data = read_yaml_file(file_path)
    return data["model"]
    
def PDE_from_yaml(yaml_dict: dict):
    """
    Constructs the PDE class from the YAML dictionary.

    Parameters:
    yaml_dict (dict): The YAML dictionary containing the PDE parameters.

    Returns:
    object: The constructed PDE class.
    """
    # If the dict has a key "name" we try to use an already implemented PDE class
    if "name" in yaml_dict:
        return get_PDE_class(yaml_dict["name"])

    # Check if the give PDE is in divergence form
    is_in_divergence_form = yaml_dict["is_in_divergence_form"]

    # Get the source from yaml_dict
    def f(x: torch.Tensor, backend):
        return eval(yaml_dict["f"])
    
    # Get the diffusion matrix from yaml_dict
    def diffusion_matrix(x: torch.Tensor, backend):
        return eval(yaml_dict["diffusion_matrix"])
    
    # Get the boundary condition from yaml_dict
    def BC(x: torch.Tensor, backend):
        return eval(yaml_dict["BC"])
    
    # Get the analytical solution if it exists
    if yaml_dict["u"] == "None":
        u = None
    else:
        def u(x: torch.Tensor, backend):
            return eval(yaml_dict["u"])
    
    # Get the gradient of the analytical solution if it exists
    if yaml_dict["grad_u"] == "None":
        grad_u = None
    else:
        def grad_u(x: torch.Tensor, backend):
            return eval(yaml_dict["grad_u"])
    
    return construct_PDE_class(
        f=f,
        diffusion_matrix=diffusion_matrix,
        BC=BC,
        u=u,
        grad_u=grad_u,
        is_in_divergence_form=is_in_divergence_form,
    )
    
