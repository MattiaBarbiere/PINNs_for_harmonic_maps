"""
YAML Utilities for Model Loading and Configuration.

This module provides utilities for loading trained models and configurations
from YAML files, supporting all model versions and PDE types.

Functions:
    read_yaml_file: Read and parse YAML configuration files.
    load_model_v0: Load ModelV0 with embedding layer support.
    load_model_v1: Load ModelV1 with mandatory embedding layer.
    load_model_v2: Load ModelV2 with configurable activation functions.
    load_model: Automatically detect and load any model version.
    load_PDE_ymal: Extract PDE parameters from YAML.
    load_solver_ymal: Extract solver parameters from YAML.
    load_train_ymal: Extract training parameters from YAML.
    load_model_ymal: Extract model parameters from YAML.
    PDE_from_yaml: Construct PDE class from YAML specification.
"""

import os
import yaml
import torch

from hmpinn.models import *
from hmpinn.utils.utils import get_PDE_class, get_PDE_object, organise_dict
from hmpinn.PDEs.PDE_factory import construct_PDE_class

def read_yaml_file(path):
    """
    Read and parse a YAML configuration file from a model directory.
    
    Parameters:
        path: str
            Path to the model directory containing .hydra/config.yaml.
    
    Returns:
        dict
            Parsed and organized configuration dictionary.
            
    Raises:
        yaml.YAMLError: If YAML file cannot be parsed.
    """
    # Get the absolute path to the config file
    path = os.path.abspath(path)
    config_path = os.path.join(path, ".hydra", "config.yaml")
    
    with open(config_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            raise
    
    # Organize the dictionary for standardized access
    return organise_dict(params)

def load_model_v0(path, backend=torch):
    """
    Load ModelV0 with optional embedding layer from saved files.

    Parameters:
        path: str
            Path to the model directory.
        backend: torch or np, optional
            Backend to use for the model. Defaults to torch.

    Returns:
        tuple
            Tuple containing (model, errors, grad_errors, loss, BC_loss).
    """
    # Read configuration parameters
    params = read_yaml_file(path)
    
    # Extract model-specific parameters
    embeddings_per_dim = params["model"]["model_kwargs"]['embeddings_per_dim']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']
    has_embedding_layer = params["model"]["model_kwargs"]['has_embedding_layer']

    # Get PDE object with specified backend
    PDE_obj = get_PDE_object(params, backend=backend)

    # Initialize and load the model
    model = ModelV0(PDE_obj, 
                     embedding_size_per_dim=embeddings_per_dim, 
                     has_embedding_layer=has_embedding_layer,
                     nodes_hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load training history
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Load boundary condition loss if available
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except FileNotFoundError:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss

def load_model_v1(path, backend=torch):
    """
    Load ModelV1 with mandatory embedding layer from saved files.

    Parameters:
        path: str
            Path to the model directory.
        backend: torch or np, optional
            Backend to use for the model. Defaults to torch.

    Returns:
        tuple
            Tuple containing (model, errors, grad_errors, loss, BC_loss).
    """
    # Read configuration parameters
    params = read_yaml_file(path)
    
    # Extract model-specific parameters
    embeddings_per_dim = params["model"]["model_kwargs"]['embeddings_per_dim']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']

    # Get PDE object with specified backend
    PDE_obj = get_PDE_object(params, backend=backend)

    # Initialize and load the model
    model = ModelV1(PDE_obj, 
                     embedding_size_per_dim=embeddings_per_dim,
                     nodes_hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load training history
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Load boundary condition loss if available
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss

def load_model_v2(path, backend=torch):
    """
    Load ModelV2 with configurable activation functions from saved files.

    Parameters:
        path: str
            Path to the model directory.
        backend: torch or np, optional
            Backend to use for the model. Defaults to torch.

    Returns:
        tuple
            Tuple containing (model, errors, grad_errors, loss, BC_loss).
    """
    # Read configuration parameters
    params = read_yaml_file(path)
    
    # Extract model-specific parameters
    activation_function = params["model"]["model_kwargs"]['activation_function']
    hidden_layers = params["model"]["model_kwargs"]['nodes_hidden_layers']
    output_dim = params["model"]["model_kwargs"]['output_dim']

    # Get PDE object with specified backend
    PDE_obj = get_PDE_object(params, backend=backend)

    # Initialize and load the model
    model = ModelV2(PDE_obj, 
                     nodes_hidden_layers=hidden_layers,
                     activation_function=activation_function,
                     output_dim=output_dim)
    model.load_state_dict(torch.load(path + "/model.pt", map_location=torch.device('cpu')))

    # Load training history
    errors = torch.load(path + "/errors.pt")
    grad_errors = torch.load(path + "/grad_errors.pt")
    loss = torch.load(path + "/loss.pt")

    # Load boundary condition loss if available
    try:
        BC_loss = torch.load(path + "/BC_loss.pt")
    except:
        BC_loss = None

    return model, errors, grad_errors, loss, BC_loss

def load_model(path, backend=torch):
    """
    Automatically detect and load any model version from saved files.

    Parameters:
        path: str
            Path to the model directory.
        backend: torch or np, optional
            Backend to use for the model. Defaults to torch.

    Returns:
        tuple
            Tuple containing (model, errors, grad_errors, loss, BC_loss).
            
    Raises:
        ValueError: If model type is not recognized.
    """
    # Read configuration to determine model type
    params = read_yaml_file(path)
    model_type = params["model"]["type"]

    # Dispatch to appropriate loader based on model type
    if model_type == "v0":
        return load_model_v0(path, backend=backend)
    elif model_type == "v1":
        return load_model_v1(path, backend=backend)
    elif model_type == "v2":
        return load_model_v2(path, backend=backend)
    else:
        raise ValueError(f"The model type '{model_type}' is not recognized. Supported types are: v0, v1, v2.")

def load_PDE_ymal(file_path: str):
    """
    Extract PDE parameters from YAML configuration file.

    Parameters:
        file_path: str
            Path to the YAML file.

    Returns:
        dict
            PDE configuration dictionary.
    """
    data = read_yaml_file(file_path)
    return data["PDE"]

def load_solver_ymal(file_path: str):
    """
    Extract solver parameters from YAML configuration file.

    Parameters:
        file_path: str
            Path to the YAML file.

    Returns:
        dict
            Solver configuration dictionary.
    """
    data = read_yaml_file(file_path)
    return data["solver"]

def load_train_ymal(file_path: str):
    """
    Extract training parameters from YAML configuration file.

    Parameters:
        file_path: str
            Path to the YAML file.

    Returns:
        dict
            Training configuration dictionary.
    """
    data = read_yaml_file(file_path)
    return data["train"]

def load_model_ymal(file_path: str):
    """
    Extract model parameters from YAML configuration file.

    Parameters:
        file_path: str
            Path to the YAML file.

    Returns:
        dict
            Model configuration dictionary.
    """
    data = read_yaml_file(file_path)
    return data["model"]
    
def PDE_from_yaml(yaml_dict: dict):
    """
    Construct PDE class from YAML dictionary specification.
    
    Creates a PDE class either from predefined implementations or by
    dynamically constructing one from mathematical expressions.

    Parameters:
        yaml_dict: dict
            YAML dictionary containing PDE parameters and mathematical expressions.

    Returns:
        class
            Constructed PDE class ready for instantiation.
    """
    # If the dict has a "name" key, use a predefined PDE class
    if "name" in yaml_dict:
        return get_PDE_class(yaml_dict["name"])

    # Otherwise, construct PDE dynamically from mathematical expressions
    is_in_divergence_form = yaml_dict["is_in_divergence_form"]

    # Define source term function from string expression
    def f(x: torch.Tensor, backend):
        return eval(yaml_dict["f"])
    
    # Define diffusion matrix function from string expression
    def diffusion_matrix(x: torch.Tensor, backend):
        return eval(yaml_dict["diffusion_matrix"])
    
    # Define boundary condition function from string expression
    def BC(x: torch.Tensor, backend):
        return eval(yaml_dict["BC"])
    
    # Define analytical solution if provided
    if yaml_dict["u"] == "None":
        u = None
    else:
        def u(x: torch.Tensor, backend):
            return eval(yaml_dict["u"])
    
    # Define gradient of analytical solution if provided
    if yaml_dict["grad_u"] == "None":
        grad_u = None
    else:
        def grad_u(x: torch.Tensor, backend):
            return eval(yaml_dict["grad_u"])
    
    # Construct and return the PDE class
    return construct_PDE_class(
        f=f,
        diffusion_matrix=diffusion_matrix,
        BC=BC,
        u=u,
        grad_u=grad_u,
        is_in_divergence_form=is_in_divergence_form,
    )

