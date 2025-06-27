"""
General Utilities for hmpinn Library.

This module provides utility functions for PDE and model management,
configuration processing, and data organization.

Functions:
    get_PDE_class: Get PDE class by name.
    get_PDE_object: Create PDE object from configuration.
    get_model_class: Get model class by type.
    flatten_dict: Flatten nested dictionaries.
    organise_dict: Organize configuration dictionaries.
"""

import copy 

from hmpinn.PDEs import * 
from hmpinn.models import * 
from hmpinn import DEFAULT_CONFIG
from hmpinn.PDEs import PDE_NAME_TO_CLASS

def get_PDE_class(poisson_equation: str):
    """
    Get the PDE class corresponding to a given name.

    Parameters:
        poisson_equation: str
            The name of the PDE.

    Returns:
        class
            The PDE class corresponding to the given name.
            
    Raises:
        ValueError: If the PDE name is not recognized.
    """
    if poisson_equation not in PDE_NAME_TO_CLASS:
        raise ValueError(f"The PDE {poisson_equation} is not recognized")
    
    return PDE_NAME_TO_CLASS[poisson_equation]

def get_PDE_object(params: dict, backend) -> object:
    """
    Create a PDE object from configuration parameters.

    Parameters:
        params: dict
            Configuration parameters dictionary (see DEFAULT_CONFIG for structure).
        backend: torch or np
            Backend to use for the PDE (torch or numpy).

    Returns:
        object
            Initialized PDE object with specified parameters.
    """
    # Work with a copy to avoid modifying the original
    params = params.copy()

    # Extract PDE configuration
    pde_params = params["PDE"]
    pde_kwargs = pde_params["PDE_kwargs"]

    # Add backend to the parameters
    pde_kwargs["backend"] = backend
    
    # Get the PDE class and instantiate it
    name = pde_params["name"]
    pde_class = get_PDE_class(name)

    return pde_class(**pde_kwargs)

def get_model_class(model_type: str):
    """
    Get the model class corresponding to a given type.

    Parameters:
        model_type: str
            The type/version of the model (v0, v1, or v2).

    Returns:
        class
            The model class corresponding to the given type.
            
    Raises:
        ValueError: If the model type is not recognized.
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
    Flatten a nested dictionary into a single-level dictionary.

    Parameters:
        d: dict
            The nested dictionary to flatten.

    Returns:
        dict
            Flattened dictionary with all nested keys at the top level.
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)

def organise_dict(d: dict) -> dict:
    """
    Organize configuration dictionary into standardized structure.
    
    This function restructures configuration dictionaries to match the standard
    format defined in DEFAULT_CONFIG. It handles various parameter names and
    groupings for backward compatibility with older experiment configurations.

    Parameters:
        d: dict
            Dictionary to be organized.

    Returns:
        dict
            Organized dictionary following the standard structure.
            
    Raises:
        ValueError: If unrecognized keys are found or invalid parameter combinations.
    """    
    # Flatten the dictionary to simplify processing
    d = flatten_dict(d)

    # Start with a deep copy of the default configuration
    res_dict = copy.deepcopy(DEFAULT_CONFIG)

    # Process each key in the input dictionary
    for key in d.keys():
        # PDE-related keys
        if key in ["name", "poisson_equation"]:
            res_dict["PDE"]["name"] = d[key]
        
        elif key in ["a", "b", "amplitude"]:
            # Parameters specific to eigenfunction PDEs
            if res_dict["PDE"]["name"] in ["eigenfunc", "eigenfunc_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the eigenfunction PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
        
        elif key == "const_value":
            # Parameter specific to constant source PDEs
            if res_dict["PDE"]["name"] in ["const_source", "const_source_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key 'const_value' is only valid for the constant source PDE. Input was 'const_value': {d[key]} for {res_dict['PDE']['name']}")
        
        elif key in ["mu_x", "mu_y", "std_x", "std_y"]:
            # Parameters specific to Gaussian bump PDEs
            if res_dict["PDE"]["name"] in ["gaussian_bump_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the gaussian bump PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key == "K":
            # Parameter specific to convection-dominated PDEs
            if res_dict["PDE"]["name"] in ["convection_dominated", "convection_dominated_NonDF"]:
                res_dict["PDE"]["PDE_kwargs"]["K"] = d[key]
            else:
                raise ValueError(f"The key 'K' is only valid for the convection dominated PDE. Input was 'K': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key in ["curvature", "frequency_x", "frequency_y"]:
            # Parameters specific to sinusoidal boundary harmonic maps
            if res_dict["PDE"]["name"] in ["sin_boundaries_hm"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the sin boundaries harmonic map PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")
            
        elif key in ["a_left", "a_right", "b_top", "b_bottom", "degree"]:
            # Parameters specific to polynomial boundary harmonic maps
            if res_dict["PDE"]["name"] in ["poly_boundaries_hm"]:
                res_dict["PDE"]["PDE_kwargs"][key] = d[key]
            else:
                raise ValueError(f"The key '{key}' is only valid for the polynomial boundaries harmonic map PDE. Input was '{key}': {d[key]} for {res_dict['PDE']['name']}")

        # Model-related keys
        elif key == "type":
            res_dict["model"]["type"] = d[key]
        elif key == "activation_function":
            res_dict["model"]["model_kwargs"]["activation_function"] = d[key]
        elif key in ["hidden_layers", "nodes_hidden_layers"]:
            res_dict["model"]["model_kwargs"]["nodes_hidden_layers"] = d[key]
        elif key == "embeddings_per_dim":
            res_dict["model"]["model_kwargs"]["embeddings_per_dim"] = d[key]
        elif key in ["embedding_layer", "has_embedding_layer"]:
            res_dict["model"]["model_kwargs"]["has_embedding_layer"] = d[key]
        elif key == "output_dim":
            res_dict["model"]["model_kwargs"]["output_dim"] = d[key]
        
        # Training-related keys
        elif key in ["n_epochs", "epochs"]:
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
        
        # Solver-related keys
        elif key == "nx":
            res_dict["solver"]["nx"] = d[key]
        elif key == "ny":
            res_dict["solver"]["ny"] = d[key]
        elif key == "p":
            res_dict["solver"]["p"] = d[key]
        
        # Legacy key for backward compatibility
        elif key == "numb_batches":
            res_dict["train"]["numb_batches"] = d[key]
        else:
            # Unrecognized key
            raise ValueError(f"The key {key} was not added to the organised dict")
        
    return res_dict