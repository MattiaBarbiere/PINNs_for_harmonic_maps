"""
Constants and Default Configurations.

This module provides default configuration settings and path constants
for the hmpinn library, including default hyperparameters and file paths.

Constants:
    IMAGE_FOLDER_PATH: Default path for saving generated images.
    DEFAULT_CONFIG: Default configuration dictionary for experiments.
"""

import os

# Get the absolute path to this file
abs_path = os.path.abspath(__file__)

# Get the path to the hmpinn directory (parent of parent directory)
hmpinn_dir = os.path.dirname(os.path.dirname(abs_path))

# Path to the folder where images will be saved
IMAGE_FOLDER_PATH = os.path.join(hmpinn_dir, 'report_images')

# Default configuration dictionary for experiments
# This provides standard settings that can be overridden in specific experiments
DEFAULT_CONFIG = {
    'defaults': ['_self_'], 
    
    # PDE configuration
    'PDE': {
        'name': 'diff',           # Default PDE type (non-symmetric diffusion)
        'PDE_kwargs': {}          # Additional PDE-specific parameters
    }, 
    
    # Model configuration
    'model': {
        'type': 'v2',             # Model version (v0, v1, or v2)
        'model_kwargs': { 
            'nodes_hidden_layers': [64, 64, 64, 64, 64, 64],   # Hidden layer sizes
            'activation_function': 'gelu',                     # Activation function
            'has_embedding_layer': False,                      # Whether to use embedding layer
            'embeddings_per_dim': None,                        # Embedding dimensions
            'output_dim': 1                                    # Output dimension
        }
    }, 
    
    # Training configuration
    'train': {
        'batch_size': 128,              # Training batch size
        'n_epochs': 15000,              # Number of training epochs
        'optimizer': 'Adam',            # Optimizer type
        'optimizer_threshold': 7000,    # Epoch to switch to LBFGS
        'loss_BC_weight': 20,           # Boundary condition loss weight
        'save_BC_loss': True,           # Whether to save BC loss history
        'boundary_batch_ratio': 1,      # Ratio of boundary to interior points
        'seed': 42,                     # Random seed for reproducibility
        'interior_sampler': None,       # Custom interior point sampler
        'boundary_sampler': None        # Custom boundary point sampler
    }, 
    
    # Benchmark solver configuration
    'solver': {
        'nx': 21,       # Number of grid points in x-direction
        'ny': None,     # Number of grid points in y-direction (None = same as nx)
        'p': 3          # Polynomial degree for finite element method
    }
}