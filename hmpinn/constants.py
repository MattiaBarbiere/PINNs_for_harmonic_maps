import os

# Get the absolute path
abs_path = os.path.abspath(__file__)

# Get the path to the hmpinn directory
hmpinn_dir = os.path.dirname(os.path.dirname(abs_path))

# Path to the folder where images will be saved
IMAGE_FOLDER_PATH = os.path.join(hmpinn_dir, 'report_images')


DEFAULT_CONFIG = {'defaults': ['_self_'], 
                    'PDE': {'name': 'diff',
                            'PDE_kwargs':{}
                            }, 
                    'model': {'type': 'v2',
                              'model_kwargs':{ 
                                'nodes_hidden_layers': [64, 64, 64, 64, 64, 64], 
                                'activation_function': 'gelu', 
                                'has_embedding_layer': False,
                                'embeddings_per_dim': None,
                                'output_dim': 1}
                                }, 
                    'train': {'batch_size': 128, 
                             'n_epochs': 15000, 
                             'optimizer': 'Adam', 
                             'optimizer_threshold': 7000, 
                             'loss_BC_weight': 20, 
                             'save_BC_loss': True, 
                             'boundary_batch_ratio': 1, 
                             'seed': 42, 
                             'interior_sampler': None, 
                             'boundary_sampler': None}, 
                    'solver': {'nx': 21, 
                               'ny': None, 
                               'p': 3}
                               
                }