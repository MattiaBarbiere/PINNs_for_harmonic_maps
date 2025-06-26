"""
Base Sampler Class.

This module provides the abstract base class for all point sampling implementations
in the hmpinn library, defining common functionality for grid generation and sampling.

Classes:
    BaseSampler: Abstract base class for domain point samplers.
"""

import torch
from abc import ABC, abstractmethod

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting theme
sns.set_theme(style="white", palette=None)

class BaseSampler(ABC):
    """
    Abstract base class for domain point samplers.
    
    This class provides common functionality for sampling points from rectangular domains
    with customizable grid structures. It handles grid generation, size computation,
    and random seed management.
    
    Attributes:
        x_interval: Domain bounds in x-direction.
        y_interval: Domain bounds in y-direction.
        x_grid: Grid points in x-direction.
        y_grid: Grid points in y-direction.
        nx: Number of grid cells in x-direction.
        ny: Number of grid cells in y-direction.
        lengths: Widths of grid cells in x-direction.
        heights: Heights of grid cells in y-direction.
        default_batch_size: Default number of points to sample.
        seed: Random seed for reproducibility.
    """
    
    def __init__(self, x_interval=(0.0, 1.0), y_interval=(0.0, 1.0), x_grid=None, y_grid=None, default_batch_size=128, seed=None):
        """
        Initialize the base sampler with domain and grid specifications.

        Parameters:
            x_interval: tuple or list of floats, optional
                Domain bounds in x-direction. Defaults to (0.0, 1.0).
            y_interval: tuple or list of floats, optional
                Domain bounds in y-direction. Defaults to (0.0, 1.0).
            x_grid: int, list, tuple, or torch.Tensor, optional
                Grid specification in x-direction. If None, uses interval endpoints.
                If int, creates uniform grid with that many points.
                If list/tuple/tensor, uses those specific grid points. Defaults to None.
            y_grid: int, list, tuple, or torch.Tensor, optional
                Grid specification in y-direction. Same format as x_grid. Defaults to None.
            default_batch_size: int, optional
                Default number of points to sample per batch. Defaults to 128.
            seed: int, optional
                Random seed for reproducible sampling. Defaults to None.
        """
        # Store domain intervals as tensors
        self.x_interval = torch.tensor(x_interval, dtype=torch.float32)
        self.y_interval = torch.tensor(y_interval, dtype=torch.float32)
        self.default_batch_size = int(default_batch_size)
        self.seed = seed

        # Process x_grid specification
        if x_grid is None:
            x_grid = torch.tensor(x_interval, dtype=torch.float32)
        elif isinstance(x_grid, torch.Tensor):
            assert x_grid.dtype == torch.float32, "x_grid must be a float tensor"
        elif isinstance(x_grid, int):
            x_grid = torch.linspace(x_interval[0], x_interval[1], x_grid, dtype=torch.float32)
        elif isinstance(x_grid, (list, tuple)):
            x_grid = torch.tensor(x_grid, dtype=torch.float32)
        else:
            raise ValueError("x_grid must be an int, list (of floats), tuple (of floats), or torch.tensor")
        
        # Process y_grid specification
        if y_grid is None:
            y_grid = torch.tensor(y_interval, dtype=torch.float32)
        elif isinstance(y_grid, torch.Tensor):
            assert y_grid.dtype == torch.float32, "y_grid must be a float tensor"
        elif isinstance(y_grid, int):
            y_grid = torch.linspace(y_interval[0], y_interval[1], y_grid, dtype=torch.float32)
        elif isinstance(y_grid, (list, tuple)):
            y_grid = torch.tensor(y_grid, dtype=torch.float32)
        else:
            raise ValueError("y_grid must be an int, list (of floats), tuple (of floats), or torch.tensor")

        # Validate grid specifications
        assert x_grid.shape[0] > 0, "x_grid must not be empty"
        assert y_grid.shape[0] > 0, "y_grid must not be empty"
        assert x_grid.ndim == 1, "x_grid must be a 1D tensor"
        assert y_grid.ndim == 1, "y_grid must be a 1D tensor"
        assert torch.all(x_grid >= x_interval[0]) and torch.all(x_grid <= x_interval[1]), "x_grid must be within x_interval"
        assert torch.all(y_grid >= y_interval[0]) and torch.all(y_grid <= y_interval[1]), "y_grid must be within y_interval"

        # Sort grids for consistency
        x_grid, _ = torch.sort(x_grid)
        y_grid, _ = torch.sort(y_grid)

        # Ensure grids include interval boundaries
        if x_grid[0] != x_interval[0]:
            x_grid = torch.cat((torch.tensor([x_interval[0]]), x_grid))
        if x_grid[-1] != x_interval[1]:
            x_grid = torch.cat((x_grid, torch.tensor([x_interval[1]])))
        if y_grid[0] != y_interval[0]:
            y_grid = torch.cat((torch.tensor([y_interval[0]]), y_grid))
        if y_grid[-1] != y_interval[1]:
            y_grid = torch.cat((y_grid, torch.tensor([y_interval[1]])))

        # Store processed grids
        self.x_grid = x_grid
        self.y_grid = y_grid

        # Compute number of grid cells (subtract 1 for cell counting)
        self.nx = x_grid.shape[0] - 1
        self.ny = y_grid.shape[0] - 1

        # Compute grid cell dimensions
        self.lengths, self.heights = self.compute_sizes()
        
        # Set random seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)
    
    def compute_sizes(self):
        """
        Compute the width and height of each grid cell.

        Returns:
            tuple
                Tuple of (lengths, heights) where lengths are cell widths
                and heights are cell heights.
        """
        dx = self.x_grid[1:] - self.x_grid[:-1]
        dy = self.y_grid[1:] - self.y_grid[:-1]
        return dx, dy
    
    def change_seed(self, new_seed):
        """
        Update the random seed for sampling.

        Parameters:
            new_seed: int or None
                New random seed. If None, no change is made.
        """
        if new_seed is not None:
            self.seed = new_seed
            torch.manual_seed(new_seed)

    def change_default_batch_size(self, new_batch_size):
        """
        Update the default batch size for sampling.

        Parameters:
            new_batch_size: int or None
                New default batch size. If None, no change is made.
        """
        if new_batch_size is not None:
            self.default_batch_size = int(new_batch_size)
    
    @abstractmethod
    def sample_batch(self, batch_size=None, weighted=True, seed=None):
        """
        Sample a batch of points from the domain.
        
        Must be implemented by subclasses to define specific sampling behavior.

        Parameters:
            batch_size: int, optional
                Number of points to sample. If None, uses default_batch_size.
            weighted: bool, optional
                Whether to weight sampling by cell area/length. Defaults to True.
            seed: int, optional
                Random seed for this sampling operation. Overrides instance seed.

        Returns:
            torch.Tensor
                Sampled points of shape (batch_size, 2) with gradient tracking enabled.
        """
        pass

    @abstractmethod
    def plot_grid(self, show_random_sample=False, weighted_sample=True, seed=None):
        """
        Visualize the sampling grid and optionally show sample points.
        
        Must be implemented by subclasses to provide appropriate visualization.

        Parameters:
            show_random_sample: bool, optional
                Whether to overlay sample points on the grid. Defaults to False.
            weighted_sample: bool, optional
                Whether to use weighted sampling for displayed samples. Defaults to True.
            seed: int, optional
                Random seed for sample generation. Overrides instance seed.
        """
        pass
