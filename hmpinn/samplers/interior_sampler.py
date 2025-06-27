"""
Interior Domain Point Sampler.

This module provides functionality for sampling points from the interior of
rectangular domains using weighted grid-based sampling strategies.

Classes:
    InteriorSampler: Sampler for interior domain points with grid-based weighting.
"""

import torch
import itertools

from hmpinn.samplers.base import BaseSampler

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting theme
sns.set_theme(style="white", palette=None)

class InteriorSampler(BaseSampler):
    """
    Sampler for interior points of rectangular domains.
    
    This class implements weighted sampling from grid cells within the domain interior,
    where sampling probability can be proportional to cell area for better coverage
    of domains with non-uniform grid spacing.
    
    Attributes:
        index_grid: Grid cell indices for efficient sampling.
        areas: Areas of each grid cell for weighted sampling.
    """
    
    def __init__(self, x_interval=(0.0, 1.0), y_interval=(0.0, 1.0), x_grid=None, y_grid=None, default_batch_size=128, seed=None):
        """
        Initialize the interior domain sampler.

        Parameters:
            x_interval: tuple or list of floats, optional
                Domain bounds in x-direction. Defaults to (0.0, 1.0).
            y_interval: tuple or list of floats, optional
                Domain bounds in y-direction. Defaults to (0.0, 1.0).
            x_grid: int, list, tuple, or torch.Tensor, optional
                Grid specification in x-direction. If None, uses interval endpoints. Defaults to None.
            y_grid: int, list, tuple, or torch.Tensor, optional
                Grid specification in y-direction. If None, uses interval endpoints. Defaults to None.
            default_batch_size: int, optional
                Default number of points to sample per batch. Defaults to 128.
            seed: int, optional
                Random seed for reproducible sampling. Defaults to None.
        """
        super().__init__(x_interval, y_interval, x_grid, y_grid, default_batch_size, seed)
        
        # Create index grid for efficient cell selection
        index_grid = torch.stack(torch.meshgrid(torch.arange(self.nx), torch.arange(self.ny), indexing='ij'), dim=-1)
        self.index_grid = index_grid.reshape(-1, 2)

        # Precompute cell areas for weighted sampling
        self.areas = self.compute_areas()
    
    def compute_areas(self):
        """
        Compute the area of each grid cell.

        Returns:
            torch.Tensor
                1D tensor of shape (nx * ny) containing the area of each grid cell.
        """
        areas = torch.outer(self.lengths, self.heights).reshape(-1)
        return areas

    def sample_batch_separated_coords(self, batch_size=None, weighted=True, seed=None):
        """
        Sample interior points returning x and y coordinates separately.
        
        Useful for applications that need separate coordinate handling.

        Parameters:
            batch_size: int, optional
                Number of points to sample. If None, uses default_batch_size.
            weighted: bool, optional
                Whether to weight sampling by cell area. Defaults to True.
            seed: int, optional
                Random seed for this sampling operation. Overrides instance seed.

        Returns:
            tuple
                Tuple of (x_coordinates, y_coordinates) tensors of shape (batch_size,).
                Note: These tensors do not have gradient tracking enabled.
        """
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size
        else:
            batch_size = int(batch_size)

        # Update seed if provided
        self.change_seed(new_seed=seed)
        
        # Compute sampling probabilities
        if weighted:
            probabilities = self.areas / torch.sum(self.areas)
        else:
            probabilities = torch.ones(self.nx * self.ny) / (self.nx * self.ny)
        
        # Sample grid cells based on probabilities
        index = torch.multinomial(probabilities, batch_size, replacement=True)
        grid_cell_indices = self.index_grid[index, :]
        
        # Get bottom-left corners of selected grid cells
        x_coordinate = self.x_grid[grid_cell_indices[:, 0]]
        y_coordinate = self.y_grid[grid_cell_indices[:, 1]]
        
        # Add random perturbations within each cell
        x_perturbation = torch.rand(batch_size) * self.lengths[grid_cell_indices[:, 0]]
        y_perturbation = torch.rand(batch_size) * self.heights[grid_cell_indices[:, 1]]

        # Combine coordinates and perturbations
        x_coordinate = x_coordinate + x_perturbation
        y_coordinate = y_coordinate + y_perturbation
        
        return (x_coordinate, y_coordinate)
    
    def sample_batch(self, batch_size=None, weighted=True, seed=None):
        """
        Sample points from the interior of the domain.

        Parameters:
            batch_size: int, optional
                Number of points to sample. If None, uses default_batch_size.
            weighted: bool, optional
                Whether to weight sampling by cell area. Defaults to True.
            seed: int, optional
                Random seed for this sampling operation. Overrides instance seed.

        Returns:
            torch.Tensor
                Tensor of shape (batch_size, 2) containing sampled interior points
                with gradient tracking enabled.
        """
        # Get separated coordinates and combine them
        x_coords, y_coords = self.sample_batch_separated_coords(batch_size, weighted, seed)
        return torch.stack((x_coords, y_coords), dim=-1).requires_grad_(True)

    def plot_grid(self, show_random_sample=False, weighted_sample=True, seed=None):
        """
        Visualize the sampling grid and optionally show sample points.

        Parameters:
            show_random_sample: bool, optional
                Whether to overlay sample points on the grid. Defaults to False.
            weighted_sample: bool, optional
                Whether to use weighted sampling for displayed samples. Defaults to True.
            seed: int, optional
                Random seed for sample generation. Overrides instance seed.
        """
        # Convert grids to numpy for plotting
        x_grid = self.x_grid.detach().numpy()
        y_grid = self.y_grid.detach().numpy()
        
        # Plot grid points and lines
        grid_points = list(itertools.product(x_grid, y_grid))
        plt.scatter([x[0] for x in grid_points], [x[1] for x in grid_points], color="k", s=2)
        plt.vlines(x_grid, ymin=self.y_interval[0], ymax=self.y_interval[1], color='k', linewidth=0.5, linestyles='--')
        plt.hlines(y_grid, xmin=self.x_interval[0], xmax=self.x_interval[1], color='k', linewidth=0.5, linestyles='--')
        
        # Optionally show random samples
        if show_random_sample:
            samples = self.sample_batch(100, weighted=weighted_sample, seed=seed).detach().numpy()
            plt.scatter(samples[:, 0], samples[:, 1], color="r", s=2)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()