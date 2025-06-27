"""
Boundary Domain Point Sampler.

This module provides functionality for sampling points from the boundary of
rectangular domains using weighted grid-based sampling strategies.

Classes:
    BoundarySampler: Sampler for boundary domain points with edge-based weighting.
"""

import torch
import itertools

from hmpinn.samplers.base import BaseSampler

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# Set plotting theme
sns.set_theme(style="white", palette=None)

class BoundarySampler(BaseSampler):
    """
    Sampler for boundary points of rectangular domains.
    
    This class implements weighted sampling from boundary edges of the domain,
    where sampling probability can be proportional to edge length for better coverage
    of domains with non-uniform grid spacing.
    """
    
    def __init__(self, x_interval=(0.0, 1.0), y_interval=(0.0, 1.0), x_grid=None, y_grid=None, default_batch_size=128, seed=None):
        """
        Initialize the boundary domain sampler.

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
    
    def sample_batch(self, batch_size=None, weighted=True, seed=None):
        """
        Sample points from the boundary of the domain.

        Parameters:
            batch_size: int, optional
                Number of points to sample. If None, uses default_batch_size.
            weighted: bool, optional
                Whether to weight sampling by edge length. Defaults to True.
            seed: int, optional
                Random seed for this sampling operation. Overrides instance seed.

        Returns:
            torch.Tensor
                Tensor of shape (batch_size, 2) containing sampled boundary points
                with gradient tracking enabled.
        """
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.default_batch_size
        else:
            batch_size = int(batch_size)
        
        # Update seed if provided
        self.change_seed(new_seed=seed)
        
        # Compute sampling probabilities for each interval
        if weighted:
            probabilities_x = self.lengths / torch.sum(self.lengths)
            probabilities_y = self.heights / torch.sum(self.heights)
        else:
            probabilities_x = torch.ones(self.nx) / self.nx
            probabilities_y = torch.ones(self.ny) / self.ny
        
        # Allocate points between x and y boundaries
        batch_size_x = batch_size // 2
        batch_size_y = batch_size - batch_size_x

        # Sample x-direction intervals and add perturbations
        x_indices = torch.multinomial(probabilities_x, batch_size_x, replacement=True)
        x_coordinates = self.x_grid[x_indices]
        x_perturbation = torch.rand(batch_size_x) * self.lengths[x_indices]
        x_coordinates = x_coordinates + x_perturbation

        # Sample y-direction intervals and add perturbations
        y_indices = torch.multinomial(probabilities_y, batch_size_y, replacement=True)
        y_coordinates = self.y_grid[y_indices]
        y_perturbation = torch.rand(batch_size_y) * self.heights[y_indices]
        y_coordinates = y_coordinates + y_perturbation

        # Project x-coordinates onto top/bottom boundary
        y_boundary = torch.randint(0, 2, (batch_size_x,), device=x_coordinates.device)
        y_boundary = self.y_interval[y_boundary]

        # Project y-coordinates onto left/right boundary
        x_boundary = torch.randint(0, 2, (batch_size_y,), device=y_coordinates.device)
        x_boundary = self.x_interval[x_boundary]

        # Combine coordinates to form boundary points
        x_coordinates = torch.stack((x_coordinates, y_boundary), dim=-1)
        y_coordinates = torch.stack((x_boundary, y_coordinates), dim=-1)

        # Concatenate and shuffle the boundary points
        result = torch.cat((x_coordinates, y_coordinates), dim=0)
        
        return result[torch.randperm(result.shape[0])].requires_grad_(True)
        
    def plot_grid(self, show_random_sample=False, weighted_sample=True, seed=None):
        """
        Visualize the boundary sampling grid and optionally show sample points.

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
        
        # Plot boundary grid points
        plt.scatter(x_grid, [0.0] * len(x_grid), color="k", s=5, zorder=20)
        plt.scatter(x_grid, [1.0] * len(x_grid), color="k", s=5, zorder=20)
        plt.scatter([0.0] * len(y_grid), y_grid, color="k", s=5, zorder=20)
        plt.scatter([1.0] * len(y_grid), y_grid, color="k", s=5, zorder=20)
        
        # Add boundary lines
        plt.vlines([0.0, 1.0], ymin=self.y_interval[0], ymax=self.y_interval[1], color='k', linewidth=0.5, linestyles='--')
        plt.hlines([0.0, 1.0], xmin=self.x_interval[0], xmax=self.x_interval[1], color='k', linewidth=0.5, linestyles='--')
        
        # Optionally show random samples
        if show_random_sample:
            samples = self.sample_batch(100, weighted=weighted_sample, seed=seed).detach().numpy()
            plt.scatter(samples[:, 0], samples[:, 1], color="r", s=5, zorder=10)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()