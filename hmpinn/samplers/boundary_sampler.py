"""
Boundary Domain Point Sampler.

This module provides functionality for sampling points from the boundary of
rectangular domains using weighted grid-based sampling strategies.

Classes:
    BoundarySampler: Sampler for boundary domain points with edge-based weighting.
"""

import torch

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
    
    Attributes:
        edge_lengths: Lengths of each boundary edge for weighted sampling.
        boundary_points: Precomputed boundary points for efficient sampling.
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
        
        # Precompute boundary characteristics
        self.edge_lengths = self.compute_edge_lengths()
        self.boundary_points = self.generate_boundary_points()
    
    def compute_edge_lengths(self):
        """
        Compute the total length of each boundary edge.

        Returns:
            torch.Tensor
                1D tensor containing the length of each boundary edge.
        """
        # Calculate lengths of the four boundary edges
        left_right_length = self.y_interval[1] - self.y_interval[0]
        top_bottom_length = self.x_interval[1] - self.x_interval[0]
        
        # Return lengths for [left, right, bottom, top] edges
        return torch.tensor([left_right_length, left_right_length, top_bottom_length, top_bottom_length])

    def generate_boundary_points(self):
        """
        Generate all possible boundary points based on the grid.

        Returns:
            list
                List of tensors, each containing boundary points for one edge.
        """
        boundary_edges = []
        
        # Left edge (x = x_min)
        left_edge = torch.stack([
            torch.full((len(self.y_grid),), self.x_interval[0]),
            self.y_grid
        ], dim=1)
        boundary_edges.append(left_edge)
        
        # Right edge (x = x_max)
        right_edge = torch.stack([
            torch.full((len(self.y_grid),), self.x_interval[1]),
            self.y_grid
        ], dim=1)
        boundary_edges.append(right_edge)
        
        # Bottom edge (y = y_min)
        bottom_edge = torch.stack([
            self.x_grid,
            torch.full((len(self.x_grid),), self.y_interval[0])
        ], dim=1)
        boundary_edges.append(bottom_edge)
        
        # Top edge (y = y_max)
        top_edge = torch.stack([
            self.x_grid,
            torch.full((len(self.x_grid),), self.y_interval[1])
        ], dim=1)
        boundary_edges.append(top_edge)
        
        return boundary_edges
    
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
        
        # Compute sampling probabilities for each edge
        if weighted:
            probabilities = self.edge_lengths / torch.sum(self.edge_lengths)
        else:
            # Equal probability for each edge
            probabilities = torch.ones(4) / 4
        
        # Sample which edges to select points from
        edge_indices = torch.multinomial(probabilities, batch_size, replacement=True)
        
        # Collect sampled points
        sampled_points = []
        for edge_idx in edge_indices:
            # Get points from the selected edge
            edge_points = self.boundary_points[edge_idx.item()]
            
            # Randomly select a point from this edge
            point_idx = torch.randint(0, len(edge_points), (1,))
            sampled_points.append(edge_points[point_idx])
        
        # Stack all sampled points and enable gradient tracking
        return torch.cat(sampled_points, dim=0).requires_grad_(True)

    def plot_grid(self, show_random_sample=False, weighted_sample=True, seed=None):
        """
        Visualize the boundary sampling grid and optionally show sample points.

        Parameters
            show_random_sample bool
                If True, samples from the grid are shown. Default is False
            weighted_sample bool
                If True, the samples are drawn from the grid with a probability proportional to 
                                    the area of the grid cell. Default is True
            seed int
                The seed to use for the random number generator. This will override the seed set in the constructor.
                        Default is None (uses the seed set in the constructor)
        """
        x_grid = self.x_grid.detach().numpy()
        y_grid = self.y_grid.detach().numpy()
        plt.scatter(x_grid, [0.0] * len(x_grid), color="k", s=5, zorder=20)
        plt.scatter(x_grid, [1.0] * len(x_grid), color="k", s=5, zorder=20)
        plt.scatter([0.0] * len(y_grid), y_grid, color="k", s=5, zorder=20)
        plt.scatter([1.0] * len(y_grid), y_grid, color="k", s=5, zorder=20)
        plt.vlines([0.0, 1.0], ymin=self.y_interval[0], ymax=self.y_interval[1], color='k', linewidth=0.5, linestyles='--')
        plt.hlines([0.0, 1.0], xmin=self.x_interval[0], xmax=self.x_interval[1], color='k', linewidth=0.5, linestyles='--')
        
        if show_random_sample:
            samples = self.sample_batch(100, weighted=weighted_sample, seed = seed).detach().numpy()
            plt.scatter(samples[:, 0], samples[:, 1], color="r", s=5, zorder=10)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()