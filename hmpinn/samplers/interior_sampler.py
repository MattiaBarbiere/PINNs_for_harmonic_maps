import torch
import itertools
from hmpinn.samplers.base import BaseSampler

## For plotting
import seaborn as sns
import matplotlib.pyplot as plt
#Setting theme
sns.set_theme(style="white", palette=None)

class InteriorSampler(BaseSampler):
    def __init__(self, x_interval = (0.0, 1.0), y_interval = (0.0, 1.0), x_grid = None, y_grid = None, default_batch_size=128, seed = None):
        """
        Parameters
        x_interval (tuple or list of floats): The interval of the x axis to sample from. Default is (0.0, 1.0)
        y_interval (tuple or list of floats): The interval of the y axis to sample from. Default is (0.0, 1.0)
        x_grid (tuple or list or int or 1D torch.tensor): The values of x that will be used to form a grid. 
                                                            If None, no grid along x will be used. 
                                                            Default is None
        y_grid (tuple or list or int or 1D torch.tensor): The values of y that will be used to form a grid. 
                                                            If None, no grid along y will be used. 
                                                            Default is None
        default_batch_size (int): The number of points to sample. Default is 128
        seed (int): The seed to use for the random number generator. Default is None
            
        """
        super().__init__(x_interval, y_interval, x_grid, y_grid, default_batch_size, seed)
        
        # Create the index grid
        index_grid = torch.stack(torch.meshgrid(torch.arange(self.nx), torch.arange(self.ny), indexing='ij'), dim=-1)
        index_grid = index_grid.reshape(-1, 2)
        self.index_grid = index_grid

        # Compute the areas of the grid cells
        self.areas = self.compute_areas()
    
    
    def compute_areas(self):
        """
        Computes the areas of all the grid cells

        Returns
        torch.tensor: A 1D tensor of shape (nx * ny) containing the areas of each grid cell
        """
        areas = torch.outer(self.lengths, self.heights).reshape(-1)
        return areas

    def sample_batch_separated_coords(self, batch_size=None, weighted = True, seed = None):
        """
        Sample points from the interior of the domain returning the x and y coordinates separately

        Parameters
        batch_size (int): The number of points to sample. If None, the batch size set in the constructor will be used.
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor. 
                    Default is None (uses the seed set in the constructor)
        weighted (bool): If True, the points are sampled from the grid with a probability proportional to the area of the grid cell.

        Returns
        (torch.tensor, torch.tensor): A tuple containing the x (batch_size,) and y (batch_size,) coordinates of the sampled points
        NOTE: These tensors are not differentiable
        
        """
        # Set the batch size to default if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        else:
            batch_size = int(batch_size)

        # Set the seed if provided
        self.change_seed(new_seed=seed)
        
        # Compute the probabilities to sample the grid cells
        if weighted:
            probabilities = self.areas / torch.sum(self.areas)
        else:
            probabilities = torch.ones(self.nx * self.ny) / (self.nx * self.ny)
        
        # Sample which grid cell to sample from
        index = torch.multinomial(probabilities, batch_size, replacement=True)

        grid_cell_indices = self.index_grid[index, :]
        
        # Get the corresponding (bottom left) corners of the grid cell that were sampled
        x_coordinate = self.x_grid[grid_cell_indices[:, 0]]
        y_coordinate = self.y_grid[grid_cell_indices[:, 1]]
        
        # Generate unifrom random numbers
        x_perturbation = torch.rand(batch_size) * self.lengths[grid_cell_indices[:, 0]]
        y_perturbation = torch.rand(batch_size) * self.heights[grid_cell_indices[:, 1]]

        # add requires grad
        x_coordinate = x_coordinate + x_perturbation
        y_coordinate = y_coordinate + y_perturbation
        return (x_coordinate, y_coordinate)
    
    def sample_batch(self, batch_size=None, weighted = True, seed = None):
        """
        Sample points from the interior of the domain

        Parameters
        batch_size (int): The number of points to sample. If None, the batch size set in the constructor will be used.
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor. 
                    Default is None (uses the seed set in the constructor)
        weighted (bool): If True, the points are sampled from the grid with a probability proportional to the area of the grid cell.

        Returns
        torch.tensor: A tensor of shape (batch_size, 2) containing the x and y coordinates of the sampled points
        """
        return torch.stack(self.sample_batch_separated_coords(batch_size, weighted, seed), dim=-1).requires_grad_(True)

    def plot_grid(self, show_random_sample = False, weighted_sample = True, seed = None):
        """
        Plots the grid and optionally samples from the grid

        Parameters
        show_random_sample (bool): If True, samples from the grid are shown. Default is False
        weighted_sample (bool): If True, the samples are drawn from the grid with a probability proportional to 
                                the area of the grid cell. Default is True
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor.
                    Default is None (uses the seed set in the constructor)
        """
        x_grid = self.x_grid.detach().numpy()
        y_grid = self.y_grid.detach().numpy()
        grid_points = list(itertools.product(x_grid, y_grid))
        plt.scatter([x[0] for x in grid_points], [x[1] for x in grid_points], color="k", s=2)
        plt.vlines(x_grid, ymin=self.y_interval[0], ymax=self.y_interval[1], color='k', linewidth=0.5, linestyles='--')
        plt.hlines(y_grid, xmin=self.x_interval[0], xmax=self.x_interval[1], color='k', linewidth=0.5, linestyles='--')
        
        if show_random_sample:
            samples = self.sample_batch(100, weighted=weighted_sample, seed = seed).detach().numpy()
            plt.scatter(samples[:, 0], samples[:, 1], color="r", s=2)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()