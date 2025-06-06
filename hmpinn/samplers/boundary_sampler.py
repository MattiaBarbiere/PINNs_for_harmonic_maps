from hmpinn.samplers.base import BaseSampler
import torch
import itertools

## For plotting
import seaborn as sns
import matplotlib.pyplot as plt
#Setting theme
sns.set_theme(style="white", palette=None)

class BoundarySampler(BaseSampler):
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
    
    def sample_batch(self, batch_size=None, weighted = True,  seed = None):
        """
        Sample points from the boundary of the domain

        Parameters
        batch_size (int): The number of points to sample. If None, the batch size set in the constructor will be used.
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor. 
                    Default is None (uses the seed set in the constructor)
        weighted (bool): If True, the points are sampled from the grid with a probability proportional to the size of the interval.

        Returns
        torch.tensor: A tensor of shape (batch_size, 2) containing the x and y coordinates of the sampled points
        """
        # Set the batch size to default if not provided
        if batch_size is None:
            batch_size = self.default_batch_size
        else:
            batch_size = int(batch_size)
        
        # Set the seed if provided
        self.change_seed(new_seed=seed)
        
        # Compute the probabilities to sample the intervals
        if weighted:
            probabilities_x = self.lengths / torch.sum(self.lengths)
            probabilities_y = self.heights / torch.sum(self.heights)
        else:
            probabilities_x = torch.ones(self.nx) / self.nx
            probabilities_y = torch.ones(self.ny) / self.ny
        
        # Choose number of points to projected onto x=0/x=1 and y=0/y=1
        batch_size_x = batch_size // 2
        batch_size_y = batch_size - batch_size_x

        # Sample the x coordinates
        x_indices = torch.multinomial(probabilities_x, batch_size_x, replacement=True)
        x_coordinates = self.x_grid[x_indices]

        # Sample the y coordinates
        y_indices = torch.multinomial(probabilities_y, batch_size_y, replacement=True)
        y_coordinates = self.y_grid[y_indices]

        # Generate unifrom random numbers and perturb the coordinates
        x_perturbation = torch.rand(batch_size_x) * self.lengths[x_indices]
        x_coordinates = x_coordinates + x_perturbation
        y_perturbation = torch.rand(batch_size_y) * self.heights[y_indices]
        y_coordinates = y_coordinates + y_perturbation

        # Generate tensor that projects the x_coordinates onto the y=y_interval[0] or y=y_interval[1] boundary with equal probability
        y_boundary = torch.randint(0, 2, (batch_size_x,), device=x_coordinates.device)
        y_boundary = self.y_interval[y_boundary]

        # Generate tensor that projects the y_coordinates onto the x=x_interval[0] or x=x_interval[1] boundary with equal probability
        x_boundary = torch.randint(0, 2, (batch_size_y,), device=y_coordinates.device)
        x_boundary = self.x_interval[x_boundary]

        # Project the x coordinates onto the y=0 or y=1 boundary and the y coordinates onto the x=0 or x=1 boundary
        x_coordinates = torch.stack((x_coordinates, y_boundary), dim=-1)
        y_coordinates = torch.stack((x_boundary, y_coordinates), dim=-1)

        # Concatenate the x and y coordinates
        result = torch.cat((x_coordinates, y_coordinates), dim=0)
        
        # Return a shuffled verison of the coordinates
        return result[torch.randperm(result.shape[0])].requires_grad_(True)

        
        
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