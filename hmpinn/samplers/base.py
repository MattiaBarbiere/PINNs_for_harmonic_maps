import torch
from abc import ABC, abstractmethod

## For plotting
import seaborn as sns
import matplotlib.pyplot as plt
#Setting theme
sns.set_theme(style="white", palette=None)


class BaseSampler(ABC):
    def __init__(self, x_interval = (0.0, 1.0), y_interval = (0.0, 1.0), x_grid = None, y_grid = None, default_batch_size = 128, seed = None):
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
        self.x_interval = torch.tensor(x_interval, dtype=torch.float32)
        self.y_interval = torch.tensor(y_interval, dtype=torch.float32)
        self.default_batch_size = int(default_batch_size)
        self.seed = seed

        # Generate the grid
        if x_grid is None:
            x_grid = torch.tensor(x_interval, dtype=torch.float32)
        elif isinstance(x_grid, torch.Tensor):
            assert x_grid.dtype == torch.float32, "x_grid must be a float tensor"
        elif isinstance(x_grid, int):
            x_grid = torch.linspace(x_interval[0], x_interval[1], x_grid, dtype=torch.float32)
        elif isinstance(x_grid, list) or isinstance(x_grid, tuple):
            x_grid = torch.tensor(x_grid, dtype=torch.float32)
        else:
            raise ValueError("x_grid must be an int, list (of floats),tuple (of floats), or torch.tensor")
        
        if y_grid is None:
            y_grid = torch.tensor(y_interval, dtype=torch.float32)
        elif isinstance(y_grid, torch.Tensor):
            assert y_grid.dtype == torch.float32, "y_grid must be a float tensor"
        elif isinstance(y_grid, int):
            y_grid = torch.linspace(y_interval[0], y_interval[1], y_grid, dtype=torch.float32)
        elif isinstance(y_grid, list) or isinstance(y_grid, tuple):
            y_grid = torch.tensor(y_grid, dtype=torch.float32)
        else:
            raise ValueError("y_grid must be an int, list (of floats),tuple (of floats), or torch.tensor")

        
        # Some checks
        assert x_grid.shape[0] > 0, "x_grid must not be empty"
        assert y_grid.shape[0] > 0, "y_grid must not be empty"
        assert x_grid.ndim == 1, "x_grid must be a 1D tensor"
        assert y_grid.ndim == 1, "y_grid must be a 1D tensor"
        assert torch.all(x_grid >= x_interval[0]) and torch.all(x_grid <= x_interval[1]), "x_grid must be within x_interval"
        assert torch.all(y_grid >= y_interval[0]) and torch.all(y_grid <= y_interval[1]), "y_grid must be within y_interval"

        # Sort the tensors
        x_grid, _ = torch.sort(x_grid)
        y_grid, _ = torch.sort(y_grid)


        # Check that the grid starts and ends with the interval boundary
        if x_grid[0] != x_interval[0]:
            x_grid = torch.cat((torch.tensor([x_interval[0]]), x_grid))
        if x_grid[-1] != x_interval[1]:
            x_grid = torch.cat((x_grid, torch.tensor([x_interval[1]])))
        if y_grid[0] != y_interval[0]:
            y_grid = torch.cat((torch.tensor([y_interval[0]]), y_grid))
        if y_grid[-1] != y_interval[1]:
            y_grid = torch.cat((y_grid, torch.tensor([y_interval[1]])))

        # Store the grid
        self.x_grid = x_grid
        self.y_grid = y_grid

        # Store the grid size (we subtract 1 so that we only sample the bottom left corner of each grid cell)
        self.nx = x_grid.shape[0] - 1
        self.ny = y_grid.shape[0] - 1

        # Compute the length and height of each grid cell
        self.lengths, self.heights = self.compute_sizes()
        
        # Set the seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
    
    def compute_sizes(self):
        """
        Computes the length and height of the grid cells

        Returns
        (torch.tensor, torch.tensor): A tuple containing the length and height of the grid cells
        """
        
        dx = self.x_grid[1:] - self.x_grid[:-1]
        dy = self.y_grid[1:] - self.y_grid[:-1]
        return dx, dy
    
    def change_seed(self, new_seed):
        """
        Change the seed of the random number generator

        Parameters
        new_seed (int): The new seed to use. If None nothing happens
        """
        # Check that new seed is not None
        if new_seed is not None:
            self.seed = new_seed
            torch.manual_seed(new_seed)

    def change_default_batch_size(self, new_batch_size):
        """
        Change the batch size

        Parameters
        new_batch_size (int): The new batch size to use. If None nothing happens
        """
        # Check that new batch size is not None
        if new_batch_size is not None:
            self.batch_size = int(new_batch_size)
    
    @abstractmethod
    def sample_batch(self, batch_size = None, weighted = True, seed = None):
        """
        Sample points

        Parameters
        batch_size (int): The number of points to sample. If not None, this will override the batch size set in the constructor.
                            If None, the batch size set in the constructor will be used.
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor. 
                    Default is None (uses the seed set in the constructor)
        weighted (bool): If True, the points are sampled from the grid with a probability proportional to the area/length 
                            of the grid cells/intervals.

        Returns
        torch.tensor: A tensor of shape (batch_size, 2) containing the x and y coordinates of the sampled points
        """
        pass


    @abstractmethod
    def plot_grid(self, show_random_sample = False, weighted_sample = True, seed = None):
        """"
        Plots the grid and optionally samples from the grid

        Parameters
        show_random_sample (bool): If True, samples from the grid are shown. Default is False
        weighted_sample (bool): If True, the samples are drawn from the grid with a probability proportional to 
                                the area/length of the grid cels/intervals. Default is True
        seed (int): The seed to use for the random number generator. This will override the seed set in the constructor.
                    Default is None (uses the seed set in the constructor)
        """
        pass
