import torch
import os
from matplotlib import pyplot as plt
from hmpinn.constants import IMAGE_FOLDER_PATH

def construct_meshgrid(resolution=1001, grid_size_x=2, grid_size_y=2):
    """
    Constructs a meshgrid of points in the square [0, 1]x[0, 1]
    
    Parameters
    resolution: int
        The resolution of the meshgrid
    grid_size_x: int
        Number of vertical lines in the grid
    grid_size_y: int
        Number of horizontal lines in the grid
    """
    # Tensor that goes from 0 to 1
    tensor_0_to_1 = torch.linspace(0, 1, resolution)
    grid_x = torch.linspace(0, 1, grid_size_x)
    grid_y = torch.linspace(0, 1, grid_size_y)

    # Iterate over the grid points and add them to the mesh grid
    XY = torch.stack([0 * torch.ones(resolution), tensor_0_to_1], dim=-1)
    for i in range(1, grid_size_x):
        grid_line = torch.stack([grid_x[i] * torch.ones(resolution), tensor_0_to_1], dim=-1)
        XY = torch.cat([XY, grid_line], dim=0)

    for j in range(grid_size_y):
        grid_line = torch.stack([tensor_0_to_1, grid_y[j] * torch.ones(resolution)], dim=-1)
        XY = torch.cat([XY, grid_line], dim=0)

    return XY

def plot_square_grid(ax=None, resolution=1001, grid_size_x=5, grid_size_y=5, 
                     title="Square Grid", file_name=None, fig_size=(7, 7)):
    """
    Plots the grid of the square [0, 1]x[0, 1]
    
    Parameters
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, a new figure and axes will be created.
    resolution: int
        The resolution of the meshgrid
    grid_size_x: int
        Number of vertical lines in the grid
    grid_size_y: int
        Number of horizontal lines in the grid
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Get the mesh grid
    inputs = construct_meshgrid(resolution=resolution, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

    # Plot the square grid
    ax.plot(inputs[:, 0], inputs[:, 1], 'b.', markersize=1)
    ax.set_title(title)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    if ax is None:
        plt.show()

def plot_square_boundary(ax=None, resolution=1001, 
                     title="Square Grid", file_name=None, fig_size=(7, 7)):
    """
    Plots the boundary of the square [0, 1]x[0, 1]
    
    Parameters
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, a new figure and axes will be created.
    resolution: int
        The resolution of the meshgrid
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    """
    plot_square_grid(ax=ax, resolution=resolution, grid_size_x=2, grid_size_y=2,
                     title=title, file_name=file_name, fig_size=fig_size)

def plot_hm_grid(harmonic_map: callable, ax=None, resolution=1001, grid_size_x=5, grid_size_y=5, 
                 title="Harmonic Map Grid", file_name=None, fig_size=(7, 7), **hm_kwargs):
    """
    Plots the grid of the harmonic map
    
    Parameters
    harmonic_map: callable
        The harmonic map to be plotted
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, a new figure and axes will be created.
    resolution: int
        The resolution of the meshgrid
    grid_size_x: int
        Number of vertical lines in the grid
    grid_size_y: int
        Number of horizontal lines in the grid
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    hm_kwargs: dict
        Additional keyword arguments to pass to the harmonic map function
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Get the mesh grid
    inputs = construct_meshgrid(resolution=resolution, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

    # Evaluate the harmonic map on the square
    with torch.no_grad():
        result = harmonic_map(inputs, **hm_kwargs)

    # Plot the harmonic map grid
    ax.plot(result[:, 0], result[:, 1], "b.", markersize=1)
    ax.set_title(title)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    if ax is None:
        plt.show()

def plot_hm_boundary(harmonic_map: callable, ax=None, resolution=1001,
                    title="Harmonic Map Boundary", file_name=None, fig_size=(7, 7), **hm_kwargs):
    """
    Plots the boundary of the harmonic map

    Parameters
    harmonic_map: callable
        The harmonic map to be plotted
    ax: matplotlib.axes.Axes
        The axes to plot on. If None, a new figure and axes will be created.
    resolution: int
        The resolution of the meshgrid
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    hm_kwargs: dict
        Additional keyword arguments to pass to the harmonic map function
    """
    plot_hm_grid(harmonic_map=harmonic_map, ax=ax, resolution=resolution, grid_size_x=2, grid_size_y=2, title=title, 
                    file_name=file_name, fig_size=fig_size, **hm_kwargs)



def plot_grid_comparison(harmonic_map: callable, resolution=1001, grid_size_x=2, grid_size_y=2,
                         title="Harmonic Map Boundary", file_name=None, fig_size=(14, 7), **hm_kwargs):
    """
    Plots the grid of the square vs the grid of the harmonic map
    
    Parameters
    harmonic_map: callable
        The harmonic map to be plotted
    resolution: int
        The resolution of the plot
    grid_size_x: int
        Number of vertical lines in the grid
    grid_size_y: int
        Number of horizontal lines in the grid
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    hm_kwargs: dict
        Additional keyword arguments to pass to the harmonic map function
    """
    # Begin the plot
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    # Call the other function to plot the square grid and harmonic map grid
    plot_square_grid(ax=ax[0], resolution=resolution, title="Original Square", grid_size_x=grid_size_x, grid_size_y=grid_size_y)
    
    # Plot the harmonic map
    plot_hm_grid(harmonic_map=harmonic_map, ax=ax[1], resolution=resolution,
                 grid_size_x=grid_size_x, grid_size_y=grid_size_y, title=title, **hm_kwargs)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    plt.show()

def plot_boundary_comparison(harmonic_map: callable, resolution=1001,
                            title="Harmonic Map Boundary", file_name=None, fig_size=(14, 7), **hm_kwargs):
    """
    Plots the boundary of the square vs the boundary of the harmonic map
    
    Parameters
    harmonic_map: callable
        The harmonic map to be plotted
    resolution: int
        The resolution of the plot
    title: str
        The title of the plot
    file_name: str
        The name of the file to save the plot to (optional). If None, the plot will not be saved. 
        Note that the file extension should be included.
    fig_size: tuple
        The size of the figure
    hm_kwargs: dict
        Additional keyword arguments to pass to the harmonic map function
    """
    plot_grid_comparison(harmonic_map=harmonic_map, resolution=resolution, grid_size_x=2, grid_size_y=2,
                            title=title, file_name=file_name, fig_size=fig_size, **hm_kwargs)