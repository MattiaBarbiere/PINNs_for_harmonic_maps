"""
Harmonic Map Plotting Functions.

This module provides specialized plotting functions for visualizing harmonic maps,
including grid transformations and boundary comparisons between the unit square
and the mapped domain.

Functions:
    construct_meshgrid: Create grid points for visualization.
    plot_square_grid: Plot grid lines on the unit square.
    plot_square_boundary: Plot boundary of the unit square.
    plot_hm_grid: Plot transformed grid under harmonic map.
    plot_hm_boundary: Plot transformed boundary under harmonic map.
    plot_grid_comparison: Compare original and transformed grids.
    plot_boundary_comparison: Compare original and transformed boundaries.
"""

import os
import torch
from matplotlib import pyplot as plt

from hmpinn.constants import IMAGE_FOLDER_PATH

def construct_meshgrid(resolution=1001, grid_size_x=2, grid_size_y=2):
    """
    Construct a meshgrid of points for visualization of grid lines.
    
    Creates vertical and horizontal grid lines on the unit square [0,1]x[0,1]
    for visualizing deformations under harmonic maps.
    
    Parameters:
        resolution: int, optional
            Number of points along each grid line. Defaults to 1001.
        grid_size_x: int, optional
            Number of vertical grid lines. Defaults to 2.
        grid_size_y: int, optional
            Number of horizontal grid lines. Defaults to 2.
            
    Returns:
        torch.Tensor
            Grid points of shape (num_points, 2) representing all grid lines.
    """
    # Create coordinate arrays
    tensor_0_to_1 = torch.linspace(0, 1, resolution)
    grid_x = torch.linspace(0, 1, grid_size_x)
    grid_y = torch.linspace(0, 1, grid_size_y)

    # Start with leftmost vertical line
    XY = torch.stack([torch.zeros(resolution), tensor_0_to_1], dim=-1)
    
    # Add remaining vertical lines
    for i in range(1, grid_size_x):
        grid_line = torch.stack([grid_x[i] * torch.ones(resolution), tensor_0_to_1], dim=-1)
        XY = torch.cat([XY, grid_line], dim=0)

    # Add horizontal lines
    for j in range(grid_size_y):
        grid_line = torch.stack([tensor_0_to_1, grid_y[j] * torch.ones(resolution)], dim=-1)
        XY = torch.cat([XY, grid_line], dim=0)

    return XY

def plot_square_grid(ax=None, resolution=1001, grid_size_x=5, grid_size_y=5, 
                     title="Square Grid", file_name=None, fig_size=(7, 7)):
    """
    Plot grid lines on the unit square domain.
    
    Visualizes the regular grid structure of the unit square [0,1]x[0,1]
    before transformation by a harmonic map.
    
    Parameters:
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        resolution: int, optional
            Number of points per grid line. Defaults to 1001.
        grid_size_x: int, optional
            Number of vertical grid lines. Defaults to 5.
        grid_size_y: int, optional
            Number of horizontal grid lines. Defaults to 5.
        title: str, optional
            Title for the plot. Defaults to "Square Grid".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (7, 7).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Generate grid points
    inputs = construct_meshgrid(resolution=resolution, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

    # Plot the grid as scattered points
    ax.plot(inputs[:, 0], inputs[:, 1], 'b.', markersize=1)
    ax.set_title(title)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    if ax is None:
        plt.show()

def plot_square_boundary(ax=None, resolution=1001, 
                     title="Square Boundary", file_name=None, fig_size=(7, 7)):
    """
    Plot the boundary of the unit square domain.
    
    Visualizes only the boundary edges of the unit square [0,1]x[0,1]
    before transformation by a harmonic map.
    
    Parameters:
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        resolution: int, optional
            Number of points per boundary edge. Defaults to 1001.
        title: str, optional
            Title for the plot. Defaults to "Square Boundary".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (7, 7).
    """
    plot_square_grid(ax=ax, resolution=resolution, grid_size_x=2, grid_size_y=2,
                     title=title, file_name=file_name, fig_size=fig_size)

def plot_hm_grid(harmonic_map: callable, ax=None, resolution=1001, grid_size_x=5, grid_size_y=5, 
                 title="Harmonic Map Grid", file_name=None, fig_size=(7, 7), **hm_kwargs):
    """
    Plot grid lines transformed by a harmonic map.
    
    Visualizes how the regular grid of the unit square is deformed
    when transformed by the given harmonic map function.
    
    Parameters:
        harmonic_map: callable
            Function that maps points from unit square to target domain.
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        resolution: int, optional
            Number of points per grid line. Defaults to 1001.
        grid_size_x: int, optional
            Number of vertical grid lines. Defaults to 5.
        grid_size_y: int, optional
            Number of horizontal grid lines. Defaults to 5.
        title: str, optional
            Title for the plot. Defaults to "Harmonic Map Grid".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (7, 7).
        **hm_kwargs: dict
            Additional keyword arguments passed to harmonic_map function.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Generate original grid points
    inputs = construct_meshgrid(resolution=resolution, grid_size_x=grid_size_x, grid_size_y=grid_size_y)

    # Transform grid points through harmonic map
    with torch.no_grad():
        result = harmonic_map(inputs, **hm_kwargs)

    # Plot the transformed grid
    ax.plot(result[:, 0], result[:, 1], "b.", markersize=1)
    ax.set_title(title)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    if ax is None:
        plt.show()

def plot_hm_boundary(harmonic_map: callable, ax=None, resolution=1001,
                    title="Harmonic Map Boundary", file_name=None, fig_size=(7, 7), **hm_kwargs):
    """
    Plot boundary transformed by a harmonic map.
    
    Visualizes how the boundary of the unit square is deformed
    when transformed by the given harmonic map function.

    Parameters:
        harmonic_map: callable
            Function that maps points from unit square to target domain.
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        resolution: int, optional
            Number of points per boundary edge. Defaults to 1001.
        title: str, optional
            Title for the plot. Defaults to "Harmonic Map Boundary".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (7, 7).
        **hm_kwargs: dict
            Additional keyword arguments passed to harmonic_map function.
    """
    plot_hm_grid(harmonic_map=harmonic_map, ax=ax, resolution=resolution, grid_size_x=2, grid_size_y=2, 
                 title=title, file_name=file_name, fig_size=fig_size, **hm_kwargs)

def plot_grid_comparison(harmonic_map: callable, resolution=1001, grid_size_x=2, grid_size_y=2,
                         title="Harmonic Map Boundary", file_name=None, fig_size=(14, 7), **hm_kwargs):
    """
    Compare original square grid with transformed grid side by side.
    
    Creates a side-by-side comparison showing the original unit square grid
    and its transformation under the harmonic map.
    
    Parameters:
        harmonic_map: callable
            Function that maps points from unit square to target domain.
        resolution: int, optional
            Number of points per grid line. Defaults to 1001.
        grid_size_x: int, optional
            Number of vertical grid lines. Defaults to 2.
        grid_size_y: int, optional
            Number of horizontal grid lines. Defaults to 2.
        title: str, optional
            Title for the transformed grid plot. Defaults to "Harmonic Map Transformation".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (14, 7).
        **hm_kwargs: dict
            Additional keyword arguments passed to harmonic_map function.
    """
    # Create subplot layout
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    # Plot original square grid and transformed grid
    plot_square_grid(ax=ax[0], resolution=resolution, title="Original Square", 
                    grid_size_x=grid_size_x, grid_size_y=grid_size_y)
    
    plot_hm_grid(harmonic_map=harmonic_map, ax=ax[1], resolution=resolution,
                 grid_size_x=grid_size_x, grid_size_y=grid_size_y, title=title, **hm_kwargs)

    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), dpi=300, bbox_inches='tight')

    plt.show()

def plot_boundary_comparison(harmonic_map: callable, resolution=1001,
                            title="Harmonic Map Boundary", file_name=None, fig_size=(14, 7), **hm_kwargs):
    """
    Compare original square boundary with transformed boundary side by side.
    
    Creates a side-by-side comparison showing the original unit square boundary
    and its transformation under the harmonic map.
    
    Parameters:
        harmonic_map: callable
            Function that maps points from unit square to target domain.
        resolution: int, optional
            Number of points per boundary edge. Defaults to 1001.
        title: str, optional
            Title for the transformed boundary plot. Defaults to "Harmonic Map Boundary".
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        fig_size: tuple, optional
            Figure size as (width, height). Defaults to (14, 7).
        **hm_kwargs: dict
            Additional keyword arguments passed to harmonic_map function.
    """
    plot_grid_comparison(harmonic_map=harmonic_map, resolution=resolution, grid_size_x=2, grid_size_y=2,
                         title=title, file_name=file_name, fig_size=fig_size, **hm_kwargs)