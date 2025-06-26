"""
Error Plotting Functions.

This module provides functions for visualizing training errors, model performance,
and comparisons between neural network solutions and reference solutions.

Functions:
    prepare_single_data: Validate data for plotting.
    prepare_data: Validate multiple data arrays for plotting.
    plot_errors_in_row: Plot multiple error curves in a row layout.
    plot_errors_in_column: Plot multiple error curves in a column layout.
    plot_errors_in_grid: Plot multiple error curves in a grid layout.
    plot_errors_from_data: Plot error curves from data arrays.
    plot_errors_from_path: Plot error curves from saved model.
    plot_model_vs_function_error: Plot error contour between model and function.
    plot_model_vs_analytical_error: Plot error contour against analytical solution.
    plot_model_vs_benchmark_error: Plot error contour against benchmark solver.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from hmpinn import IMAGE_FOLDER_PATH
from hmpinn.models import *
from hmpinn.utils.yaml_utils import load_model, read_yaml_file
from hmpinn.PDEs import *
from hmpinn.benchmark_solver import BenchmarkSolver
from hmpinn.plotting.utils import eval_model_and_function

# Global font size for all plots
GLOBAL_FONT_SIZE = 40
plt.rcParams.update({'font.size': GLOBAL_FONT_SIZE})

def prepare_single_data(data):
    """
    Validate single data array for plotting.
    
    Checks if data is suitable for plotting by ensuring it exists,
    is not empty, and contains valid (non-NaN, non-infinite) values.
    
    Parameters:
        data: array-like or None
            The data array to validate.
    
    Returns:
        array-like or None
            The data if valid for plotting, None otherwise.
    """
    if data is None or len(data) == 0:
        return None
    if np.all(np.isnan(data)) or np.all(np.isinf(data)):
        return None
    return data

def prepare_data(errors, grad_errors, loss, BC_loss):
    """
    Validate multiple data arrays for plotting.

    Parameters:
        errors: array-like or None
            The error values per epoch.
        grad_errors: array-like or None
            The gradient error values per epoch.
        loss: array-like or None
            The loss values per epoch.
        BC_loss: array-like or None
            The boundary condition loss values per epoch.
    
    Returns:
        tuple
            Tuple of validated data arrays (may contain None values).
    """
    return (prepare_single_data(errors), 
            prepare_single_data(grad_errors), 
            prepare_single_data(loss), 
            prepare_single_data(BC_loss))

def plot_errors_in_row(path_list, title_list=None, file_name=None):
    """
    Plot multiple error curves in a row layout.

    Parameters:
        path_list: list
            The list of paths of the models.
        title_list: list, optional
            The list of titles for each subplot.
        file_name: str, optional
            The name of the file to save the plot to (include extension). If None, the plot will not be saved.
            Note that the file extension should be included.
    
    Returns:
        None
            Displays the plot.
    """
    fig, axs = plt.subplots(1, len(path_list), figsize=(12, 5), sharex=True, sharey=True)

    for i in range(len(path_list)):
        model, errors, grad_errors, loss, BC_loss = load_model(path_list[i])
        errors, grad_errors, loss, BC_loss = prepare_data(errors, grad_errors, loss, BC_loss)
        if errors is not None:
            axs[i].plot(errors, label="Error", color='C0')
        if grad_errors is not None:
            axs[i].plot(grad_errors, label="Grad Error", color='C1')
        if loss is not None:
            axs[i].plot(loss, label="Loss", color='C2')
        if BC_loss is not None:
            axs[i].plot(BC_loss, label="BC Loss", color='C3')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Error")

        if title_list is not None:
            axs[i].set_title(title_list[i])

    if file_name is not None:
        fig.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_errors_in_column(path_list, title_list=None, file_name=None):
    """
    Plot multiple error curves in a column layout.

    Parameters:
        path_list: list
            The list of paths of the models.
        title_list: list, optional
            The list of titles for each subplot.
        file_name: str, optional
            The name of the file to save the plot to (include extension). If None, the plot will not be saved.
            Note that the file extension should be included.
    
    Returns:
        None
            Displays the plot.
    """
    fig, axs = plt.subplots(len(path_list), 1, figsize=(5, 12), sharex=True, sharey=True)

    for i in range(len(path_list)):
        model, errors, grad_errors, loss, BC_loss = load_model(path_list[i])
        errors, grad_errors, loss, BC_loss = prepare_data(errors, grad_errors, loss, BC_loss)
        if errors is not None:
            axs[i].plot(errors, label="Error", color='C0')
        if grad_errors is not None:
            axs[i].plot(grad_errors, label="Grad Error", color='C1')
        if loss is not None:
            axs[i].plot(loss, label="Loss", color='C2')
        if BC_loss is not None:
            axs[i].plot(BC_loss, label="BC Loss", color='C3')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Error")

        if title_list is not None:
            axs[i].set_title(title_list[i])

    if file_name is not None:
        fig.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)

    plt.show()
    

def plot_errors_in_grid(path_list, number_of_rows=3, number_of_columns=3, title_list=None, file_name=None):
    """
    Plot multiple error curves in a grid layout.

    Parameters:
        path_list: list
            The list of paths of the models.
        number_of_rows: int
            The number of rows in the grid.
        number_of_columns: int
            The number of columns in the grid.
        title_list: list, optional
            The list of titles for each subplot.
        file_name: str, optional
            The name of the file to save the plot to (include extension). If None, the plot will not be saved.
            Note that the file extension should be included.
    
    Returns:
        None
            Displays the plot.
    
    Raises:
        ValueError: If the number of paths does not match the grid size.
    """
    if len(path_list) != number_of_rows * number_of_columns:
        raise ValueError(f"The number of paths {len(path_list)} is not equal to the number of rows {number_of_rows} times the number of columns {number_of_columns}.")
    
    if number_of_rows == 1:
        return plot_errors_in_row(path_list, title_list, file_name=file_name)
    if number_of_columns == 1:
        return plot_errors_in_column(path_list, title_list, file_name=file_name)

    fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(12, 12), sharex=True, sharey=True)

    for j in range(number_of_rows):
        for i in range(number_of_columns):
            model, errors, grad_errors, loss, BC_loss = load_model(path_list[number_of_columns*j+i])
            errors, grad_errors, loss, BC_loss = prepare_data(errors, grad_errors, loss, BC_loss)
            if errors is not None:
                axs[j, i].plot(errors, label="Error", color='C0')
            if grad_errors is not None:
                axs[j, i].plot(grad_errors, label="Grad Error", color='C1')
            if loss is not None:
                axs[j, i].plot(loss, label="Loss", color='C2')
            if BC_loss is not None:
                axs[j, i].plot(BC_loss, label="BC Loss", color='C3')
            axs[j, i].legend()
            axs[j, i].set_yscale('log')
            axs[j, i].set_xlabel("Epoch")
            axs[j, i].set_ylabel("Error")

            if title_list is not None:
                axs[j, i].set_title(title_list[number_of_columns*j+i])

    if file_name is not None:
        fig.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_errors_from_data(errors, grad_errors, loss, BC_loss=None, title=None, file_name=None, with_legend=True):
    """
    Plot error curves from data arrays.
    
    Creates a log-scale plot of training errors including model error,
    gradient error, residual loss, and boundary condition loss.

    Parameters:
        errors: array-like or None
            The model error values per epoch.
        grad_errors: array-like or None
            The gradient error values per epoch.
        loss: array-like or None
            The residual loss values per epoch.
        BC_loss: array-like or None, optional
            The boundary condition loss values per epoch.
        title: str, optional
            Title for the plot.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        with_legend: bool, optional
            Whether to show legend. Defaults to True.
    """
    # Validate data arrays
    errors, grad_errors, loss, BC_loss = prepare_data(errors, grad_errors, loss, BC_loss)
    
    # Plot each available error type
    if errors is not None:
        plt.plot(errors, label="Error", color='C0')
    if grad_errors is not None:
        plt.plot(grad_errors, label="Grad Error", color='C1')
    if loss is not None:
        plt.plot(loss, label="Loss", color='C2')
    if BC_loss is not None:
        plt.plot(BC_loss, label="BC Loss", color='C3')
    
    # Configure plot appearance
    if with_legend:
        plt.legend()
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    if title is not None:
        plt.title(title)
    
    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_errors_from_path(path, title=None, file_name=None, with_legend=True):
    """
    Plot error curves from a saved model.
    
    Loads a saved model and plots its training error history.
    
    Parameters:
        path: str
            Path to the saved model directory.
        title: str, optional
            Title for the plot.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
        with_legend: bool, optional
            Whether to show legend. Defaults to True.
    """
    # Load model and training history
    model, errors, grad_errors, loss, BC_loss = load_model(path)
    plot_errors_from_data(errors, grad_errors, loss, BC_loss, title, file_name, with_legend=with_legend)

def plot_model_vs_function_error(model, func, title="", levels=100, resolution=100, file_name=None):
    """
    Plot error contour between model and reference function.
    
    Creates a logarithmic contour plot showing the absolute error
    between a neural network model and a reference function.

    Parameters:
        model: torch.nn.Module or str
            The trained model or path to saved model.
        func: callable
            The reference function to compare against.
        title: str, optional
            Title for the plot.
        levels: int, optional
            Number of contour levels. Defaults to 100.
        resolution: int, optional
            Grid resolution for evaluation. Defaults to 100.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
    """
    # Evaluate model and function on grid
    X, Y, F, U = eval_model_and_function(model, func, resolution=resolution)
    res = np.abs(F - U)

    # Set up logarithmic contour levels
    vmin, vmax = res.min(), res.max()
    if vmin <= 0:
        vmin = vmax * 1e-3  # Avoid log(0) issues

    levels_list = np.logspace(np.log10(vmin), np.log10(vmax), num=levels)

    # Create contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, res, cmap="YlOrRd", levels=levels_list, norm=LogNorm())
    cbar = plt.colorbar(contour)
    cbar.set_label('Absolute Error')
    cbar.formatter = FormatStrFormatter('%.1e')
    cbar.update_ticks()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_model_vs_analytical_error(model, title="", levels=100, resolution=100, print_PDE=False, file_name=None):
    """
    Plot error contour between model and analytical solution.
    
    Creates a logarithmic contour plot showing the absolute error
    between a neural network model and the analytical solution of its PDE.

    Parameters:
        model: torch.nn.Module, str, or list
            The model, path to model, or list containing path.
        title: str, optional
            Title for the plot.
        levels: int, optional
            Number of contour levels. Defaults to 100.
        resolution: int, optional
            Grid resolution for evaluation. Defaults to 100.
        print_PDE: bool, optional
            Whether to print PDE information. Defaults to False.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
            
    Raises:
        ValueError: If model format is not recognized or no analytical solution exists.
    """
    # Handle different model input formats
    if isinstance(model, str):
        model, _, _, _, _ = load_model(model)
    elif isinstance(model, list) and len(model) == 1:
        model, _, _, _, _ = load_model(model[0])
    elif isinstance(model, (ModelV0, ModelV1)):
        pass
    else:
        raise ValueError('The model format is not recognized')

    # Check for analytical solution availability
    if not model.PDE.has_solution:
        raise ValueError('The analytical solution does not exist for this PDE')

    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    
    plot_model_vs_function_error(model, func=model.PDE.u, title=title, levels=levels, 
                                 resolution=resolution, file_name=file_name)

def plot_model_vs_benchmark_error(path, title=None, levels=100, resolution=100, save_fig=False, 
                                   nx=None, ny=None, p=None, 
                                   print_PDE=False, file_name=None):
    """
    Plot error contour between model and benchmark solver.
    
    Creates a logarithmic contour plot showing the absolute error
    between a neural network model and a high-accuracy benchmark solver.

    Parameters:
        path: str
            Path to the saved model directory.
        title: str, optional
            Title for the plot.
        levels: int, optional
            Number of contour levels. Defaults to 100.
        resolution: int, optional
            Grid resolution for evaluation. Defaults to 100.
        save_fig: bool, optional
            Deprecated parameter (use file_name instead).
        nx: int, optional
            Benchmark solver grid points in x-direction.
        ny: int, optional
            Benchmark solver grid points in y-direction.
        p: int, optional
            Benchmark solver polynomial degree.
        print_PDE: bool, optional
            Whether to print PDE information. Defaults to False.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
    """
    # Load model parameters and create benchmark solver
    params = read_yaml_file(path)
    model, _, _, _, _ = load_model(path, backend=np)
    
    # Use provided parameters or defaults from config
    if nx is None:
        nx = params["solver"]["nx"]
    if ny is None:
        ny = params["solver"]["ny"]
    if p is None:
        p = params["solver"]["p"]
    
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    
    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    
    plot_model_vs_function_error(model, benchmark, title=title, resolution=resolution, 
                                levels=levels, file_name=file_name)