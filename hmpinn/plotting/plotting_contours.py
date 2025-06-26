"""
Contour Plotting Functions.

This module provides functions for creating contour plots of neural network solutions,
benchmark solutions, and their comparisons. Supports both individual and comparative
visualizations.

Functions:
    plot_single_contour: Plot a single contour plot.
    plot_model_separate: Plot model solution in separate figure.
    plot_benchmark_separate: Plot benchmark solution in separate figure.
    plot_model_and_benchmark_separate: Plot model and benchmark in separate figures.
    plot_model_vs_function_contour: Compare model and function side by side.
    plot_model_vs_analytical_contour: Compare model and analytical solution.
    plot_model_vs_benchmark_contour: Compare model and benchmark solver.
    compare_model_to_benchmark: Comprehensive model-benchmark comparison.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from hmpinn import IMAGE_FOLDER_PATH
from hmpinn.models import *
from hmpinn.utils.yaml_utils import load_model, read_yaml_file
from hmpinn.plotting.utils import eval_model_and_function
from hmpinn.benchmark_solver import BenchmarkSolver
from hmpinn.plotting.plotting_errors import plot_model_vs_benchmark_error

def plot_single_contour(X, Y, Z, title="Contour Plot", levels=100, figsize=(8, 6), file_name=None):
    """
    Create a single contour plot from grid data.
    
    Parameters:
        X: numpy.ndarray
            X-coordinate grid.
        Y: numpy.ndarray  
            Y-coordinate grid.
        Z: numpy.ndarray
            Values at grid points.
        title: str, optional
            Plot title. Defaults to "Contour Plot".
        levels: int, optional
            Number of contour levels. Defaults to 100.
        figsize: tuple, optional
            Figure size as (width, height). Defaults to (8, 6).
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
    """
    plt.figure(figsize=figsize)
    contour = plt.contourf(X, Y, Z, cmap='viridis', levels=levels)
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_model_separate(path, title="Model(x, y)", levels=100, resolution=100, figsize=(8, 6), file_name=None):
    """
    Plot neural network model solution in a separate figure.
    
    Parameters:
        path: str
            Path to the saved model directory.
        title: str, optional
            Plot title. Defaults to "Model(x, y)".
        levels: int, optional
            Number of contour levels. Defaults to 100.
        resolution: int, optional
            Grid resolution for evaluation. Defaults to 100.
        figsize: tuple, optional
            Figure size as (width, height). Defaults to (8, 6).
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
    """
    # Load model and create benchmark for evaluation interface
    model, _, _, _, _ = load_model(path, backend=np)
    
    params = read_yaml_file(path)
    nx = params["solver"]["nx"]
    ny = params["solver"]["ny"]
    p = params["solver"]["p"]
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    
    # Evaluate model on grid
    X, Y, F_model, _ = eval_model_and_function(model, benchmark, resolution=resolution)
    
    plot_single_contour(X, Y, F_model, title=title, levels=levels, figsize=figsize, file_name=file_name)

def plot_benchmark_separate(path, title="Benchmark Solver", levels=100, resolution=100, figsize=(8, 6), 
                           nx=None, ny=None, p=None, file_name=None):
    """
    Plot only the benchmark solver in a separate figure
    
    Parameters:
    -----------
    path : str
        Path to the model
    title : str
        Title for the plot
    levels : int
        Number of contour levels
    resolution : int
        Resolution for evaluation grid
    figsize : tuple
        Figure size
    nx, ny, p : int, optional
        Benchmark solver parameters
    file_name : str, optional
        File name to save the plot
    """
    params = read_yaml_file(path)
    model, _, _, _, _ = load_model(path, backend=np)
    
    if nx is None:
        nx = params["solver"]["nx"]
    if ny is None:
        ny = params["solver"]["ny"]
    if p is None:
        p = params["solver"]["p"]
    
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    
    X, Y, _, F_benchmark = eval_model_and_function(model, benchmark, resolution=resolution)
    
    plot_single_contour(X, Y, F_benchmark, title=title, levels=levels, figsize=figsize, file_name=file_name)

def plot_model_and_benchmark_separate(path, title_model="Model(x, y)", title_benchmark="Benchmark Solver", 
                                     levels=100, resolution=100, figsize=(8, 6),
                                     nx=None, ny=None, p=None,
                                     model_file_name=None, benchmark_file_name=None,
                                     print_PDE=False):
    """
    Plot model and benchmark solver in separate figures
    
    Parameters:
    -----------
    path : str
        Path to the model
    title_model : str
        Title for the model plot
    title_benchmark : str
        Title for the benchmark plot
    levels : int
        Number of contour levels
    resolution : int
        Resolution for evaluation grid
    figsize : tuple
        Figure size for each plot
    nx, ny, p : int, optional
        Benchmark solver parameters
    model_file_name, benchmark_file_name : str, optional
        File names to save the plots
    print_PDE : bool
        Whether to print PDE information
    """
    params = read_yaml_file(path)
    model, _, _, _, _ = load_model(path, backend=np)
    
    if nx is None:
        nx = params["solver"]["nx"]
    if ny is None:
        ny = params["solver"]["ny"]
    if p is None:
        p = params["solver"]["p"]
    
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    
    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    
    X, Y, F_model, F_benchmark = eval_model_and_function(model, benchmark, resolution=resolution)
    
    # Plot model (first figure)
    print("Plotting Model:")
    plot_single_contour(X, Y, F_model, title=title_model, levels=levels, 
                       figsize=figsize, file_name=model_file_name)
    
    # Plot benchmark (second figure)
    print("Plotting Benchmark Solver:")
    plot_single_contour(X, Y, F_benchmark, title=title_benchmark, levels=levels, 
                       figsize=figsize, file_name=benchmark_file_name)

def plot_model_vs_function_contour(model, func, title_model="Model(x, y)", title_func="", levels=20, resolution=100, file_name=None):
    """
    Compare model and reference function with side-by-side contour plots.
    
    Creates two contour plots side by side showing the model solution
    and a reference function for direct visual comparison.

    Parameters:
        model: torch.nn.Module or str
            The trained model or path to saved model.
        func: callable
            The reference function to compare against.
        title_model: str, optional
            Title for the model plot. Defaults to "Model(x, y)".
        title_func: str, optional
            Title for the function plot.
        levels: int, optional
            Number of contour levels. Defaults to 20.
        resolution: int, optional
            Grid resolution for evaluation. Defaults to 100.
        file_name: str, optional
            Filename to save plot (with extension). If None, plot not saved.
    """
    # Evaluate both model and function on grid
    X, Y, F, U = eval_model_and_function(model, func, resolution=resolution)

    plt.figure(figsize=(12, 5))

    # Plot reference function
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, U, cmap='viridis', levels=levels)
    plt.colorbar(contour)
    plt.title(title_func)
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot model solution
    plt.subplot(1, 2, 2)
    contour = plt.contourf(X, Y, F, cmap='viridis', levels=levels)
    plt.colorbar(contour)
    plt.title(title_model)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    if file_name is not None:
        plt.savefig(os.path.join(IMAGE_FOLDER_PATH, file_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_model_vs_analytical_contour(model, title_model="Model(x, y)", title_func="", levels=100, resolution=100, print_PDE=False, file_name=None):
    """
    Compare model with analytical solution using side-by-side contour plots.
    
    Parameters:
        model: torch.nn.Module, str, or list
            The model, path to model, or list containing path.
        title_model: str, optional
            Title for the model plot. Defaults to "Model(x, y)".
        title_func: str, optional
            Title for the analytical solution plot.
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

    if not model.PDE.has_solution:
        raise ValueError('The analytical solution does not exist for this PDE')
    
    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    
    plot_model_vs_function_contour(model, func=model.PDE.u, title_model=title_model, 
                                  title_func=title_func, levels=levels, resolution=resolution, file_name=file_name)

def plot_model_vs_benchmark_contour(path, title_model="Model(x, y)", title_func="", levels=100, resolution=100, save_fig=False,
                                   nx=None, ny=None, p=None,
                                   print_PDE=False, file_name=None):
    """
    path: str
        The path to the model
    title_model: str, optional
        The title of the model plot
    title_func: str, optional
        The title of the function plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the evaluation points for the functions
    save_fig: bool, optional
        Deprecated, use file_name instead
    nx: int, optional
        Number of grid points in x-direction
    ny: int, optional
        Number of grid points in y-direction
    p: int, optional
        Parameter for the benchmark solver
    print_PDE: bool, optional
        Whether to print the PDE type
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.

    Returns
    -------
    None
        Shows the plot.
    """
    params = read_yaml_file(path)
    model, _, _, _, _ = load_model(path, backend=np)
    if nx is None:
        nx = params["solver"]["nx"]
    if ny is None:
        ny = params["solver"]["ny"]
    if p is None:
        p = params["solver"]["p"]
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    plot_model_vs_function_contour(model, benchmark, title_model=title_model, title_func=title_func, resolution=resolution, levels=levels, file_name=file_name)

def compare_model_to_benchmark(path, title_contour=["Model(x,y)", "Benchmark", "True Solution"], 
                               title_error="Model vs Benchmark Error", 
                               nx=None, ny=None, p=None, 
                               print_PDE=False, plot_analytical=False):
    """
    Comprehensive comparison between model and benchmark solutions.
    
    Creates multiple plots for thorough comparison: model vs benchmark contours,
    optionally model vs analytical solution, and error visualization.

    Parameters:
        path: str
            Path to the saved model directory.
        title_contour: list of str, optional
            Titles for contour plots [model, benchmark, analytical].
        title_error: str, optional
            Title for the error plot.
        nx: int, optional
            Benchmark solver grid points in x-direction.
        ny: int, optional
            Benchmark solver grid points in y-direction.
        p: int, optional
            Benchmark solver polynomial degree.
        print_PDE: bool, optional
            Whether to print PDE information. Defaults to False.
        plot_analytical: bool, optional
            Whether to also plot analytical solution comparison. Defaults to False.
    """
    # Create model vs benchmark comparison
    plot_model_vs_benchmark_contour(path, title_model=title_contour[0], title_func=title_contour[1], 
                                    nx=nx, ny=ny, p=p, print_PDE=print_PDE)

    # Optionally add analytical solution comparison
    if plot_analytical:
        plot_model_vs_analytical_contour(path, title_model=title_contour[0], title_func=title_contour[2], 
                                         print_PDE=print_PDE)

    # Show error visualization
    plot_model_vs_benchmark_error(path, title=title_error, nx=nx, ny=ny, p=p, print_PDE=print_PDE)