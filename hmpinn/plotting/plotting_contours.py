import matplotlib.pyplot as plt
import numpy as np
import os

from hmpinn import IMAGE_FOLDER_PATH
from hmpinn.models import *
from hmpinn.utils.yaml_utils import load_model, read_yaml_file
from hmpinn.plotting.utils import eval_model_and_function
from hmpinn.benchmark_solver import BenchmarkSolver
from hmpinn.plotting.plotting_errors import plot_model_vs_benchmark_error

def plot_single_contour(X, Y, Z, title="Contour Plot", levels=100, figsize=(8, 6), file_name=None):
    """
    Plot a single contour plot
    
    Parameters:
    -----------
    X, Y, Z : numpy arrays
        Grid coordinates and values
    title : str
        Title for the plot
    levels : int
        Number of contour levels
    figsize : tuple
        Figure size
    file_name : str, optional
        File name to save the plot
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
    Plot only the model in a separate figure
    
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
    file_name : str, optional
        File name to save the plot
    """
    model, _, _, _, _ = load_model(path, backend=np)
    
    # Create benchmark solver just to use eval_model_and_function
    params = read_yaml_file(path)
    nx = params["solver"]["nx"]
    ny = params["solver"]["ny"]
    p = params["solver"]["p"]
    benchmark = BenchmarkSolver(model.PDE, nx=nx, ny=ny, p=p)
    
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
    model: object or str
        The model to evaluate or path to the model
    func: callable
        The function to compare with
    title_model: str, optional
        The title of the model plot
    title_func: str, optional
        The title of the function plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the evaluation points for the functions
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.

    Returns
    -------
    None
        Shows the plot.
    """
    X, Y, F, U = eval_model_and_function(model, func, resolution=resolution)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, U, cmap='viridis', levels=levels)
    plt.colorbar(contour)
    plt.title(title_func)
    plt.xlabel('x')
    plt.ylabel('y')

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
    model: object, str, or list
        The model to evaluate, path to the model, or list containing the path
    title_model: str, optional
        The title of the model plot
    title_func: str, optional
        The title of the function plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the evaluation points for the functions
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
    if isinstance(model, str):
        model, _, _, _, _ = load_model(model)
    elif (isinstance(model, list) and len(model) == 1):
        model, _, _, _, _ = load_model(model[0])
    elif isinstance(model, ModelV0) or isinstance(model, ModelV1):
        pass
    else:
        raise ValueError('The model is not recognized')

    if not model.PDE.has_solution:
        raise ValueError('The analytical solution does not exist')
    
    if print_PDE:
        print(f"The PDE type is {model.PDE}")
    
    plot_model_vs_function_contour(model, func=model.PDE.u, title_model=title_model, title_func=title_func, levels=levels, resolution=resolution, file_name=file_name)


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
    path: str
        The path to the model
    title_contour: list of str, optional
        The titles for the contour plots [model, benchmark, true solution]
    title_error: str, optional
        The title for the error plot
    nx: int, optional
        Number of grid points in x-direction
    ny: int, optional
        Number of grid points in y-direction
    p: int, optional
        Parameter for the benchmark solver
    print_PDE: bool, optional
        Whether to print the PDE type
    plot_analytical: bool, optional
        Whether to plot the analytical solution

    Returns
    -------
    None
        Shows the plots.
    """
    plot_model_vs_benchmark_contour(path, title_model=title_contour[0], title_func=title_contour[1], 
                                    nx=nx, ny=ny, p=p, print_PDE=print_PDE)

    if plot_analytical:
        plot_model_vs_analytical_contour(path, title_model=title_contour[0], title_func=title_contour[2], 
                                         print_PDE=print_PDE)

    plot_model_vs_benchmark_error(path, title=title_error, nx=nx, ny=ny, p=p, print_PDE=print_PDE)