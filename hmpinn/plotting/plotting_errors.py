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

GLOBAL_FONT_SIZE = 40
plt.rcParams.update({'font.size': GLOBAL_FONT_SIZE})


def prepare_single_data(data):
    """
    Check if the data is valid for plotting.
    
    Parameters:
    data: array-like
        The data to check
    
    Returns:
    data: array-like or None
        The data if valid, None otherwise.
    """
    if data is None or len(data) == 0:
        return None
    if np.all(np.isnan(data)) or np.all(np.isinf(data)):
        return None
    return data

def prepare_data(errors, grad_errors, loss, BC_loss):
    """
    Check if the data is valid for plotting.

    Parameters:
    errors: array-like
        The error values per epoch
    grad_errors: array-like
        The gradient error values per epoch
    loss: array-like
        The loss values per epoch
    BC_loss: array-like, optional
        The boundary condition loss values per epoch
    
    Returns:
    errors: array-like or None
        The error values if valid, None otherwise.
    """
    return prepare_single_data(errors), \
            prepare_single_data(grad_errors), \
            prepare_single_data(loss), \
            prepare_single_data(BC_loss)

def plot_errors_in_row(path_list, title_list=None, file_name=None):
    """
    path_list: list
        The list of paths of the models
    title_list: list, optional
        The list of titles
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    Returns
    -------
    None
        Shows the plot.
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
    path_list: list
        The list of paths of the models
    title_list: list, optional
        The list of titles
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    Returns
    -------
    None
        Shows the plot.
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
    path_list: list
        The list of paths of the models
    number_of_rows: int
        The number of rows in the grid
    number_of_columns: int
        The number of columns in the grid
    title_list: list, optional
        The list of titles
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    
    Returns
    -------
    None
        Shows the plot.
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
    errors: array-like
        The error values per epoch
    grad_errors: array-like
        The gradient error values per epoch
    loss: array-like
        The loss values per epoch
    BC_loss: array-like, optional
        The boundary condition loss values per epoch
    title: str, optional
        The title of the plot
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    Returns
    -------
    None
        Shows the plot.
    """
    errors, grad_errors, loss, BC_loss = prepare_data(errors, grad_errors, loss, BC_loss)
    if errors is not None:
        plt.plot(errors, label="Error", color='C0')
    if grad_errors is not None:
        plt.plot(grad_errors, label="Grad Error", color='C1')
    if loss is not None:
        plt.plot(loss, label="Loss", color='C2')
    if BC_loss is not None:
        plt.plot(BC_loss, label="BC Loss", color='C3')
    plt.legend()
    if not with_legend:
        plt.legend().remove()
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
    path: str
        The path to the model
    title: str, optional
        The title of the plot
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    Returns
    -------
    None
        Shows the plot.
    """
    model, errors, grad_errors, loss, BC_loss = load_model(path)
    plot_errors_from_data(errors, grad_errors, loss, BC_loss, title, file_name, with_legend=with_legend)

def plot_model_vs_function_error(model, func, title="", levels=100, resolution=100, file_name=None):
    """
    model: object or str
        The model to evaluate or path to the model
    func: callable
        The function to compare with
    title: str, optional
        The title of the plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the plot
    file_name: str, optional
        The name of the file to save the plot to (include extension). If None, the plot will not be saved.
        Note that the file extension should be included.
    Returns
    -------
    None
        Shows the plot.
    """
    X, Y, F, U = eval_model_and_function(model, func, resolution=resolution)
    res = np.abs(F - U)

    vmin, vmax = res.min(), res.max()
    if vmin <= 0:
        vmin = vmax * 1e-3  # Avoid issues with log(0)

    levels_list = np.logspace(np.log10(vmin), np.log10(vmax), num=levels)

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
    model: object, str, or list
        The model to evaluate, path to the model, or list containing the path
    title: str, optional
        The title of the plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the plot
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
    
    plot_model_vs_function_error(model, func=model.PDE.u, title=title, levels=levels, resolution=resolution, file_name=file_name)

def plot_model_vs_benchmark_error(path, title=None, levels=100, resolution=100, save_fig=False, 
                                   nx=None, ny=None, p=None, 
                                   print_PDE=False, file_name=None):
    """
    path: str
        The path to the model
    title: str, optional
        The title of the plot
    levels: int, optional
        The number of levels in the contour plot
    resolution: int, optional
        The resolution of the plot
    save_fig: bool, optional
        Whether to save the figure (deprecated, use file_name)
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
    plot_model_vs_function_error(model, benchmark, title=title, resolution=resolution, levels=levels, file_name=file_name)