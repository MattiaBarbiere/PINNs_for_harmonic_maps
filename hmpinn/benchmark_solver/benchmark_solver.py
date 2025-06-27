"""
Benchmark Solver for PDE Problems.

This file provides a BenchmarkSolver class that adapts a spline-based solver
to work with PDE problems in the hmpinn library. The solver acts as a bridge
between the PDE specification and the underlying numerical solver.

Classes:
    BenchmarkSolver: A spline-based solver adapter for PDE problems.
"""
import torch
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional

from hmpinn.benchmark_solver.solver import solve

class BenchmarkSolver:
    """
    A spline-based solver adapter for PDE problems in the hmpinn library.
    
    This class provides an interface between PDE specifications and a spline-based
    numerical solver. It handles the adaptation of PDE functions, boundary conditions,
    and diffusion matrices to work with the underlying solver implementation.
    
    Attributes:
        PDE: The PDE problem instance to solve.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        p (int): Degree of the spline polynomials.
        solver: The underlying spline solver instance.
    
    Raises:
        ValueError: If the PDE backend is not numpy.
    """
    
    def __init__(self, PDE, nx: int = 21, ny: Optional[int] = None, p: int = 3):
        """
        Initialize the BenchmarkSolver with PDE and solver parameters.
        
        Parameters:
            PDE: The PDE problem instance with numpy backend.
            nx: int, optional
                Number of grid points in x-direction. Defaults to 21.
            ny: int, optional
                Number of grid points in y-direction. If None, uses nx value. Defaults to None.
            p: int, optional
                Degree of spline polynomials. Defaults to 3.
        
        Raises:
            ValueError: If PDE backend is not numpy.
        """
        if PDE.backend != np:
            raise ValueError("PDE backend must be numpy.")
            
        self.PDE = PDE
        self.nx = nx
        self.ny = ny if ny is not None else nx
        self.p = p

        # Initialize the solver with adapted PDE functions
        self.solver = solve(
            self.adapted_diff_matrix, 
            self.adapted_f, 
            self.nx, 
            self.ny, 
            self.p,
            self.adapted_BC,
            not self.PDE.is_in_divergence_form
        )

    def stack_arrays(self, X: ArrayLike, Y: ArrayLike) -> np.ndarray:
        """
        Stack X and Y coordinate arrays into a single 2D array.
        
        Converts separate X and Y coordinate arrays into a combined array
        suitable for PDE function evaluation.
        
        Parameters:
            X: array-like
                X coordinates to evaluate at.
            Y: array-like
                Y coordinates to evaluate at.
        
        Returns:
            np.ndarray
                Reshaped array of shape (N, 2) where N is the total number of coordinate pairs.
        """
        # Stack X and Y arrays along the last axis
        XY = np.stack((X, Y), axis=-1)
        
        # Reshape to (N, 2) format for PDE evaluation
        return np.reshape(XY, (-1, 2))
    
    def adapted_f(self, X: ArrayLike, Y: ArrayLike) -> np.ndarray:
        """
        Evaluate the PDE source term at given coordinates.
        
        Adapts the PDE source function to work with the spline solver's
        expected interface by converting coordinate arrays and extracting
        the appropriate component.
        
        Parameters:
            X: array-like
                X coordinates to evaluate at.
            Y: array-like
                Y coordinates to evaluate at.
        
        Returns:
            np.ndarray
                Evaluated PDE source term at the given coordinates.
        """
        # Convert coordinate arrays to the format expected by PDE
        XY = self.stack_arrays(X, Y)
        
        # Evaluate the PDE source function and return first component
        return self.PDE.f(XY)[0]
    
    def adapted_diff_matrix(self, X: ArrayLike, Y: ArrayLike) -> np.ndarray:
        """
        Evaluate the PDE diffusion matrix at given coordinates.
        
        Adapts the PDE diffusion matrix function to work with the spline solver's
        expected interface by converting coordinate arrays and extracting
        the appropriate component.
        
        Parameters:
            X: array-like
                X coordinates to evaluate at.
            Y: array-like
                Y coordinates to evaluate at.
        
        Returns:
            np.ndarray
                Evaluated diffusion matrix at the given coordinates.
        """
        # Convert coordinate arrays to the format expected by PDE
        XY = self.stack_arrays(X, Y)
        
        # Evaluate the diffusion matrix and return first component
        return self.PDE.diffusion_matrix(XY)[0]
    
    def adapted_BC(self, X: ArrayLike, Y: ArrayLike) -> np.ndarray:
        """
        Evaluate the PDE boundary conditions at given coordinates.
        
        Adapts the PDE boundary condition function to work with the spline solver's
        expected interface by converting coordinate arrays and extracting
        the appropriate component.
        
        Parameters:
            X: array-like
                X coordinates to evaluate at.
            Y: array-like
                Y coordinates to evaluate at.
        
        Returns:
            np.ndarray
                Evaluated boundary conditions at the given coordinates.
        """
        # Convert coordinate arrays to the format expected by PDE
        XY = self.stack_arrays(X, Y)
        
        # Evaluate the boundary conditions and return first component
        return self.PDE.BC(XY)[0]
    
    def __call__(self, XY: torch.Tensor) -> np.ndarray:
        """
        Evaluate the solved PDE at given coordinates.
        
        Provides a callable interface to the solver, converting torch tensors
        to numpy arrays and splitting coordinates for evaluation.
        
        Parameters:
            XY: torch.Tensor
                Coordinates as tensor of shape (N, 2) where columns are X and Y coordinates respectively.
        
        Returns:
            np.ndarray
                Evaluated solution at the given coordinates.
        """
        # Convert torch tensor to numpy array
        XY_np = XY.detach().numpy()
        
        # Extract X and Y coordinates
        X = XY_np[:, 0]
        Y = XY_np[:, 1]
        
        # Evaluate the solver at the given coordinates
        return self.solver(X, Y)
    
    def plot(self, block: bool = True, title: str = "Benchmark Solver") -> None:
        """
        Plot the solved PDE solution.
        
        Creates a visualization of the solved PDE using the underlying
        solver's plotting functionality.
        
        Parameters:
            block: bool, optional
                Whether to block execution until plot is closed. Defaults to True.
            title: str, optional
                Title for the plot. Defaults to "Benchmark Solver".
        """
        self.solver.plot(self.nx, self.ny, title, block=block)