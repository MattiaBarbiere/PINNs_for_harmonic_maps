# A class that adapts the spline solver to the PDE problem.
from hmpinn.benchmark_solver.solver import solve
import hmpinn.benchmark_solver.jitBSpline as jitBSpline
from typing import Optional
import torch
import numpy as np
from numpy.typing import NDArray, ArrayLike

class BenchmarkSolver():
    """
    A class that adapts the spline solver to the PDE problem in the hmpinn library."""
    def __init__(self, PDE, nx: int = 21, ny: Optional[int] = None, p: int = 3):
        if PDE.backend != np:
            raise ValueError("PDE backend must be numpy.")
        self.PDE = PDE
        self.nx = nx
        self.ny = ny if ny is not None else nx
        self.p = p
        

        # Get the solver for the given PDE
        self.solver = solve(self.adapted_diff_matrix, 
                            self.adapted_f, 
                            self.nx, 
                            self.ny, 
                            self.p,
                            self.adapted_BC,
                            not self.PDE.is_in_divergence_form)
                       

    def stack_arrays(self, X: ArrayLike, Y: ArrayLike) -> torch.Tensor:
        """
        Convert numpy arrays to torch tensors.

        Parameters
        X : ArrayLike
            The X coordinates to evaluate at.
        Y : ArrayLike
            The Y coordinates to evaluate at.

        Returns
        torch.Tensor
            The merged tensor of X and Y coordinates.
        """

        # Merge the X and Y arrays into a single array
        XY = np.stack((X, Y), axis=-1)

        return np.reshape(XY, (-1, 2))
    
    def adapted_f(self, X: ArrayLike, Y: ArrayLike):
        """
        Adapt the PDE to the spline solver.

        Parameters
        X : ArrayLike
            The X coordinates to evaluate at.
        Y : ArrayLike
            The Y coordinates to evaluate at.

        Returns
        NDArray[np.float64]
            The evaluated PDE at the given X and Y coordinates.
        """
        # Convert X and Y to torch tensors
        XY = self.stack_arrays(X, Y)

        # Evaluate the PDE function
        # print("Donig", self.PDE.f(XY).shape, self.PDE.f(XY).astype(np.float64))
        return self.PDE.f(XY)[0]
    
    def adapted_diff_matrix(self, X: ArrayLike, Y: ArrayLike):
        """
        Adapt the PDE to the spline solver.

        Parameters
        X : ArrayLike
            The X coordinates to evaluate at.
        Y : ArrayLike
            The Y coordinates to evaluate at.

        Returns
        NDArray[np.float64]
            The evaluated PDE at the given X and Y coordinates.
        """
        # Convert X and Y to torch tensors
        XY = self.stack_arrays(X, Y)

        # Evaluate the PDE function
        # print("here1", self.PDE.diffusion_matrix(XY)[:, :, :].shape, self.PDE.diffusion_matrix(XY)[0].shape)
        return self.PDE.diffusion_matrix(XY)[0]
    
    def adapted_BC(self, X: ArrayLike, Y: ArrayLike):
        """
        Adapt the PDE to the spline solver.

        Parameters
        X : ArrayLike
            The X coordinates to evaluate at.
        Y : ArrayLike
            The Y coordinates to evaluate at.

        Returns
        NDArray[np.float64]
            The evaluated PDE at the given X and Y coordinates.
        """
        # Convert X and Y to torch tensors
        XY = self.stack_arrays(X, Y)

        # Evaluate the PDE functions
        # print("here1", self.PDE.BC(XY)[0].shape)
        # print("here", self.PDE.BC(XY)[0].astype(np.float64))
        return self.PDE.BC(XY)[0]
    
    def __call__(self, XY: torch.Tensor):
        """
        Call the solver with the given arguments.

        Parameters
        XY : torch.Tensor
            The X and Y coordinates to evaluate the spline at.

        Returns
        torch.Tensor
            The evaluated PDE at the given cX and Y coordinates.
        """
        # Convert XY to numpy array
        XY = XY.detach().numpy()

        # Split the XY array into X and Y arrays
        X = XY[:, 0]
        Y = XY[:, 1]

        return self.solver(X, Y)
    
    def plot(self, block = True, title = "Benchmark Solver"):
        self.solver.plot(self.nx, self.ny, title, block=block)