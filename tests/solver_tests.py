"""
Benchmark Solver Test Suite.

This module provides comprehensive testing functionality for the benchmark solver
across different PDE types, ensuring correct initialization and plotting capabilities.

The tests validate that the benchmark solver can handle various PDE configurations
and produce valid plots without errors.
"""

import numpy as np
import matplotlib.pyplot as plt

from hmpinn.utils import get_PDE_class
from hmpinn.PDEs import PDE_NAME_TO_CLASS
from hmpinn.benchmark_solver.benchmark_solver import BenchmarkSolver


# PDEs to skip during testing due to complexity or special requirements
SKIP_PDES = {
    "piecewise_diff", 
    "non_sym_hess", 
    "quarter_annulus_hm", 
    "L_bend_hm", 
    "sin_boundaries_hm", 
    "poly_boundaries_hm"
}


def test_solver(pde_name: str) -> bool:
    """
    Test the benchmark solver for a specific PDE.
    
    Creates a PDE instance with numpy backend and tests that the benchmark
    solver can initialize and plot without errors.
    
    Args:
        pde_name: Name of the PDE to test from the available PDE classes.
        
    Returns:
        bool: True if test passes successfully, False if any error occurs.
    """
    try:
        print(f"Testing {pde_name}...")
        pde_class = get_PDE_class(pde_name)
        pde = pde_class(backend=np)
        benchmark_sol = BenchmarkSolver(pde)
        benchmark_sol.plot(block=True)
        plt.close()
        return True
    except Exception as e:
        print(f"Error testing {pde_name}: {e}")
        return False


def run_all_tests() -> None:
    """
    Run benchmark solver tests for all available PDEs.
    
    Iterates through all available PDE classes and tests the benchmark solver
    functionality, skipping PDEs that are marked for exclusion.
    
    Raises:
        AssertionError: If any tests fail, listing all failed PDE names.
    """
    pde_names = PDE_NAME_TO_CLASS.keys()
    failed_tests = []
    
    for pde_name in pde_names:
        if pde_name in SKIP_PDES:
            print(f"Skipping {pde_name}...")
            continue
            
        if not test_solver(pde_name):
            failed_tests.append(pde_name)
        print()
    
    if failed_tests:
        print(f"Failed tests: {failed_tests}")
        raise AssertionError(f"Tests failed for: {', '.join(failed_tests)}")
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()