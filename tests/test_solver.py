from hmpinn.utils import get_PDE_class
import numpy as np
from hmpinn.PDEs import PDE_NAME_TO_CLASS
from hmpinn.benchmark_solver.benchmark_solver import BenchmarkSolver
import matplotlib.pyplot as plt



# A dict that gives every PDE a shortened name
PDE_names = PDE_NAME_TO_CLASS.keys()

def test_solver(PDE_name):
    print(f"Testing {PDE_name}...")
    PDE_class = get_PDE_class(PDE_name)
    PDE = PDE_class(backend=np)
    benchmark_sol = BenchmarkSolver(PDE)
    benchmark_sol.plot(block=True) 
    # Wait 10 seconds
    # time.sleep(10)
    # Close the plot
    plt.close()



if __name__ == "__main__":
    for PDE_name in PDE_names:
        # if PDE_name in ["piecewise_diff", "non_sym_hess"]:
        if PDE_name in ["piecewise_diff"]:
            # Skip the piecewise diffusion test for now
            test_solver(PDE_name)
        print("")
    print("All tests passed.")