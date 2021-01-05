# Records the number of iterations and time taken to solve problems of
# increasing size. The results are saved to iteration_results.p and can
# be plotted with plot_iterations.py.

from problems import create_problem_1
from solver import solve_problem
import numpy as np
import pickle

# Space degree
k = 1
# Permeability
mu = 1
# Initial element diameter
hi = 0.25
# Number of dofs ratio
r = 2
# Number of data points
n = 7

hs = [hi]
for i in range(n):
    hs.append(hs[-1] / np.cbrt(r))

ndofs = []
times = []
iters = []
for h in hs:
    mesh, T_0, B_e = create_problem_1(h, mu)
    results = solve_problem(mesh, k, mu, T_0)

    ndofs.append(results["ndofs"])
    times.append(results["solve_time"])
    iters.append(results["iterations"])

    # Write in each loop incase of crash etc.
    results = [ndofs, times, iters]
    pickle.dump(results, open("iteration_results.p", "wb"))
