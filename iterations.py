# Records the number of iterations and time taken to solve problems of
# increasing size. The results are saved to iteration_results.p and can
# be plotted with plot_iterations.py.

from problems import create_problem_1
from solver import solve_problem
import pickle

# Space degree
k = 1
# Permeability
mu = 1
# Initial element diameter
h = 1 / 4
# Number of dofs ratio
r = 2
# Number of data points
n = 4

ndofs = []
times = []
iters = []
for i in range(n):
    mesh, A_e, f, boundary_marker = create_problem_1(h, mu)
    results = solve_problem(mesh, k, mu, f, boundary_marker, A_e)[1]

    ndofs.append(results["ndofs"])
    times.append(results["solve_time"])
    iters.append(results["iterations"])

    # Write in each loop incase of crash etc.
    results = [ndofs, times, iters]
    pickle.dump(results, open("iteration_results.p", "wb"))
    h /= r
