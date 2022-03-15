# Computes convergence results for problems in problems.py that
# have analytical solutions. Results are pickled to convergence_results.p
# and can be plotted using plot_convergence.py.

import numpy as np
import pickle
from solver import solve_problem
import problems
from util import L2_norm

# Problem
create_problem = problems.create_problem_0
# Characteristic element size
hs = [1 / 4, 1 / 8, 1 / 16, 1 / 32]  # 1 / 64]
# Polynomial orders
ks = [1, 2]
# Coefficients
alpha = 1.0
beta = 1.0

results = {k: [] for k in ks}
iterations = {k: [] for k in ks}
meshsize = {k: [] for k in ks}
ndofs = {k: [] for k in ks}
for k in ks:
    for h in hs:
        mesh, u_e, f, boundary_marker = create_problem(h, alpha, beta)
        u, info = solve_problem(mesh, k, alpha, beta, f, boundary_marker, u_e)
        e = L2_norm(u - u_e)
        results[k].append(e)
        meshsize[k].append(h)
        iterations[k].append(info["iterations"])
        ndofs[k].append(info["ndofs"])

if mesh.comm.rank == 0:
    for k in ks:
        res = np.array(meshsize[k])
        err = np.array(results[k])
        print(f"k {k}")
        print(f"Iterations {iterations[k]}")
        print(f"Number of dofs {ndofs[k]}")
        print(f"L2 error {err}")
        print(f"Convergence rate {np.log(err[1:] / err[:-1]) / np.log(res[1:] / res[:-1])}")

# Save results
# TODO Use JSON instead
pickle.dump(results, open("convergence_results.p", "wb"))
