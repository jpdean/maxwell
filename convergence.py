# Computes convergence results for problems in problems.py that
# have analytical solutions. Results are pickled to convergence_results.p
# and can be plotted using plot_convergence.py.

import pickle
from solver import solve_problem
import problems
from util import L2_norm

# Problem
create_problem = problems.create_problem_0
# Characteristic element size
hs = [1 / 4, 1 / 8, 1 / 16, 1 / 32]
# Polynomial orders
ks = [1]
# Coefficients
alpha = 1.0
beta = 1.0

results = {}
for k in ks:
    l2_errors = []
    for h in hs:
        mesh, u_e, f, boundary_marker = create_problem(h, alpha, beta)
        u = solve_problem(mesh, k, alpha, beta, f, boundary_marker, u_e)[0]
        e = L2_norm(u - u_e)
        l2_errors.append(e)
    results[k] = (hs, l2_errors)

print(results)
# Save results
# TODO Use JSON instead
pickle.dump(results, open("convergence_results.p", "wb"))
