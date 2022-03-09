# Computes convergence results for problems in problems.py that
# have analytical solutions. Results are pickled to convergence_results.p
# and can be plotted using plot_convergence.py.

import pickle
from solver import solve_problem, compute_B
import problems
from util import L2_norm

# Problem
create_problem = problems.create_problem_1
# Characteristic element size
hs = [1 / 4, 1 / 8, 1 / 16, 1/32]
# Polynomial orders
ks = [1]
# Permeability
mu = 1.0

results = {}
for k in ks:
    l2_errors = []
    for h in hs:
        mesh, T_0, B_e = create_problem(h, mu)
        result = solve_problem(mesh, k, mu, T_0)
        B = compute_B(result["A"], k - 1)
        e = L2_norm(B - B_e)
        l2_errors.append(e)
    results[k] = (hs, l2_errors)
print(results)
# Save results
pickle.dump(results, open("convergence_results.p", "wb"))
