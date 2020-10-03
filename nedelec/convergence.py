# Computes convergence results for problems in problems.py that
# have analytical solutions. Results are picked to convergence_results.p
# and can be plotted using plot_convergence.py.

import pickle
from solver import solve_problem, compute_B
import problems
from util import L2_norm

# Problem
prob_gen = problems.create_problem_1
# Characteristic element size
hs = [1 / 8, 1 / 16, 1 / 32]
# Polynomial order
ks = [1, 2]
# Permiability
mu = 1.0

results = {}
for k in ks:
    l2_errors = []
    for h in hs:
        problem = prob_gen(h, k, mu)
        A = solve_problem(problem)
        B = compute_B(A, k - 1, problem.mesh)
        e = L2_norm(B - problem.B_e)
        l2_errors.append(e)
    results[k] = (hs, l2_errors)

# Save results
pickle.dump(results, open("convergence_results.p", "wb"))
