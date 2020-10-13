# Plots the relative error of the velocity in the L2-norm
# against the element size for different values of k.
# Also computes the order of convergence for each k. Note that
# convergence.py must be run first to generate the results.

import numpy as np
import matplotlib.pyplot as plt
import pickle


def compute_conv_rate(hs, errors):
    """Computes convergence rate.
    Args:
        hs: List of characteristic cell sizes
        errors: List of errors"""
    r = np.log(errors[-1] / errors[-2]) / \
        np.log(hs[-1] / hs[-2])
    return r


results = pickle.load(open("convergence_results.p", "rb"))

fig, ax = plt.subplots()
for k, (hs, l2_errors) in results.items():
    ax.loglog(hs, l2_errors, "-x", label=f"k = {k}")
    r = compute_conv_rate(hs, l2_errors)

    print(f"k = {k}")
    print(f"r = {r}")
    print()

ax.legend()
ax.set(xlabel='$h$', ylabel=r'$L^2(\Omega)$-norm of error in B')
ax.invert_xaxis()
plt.show()
