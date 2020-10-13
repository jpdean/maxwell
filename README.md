# Maxwell
Solver for Maxwell's equations using FEniCS-X. Currently only solves simple magnetostatics problems, see solver.py for more information about the formulation.

To solve a simple problem with a manufactured solution, run:

    python3 problems.py

See problems.py for more information.

Convergence results can be generated using

    python3 convergence.py

and plotted with

    python3 plot_convergence.py