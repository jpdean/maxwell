# Maxwell
Solver for Maxwell's equations using FEniCS-X. Currently only solves simple magnetostatics problems, see solver.py for more information about the formulation.

To solve a simple problem with a manufactured solution, run:
```bash
    python3 problems.py
```
For problem options, run
```bash
   python3 problems.py --help
```

See problems.py for more information.

Convergence results can be generated using
```bash
    python3 convergence.py
```
and plotted with
```bash
    python3 plot_convergence.py
```