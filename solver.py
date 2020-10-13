# Solver for the magnetostatic A formulation from [1]

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

# TODO Solver currently assumes homogeneous Neumann BCs everywhere. Add
# ability to use more complicated BCs

from dolfinx import Function,  FunctionSpace, solve, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl
from util import project


def solve_problem(problem):
    """Solves a magnetostatic problem.
    Args:
        problem: A problem created by problems.py
    Returns:
        A: The magnetic vector potential
    """
    V = FunctionSpace(problem.mesh, ("N1curl", problem.k))

    A = TrialFunction(V)
    v = TestFunction(V)

    mu = problem.mu
    T_0 = problem.T_0
    a = inner(1 / mu * curl(A), curl(v)) * dx

    L = inner(T_0, curl(v)) * dx

    A = Function(V)
    # NOTE That A is not unique because of the nullspace of the curl operator
    # i.e. curl(grad(\phi)) = 0 for any \phi, so for any A that is a solution,
    # A + grad(\phi) is also a solution. Hence, must use an iterative solver.
    # TODO Set up solver manually
    # TODO Use AMS
    solve(a == L, A, [], petsc_options={"ksp_type": "cg",
                                        "pc_type": "icc",
                                        "ksp_rtol": 1e-12,
                                        "ksp_monitor": None})
    return A


def compute_B(A, k, mesh):
    """Computes the magnetic field.
    Args:
        A: Magnetic vector potential
        k: Degree of DG space for B
        mesh: The mesh
    Returns:
        B: The magnetic flux density
    """
    # TODO Get k from A somehow and use k - 1 in the function space definition,
    # rather than having to do it manually
    V = VectorFunctionSpace(mesh, ("DG", k))
    B = project(curl(A), V)
    return B
