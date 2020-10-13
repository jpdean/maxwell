# TODO Add references
# TODO Add Homegeneous Dirichlet BC's!

from dolfinx import Function,  FunctionSpace, solve, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl
from util import project


def solve_problem(problem):
    # TODO Currently assumes homogeneous Neumann BCs
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
    # TODO Get AMS working properly
    solve(a == L, A, [], petsc_options={"ksp_type": "cg",
                                        "pc_type": "icc",
                                        "ksp_rtol": 1e-12,
                                        "ksp_monitor": None})
    return A


def compute_B(A, k, mesh):
    """Computes the magnetic field, B, from the magnetic vector potential, A.
    k is the degree of the space, and should be 1 less that the degree of the
    space for A.
    """
    # TODO Get k from A somehow and use k - 1 in the function space definition,
    # rather than having to do it manually
    if mesh.topology.dim == 2:
        # Function space for B
        V = FunctionSpace(mesh, ("DG", k))
        B = project(A[1].dx(0) - A[0].dx(1), V)
    elif mesh.topology.dim == 3:
        V = VectorFunctionSpace(mesh, ("DG", k))
        B = project(curl(A), V)
    return B
