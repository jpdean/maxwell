# TODO Add references
# TODO Add mu!

from dolfinx import Function,  FunctionSpace, solve, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl, Measure
from util import project


def solve_problem(problem):
    V = FunctionSpace(problem.mesh, ("N1curl", problem.k))

    A = TrialFunction(V)
    v = TestFunction(V)

    mu = problem.mu
    T_0 = problem.T_0
    a = inner(1 / mu * curl(A), curl(v)) * dx

    dx_mt = Measure("dx", subdomain_data=problem.cell_mt)
    L = inner(T_0, curl(v)) * dx_mt(2)

    A = Function(V)
    # NOTE That A is not unique because of the nullspace of the curl operator
    # i.e. curl(grad(\phi)) = 0 for any \phi, so for any A that is a solution,
    # A + grad(\phi) is also a solution. Hence, must use an iterative solver.
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
        # B = curl(A), which in 2D is given by the formula here:
        # https://www.khanacademy.org/math/multivariable-calculus/greens-theorem-and-stokes-theorem/formal-definitions-of-divergence-and-curl/a/defining-curl
        # B is unique despite A not being
        B = project(A[1].dx(0) - A[0].dx(1), V)
    elif mesh.topology.dim == 3:
        V = VectorFunctionSpace(mesh, ("DG", k))
        B = project(curl(A), V)
    return B
