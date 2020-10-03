# TODO Add references
# TODO Add mu!

from dolfinx import (Function,  FunctionSpace, solve,
                     UnitCubeMesh, VectorFunctionSpace)
import numpy as np
from mpi4py import MPI
from dolfinx.fem import assemble_scalar
from ufl import (TrialFunction, TestFunction, inner, dx, curl,
                 SpatialCoordinate, cos, pi, as_vector)
from dolfinx.io import XDMFFile


def project(f, V):
    """Projects the function f onto the space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    u = Function(V)
    solve(a == L, u, [], petsc_options={"ksp_type": "cg"})
    return u


def save_function(v, mesh, filename):
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as f:
        f.write_mesh(mesh)
        f.write_function(v)


def L2_norm(v):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(assemble_scalar(inner(v, v) * dx),
                                            op=MPI.SUM))


def solve_problem(k, mesh, T_0):
    V = FunctionSpace(mesh, ("N1curl", k))

    A = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(A), curl(v)) * dx
    L = inner(T_0, curl(v)) * dx

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


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory

# Problem 1
k = 1
n = 32
mu = 1
mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
x = SpatialCoordinate(mesh)
T_0 = as_vector((- pi * cos(x[2] * pi) / mu,
                 - pi * cos(x[0] * pi) / mu,
                 - pi * cos(x[1] * pi) / mu))
A = solve_problem(k, mesh, T_0)
save_function(A, mesh, "A.xdmf")
B = compute_B(A, k - 1, mesh)
save_function(B, mesh, "B.xdmf")
B_e = as_vector((- pi * cos(x[2] * pi),
                 - pi * cos(x[0] * pi),
                 - pi * cos(x[1] * pi)))
e = L2_norm(B - B_e)
print(f"L2-norm of error in B = {e}")
