# TODO Get 3D working
# TODO Add references

from dolfinx import (UnitSquareMesh, DirichletBC, Function,
                     FunctionSpace, Constant, solve,
                     UnitCubeMesh, VectorFunctionSpace)
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, assemble_scalar
from ufl import (TrialFunction, TestFunction, inner, dx, curl,
                 SpatialCoordinate, sin, cos, pi, as_vector)
from dolfinx.io import XDMFFile


def project(f, V):
    """Projects the function f onto the space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    u = Function(V)
    solve(a == L, u, [], petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu"})
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


def solve_problem(k, mesh, f, bound_marker):
    V = FunctionSpace(mesh, ("N1curl", k))

    # Create function and set all dofs to 0
    A_d = Function(V)
    A_d.vector.set(0.0)
    # Locate boundary facets
    d = mesh.topology.dim
    facets = locate_entities_boundary(mesh, d - 1, bound_marker)
    # Locate boundary dofs
    dofs = locate_dofs_topological(V, d - 1, facets)
    # Apply boundary condition. NOTE Since these are edge elements,
    # this means that we are setting the tangential component of
    # A to zero on the boundary. The normal component is not
    # constrained (look at the basis funcs to see this)
    bc = DirichletBC(A_d, dofs)

    A = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(A), curl(v)) * dx
    L = inner(f, v) * dx

    A = Function(V)
    # NOTE That A is not unique because of the nullspace of the curl operator
    # i.e. curl(grad(\phi)) = 0 for any \phi, so for any A that is a solution,
    # A + grad(\phi) is also a solution. Hence, must use an iterative solver.
    solve(a == L, A, bc, petsc_options={"ksp_type": "cg",
                                        "pc_type": "jacobi",
                                        "ksp_rtol": 1e-12})
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


# FIXME Move boundary markers to more sensible place
def square_bound_marker(x):
    left = np.isclose(x[0], 0)
    right = np.isclose(x[0], 1)
    bottom = np.isclose(x[1], 0)
    top = np.isclose(x[1], 1)

    l_r = np.logical_or(left, right)
    b_t = np.logical_or(bottom, top)

    return np.logical_or(l_r, b_t)


# FIXME Move boundary markers to more sensible place
def cube_bound_marker(x):
    left = np.isclose(x[0], 0)
    right = np.isclose(x[0], 1)
    bottom = np.isclose(x[1], 0)
    top = np.isclose(x[1], 1)
    back = np.isclose(x[2], 0)
    front = np.isclose(x[2], 1)

    l_r = np.logical_or(left, right)
    b_t = np.logical_or(bottom, top)
    b_f = np.logical_or(back, front)

    l_r_b_t = np.logical_or(l_r, b_t)

    return np.logical_or(l_r_b_t, b_f)


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory

# 2D Problems
# Problem 1
k = 1
n = 32
mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
f = Constant(mesh, (1, 0))
A = solve_problem(k, mesh, f, square_bound_marker)
save_function(A, mesh, "A.xdmf")
B = compute_B(A, k - 1, mesh)
save_function(B, mesh, "B.xdmf")
x = SpatialCoordinate(mesh)
B_e = x[1] - 0.5
e = L2_norm(B - B_e)
print(f"L2-norm of error in B = {e}")

# Problem 2
# k = 1
# n = 64
# mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
# x = SpatialCoordinate(mesh)
# f = as_vector((pi**2 * sin(x[1] * pi), 0))
# A = solve_problem(k, mesh, f, square_bound_marker)
# save_function(A, mesh, "A.xdmf")
# B = compute_B(A, k - 1, mesh)
# save_function(B, mesh, "B.xdmf")
# B_e = - pi * cos(x[1] * pi)
# e = L2_norm(B - B_e)
# print(f"L2-norm of error in B = {e}")

# 3D Problems
# Problem 3
# k = 1
# n = 32
# mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
# x = SpatialCoordinate(mesh)
# f = as_vector((- 2 * x[1]**2 + 2 * x[1] - 2 * x[2]**2 + 2 * x[2],
#                0,
#                0))
# A = solve_problem(k, mesh, f, cube_bound_marker)
# save_function(A, mesh, "A.xdmf")
# B = compute_B(A, k - 1, mesh)
# save_function(B, mesh, "B.xdmf")
# B_e = as_vector((0,
#                  x[1] * (x[1] - 1) * (2 * x[2] - 1),
#                  x[2] * (1 - 2 * x[1]) * (x[2] - 1)))
# e = L2_norm(B - B_e)
# print(f"L2-norm of error in B = {e}")

# Problem 4
# k = 1
# n = 16
# mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
# x = SpatialCoordinate(mesh)
# f = as_vector((2 * pi**2 * sin(x[1] * pi) * sin(x[2] * pi),
#                0,
#                0))
# A = solve_problem(k, mesh, f, cube_bound_marker)
# save_function(A, mesh, "A.xdmf")
# B = compute_B(A, k - 1, mesh)
# save_function(B, mesh, "B.xdmf")
# B_e = as_vector((0,
#                  pi * sin(x[1] * pi) * cos(x[2] * pi),
#                  - pi * sin(x[2] * pi) * cos(x[1] * pi)))
# e = L2_norm(B - B_e)
# print(f"L2-norm of error in B = {e}")
