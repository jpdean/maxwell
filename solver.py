# Solver for the magnetostatic A formulation from [1]

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

# TODO Solver currently assumes homogeneous Neumann BCs everywhere. Add
# ability to use more complicated BCs

from dolfinx import Function,  FunctionSpace, solve, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl
from util import project
from dolfinx.fem import assemble_matrix, assemble_vector
from petsc4py import PETSc
from dolfinx.cpp.fem import build_discrete_gradient
import numpy as np


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

    # TODO More steps needed here for Dirichlet boundaries
    mat = assemble_matrix(a, [])
    mat.assemble()
    vec = assemble_vector(L)


    # Create solver
    ksp = PETSc.KSP().create(problem.mesh.mpi_comm())

    # Set solver options
    ksp.setType("cg")
    ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

    # Get the preconditioner and set type (HYPRE AMS)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")

    # Build discrete gradient
    P1 = FunctionSpace(problem.mesh, ("Lagrange", 1))
    G = build_discrete_gradient(V._cpp_object, P1._cpp_object)

    # Attach discrete gradient to preconditioner
    pc.setHYPREDiscreteGradient(G)

    def x_dir(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 1
        values[1] = 0
        values[2] = 0
        return values

    def y_dir(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 0
        values[1] = 1
        values[2] = 0
        return values

    def z_dir(x):
        values = np.empty((3, x.shape[1]))
        values[0] = 0
        values[1] = 0
        values[2] = 1
        return values

    vec_P1 = VectorFunctionSpace(problem.mesh, ("Lagrange", 1))
    x_func = Function(vec_P1)
    y_func = Function(vec_P1)
    z_func = Function(vec_P1)

    x_func.interpolate(x_dir)
    y_func.interpolate(y_dir)
    z_func.interpolate(z_dir)

    x_func_in_V = project(x_func, V)
    y_func_in_V = project(y_func, V)
    z_func_in_V = project(z_func, V)


    pc.setHYPRESetEdgeConstantVectors(x_func_in_V.vector, y_func_in_V.vector, z_func_in_V.vector)

    # We are dealing with a zero conductivity problem (no mass term), so
    # we need to tell the preconditioner
    pc.setHYPRESetBetaPoissonMatrix(None)

    # Set matrix operator
    ksp.setOperators(mat)

    # Compute solution
    ksp.solve(vec, A.vector)
    ksp.view()
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
    # TODO Get k from A somehow and use k - 1 for degree of V
    V = VectorFunctionSpace(mesh, ("DG", k))
    B = project(curl(A), V)
    return B
