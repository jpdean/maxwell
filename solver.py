# Solver for the magnetostatic A formulation from [1]

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

# TODO Solver currently assumes homogeneous Neumann BCs everywhere. Add
# ability to use more complicated BCs

from dolfinx import Function,  FunctionSpace, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl, as_vector
from util import project
from dolfinx.fem import assemble_matrix, assemble_vector
from petsc4py import PETSc
from dolfinx.cpp.fem import build_discrete_gradient
import numpy as np


def solve_problem(mesh, k, mu, T_0):
    """Solves a magnetostatic problem.
    Args:
        problem: A problem created by problems.py
    Returns:
        A: The magnetic vector potential. Note that A is not unique because
           of the nullspace of the curl operator i.e. curl(grad(\phi)) = 0 for
           any \phi, so for any A that is a solution, A + grad(\phi) is also a
           solution.
    """
    V = FunctionSpace(mesh, ("N1curl", k))

    A = TrialFunction(V)
    v = TestFunction(V)

    a = inner(1 / mu * curl(A), curl(v)) * dx

    L = inner(T_0, curl(v)) * dx

    A = Function(V)

    # TODO More steps needed here for Dirichlet boundaries
    mat = assemble_matrix(a, [])
    mat.assemble()
    vec = assemble_vector(L)

    # Create solver
    ksp = PETSc.KSP().create(mesh.mpi_comm())

    # Set solver options
    ksp.setType("cg")
    ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

    # Get the preconditioner and set type
    # Based on: https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/curl-curl/demo_curl-curl.py
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("ams")

    # Build discrete gradient
    G = build_discrete_gradient(V._cpp_object,
                                FunctionSpace(mesh, ("CG", 1))._cpp_object)

    # Attach discrete gradient to preconditioner
    pc.setHYPREDiscreteGradient(G)

    cvecs = []
    for i in range(3):
        direction = as_vector([1.0 if i == j else 0.0 for j in range(3)])
        cvecs.append(project(direction, V).vector)

    pc.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])

    # We are dealing with a zero conductivity problem (no mass term), so
    # we need to tell the preconditioner
    pc.setHYPRESetBetaPoissonMatrix(None)

    # Set matrix operator
    ksp.setOperators(mat)

    ksp.setMonitor(lambda ksp, its, rnorm: print("Iteration: {}, rel. residual: {}".format(its, rnorm)))
    ksp.setFromOptions()

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
