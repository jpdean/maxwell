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
from dolfinx.cpp.fem import create_discrete_gradient
import numpy as np
from dolfinx.common import Timer


def solve_problem(mesh, k, mu, T_0, preconditioner="ams"):
    """Solves a magnetostatic problem.
    Args:
        mesh: the mesh
        k: order of space
        T_0: impressed magnetic field
        preconditioner: "ams" or "gamg"
    Returns:
        A: Magnetic vector potential
        ndofs: number of degrees of freedom
        solve_time: time taked for solver alone
        iterations: number of solver iterations
    """
    V = FunctionSpace(mesh, ("N1curl", k))

    A = TrialFunction(V)
    v = TestFunction(V)

    a = inner(1 / mu * curl(A), curl(v)) * dx

    L = inner(T_0, curl(v)) * dx

    A = Function(V)

    # TODO More steps needed here for Dirichlet boundaries
    mat = assemble_matrix(a)
    mat.assemble()
    vec = assemble_vector(L)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create solver
    ksp = PETSc.KSP().create(mesh.mpi_comm())

    # Set solver options
    ksp.setType("cg")
    ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

    pc = ksp.getPC()
    if preconditioner == "ams":
        # Based on: https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/curl-curl/demo_curl-curl.py
        pc.setType("hypre")
        pc.setHYPREType("ams")

        # Build discrete gradient
        G = create_discrete_gradient(V._cpp_object,
                                     FunctionSpace(mesh, ("CG", 1))._cpp_object)

        # Attach discrete gradient to preconditioner
        pc.setHYPREDiscreteGradient(G)

        cvecs = []
        for i in range(3):
            direction = as_vector([1.0 if i == j else 0.0 for j in range(3)])
            cvecs.append(project(direction, V))
        pc.setHYPRESetEdgeConstantVectors(cvecs[0].vector,
                                          cvecs[1].vector,
                                          cvecs[2].vector)

        # We are dealing with a zero conductivity problem (no mass term), so
        # we need to tell the preconditioner
        pc.setHYPRESetBetaPoissonMatrix(None)

        # Can set more amg options like:
        # opts = PETSc.Options()
        # opts["pc_hypre_ams_cycle_type"] = 13
    elif preconditioner == "gamg":
        pc.setType("gamg")

    # Set matrix operator
    ksp.setOperators(mat)

    ksp.setMonitor(lambda ksp, its, rnorm: print(
        "Iteration: {}, rel. residual: {}".format(its, rnorm)))

    ksp.setFromOptions()

    # Compute solution
    t = Timer()
    t.start()
    ksp.solve(vec, A.vector)
    A.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
    ksp.view()
    t.stop()
    return {"A": A,
            "ndofs": A.vector.getSize(),
            "solve_time": t.elapsed()[0],
            "iterations": ksp.its}


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
