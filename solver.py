# Solver for the magnetostatic A formulation from [1]

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

# TODO Solver currently assumes homogeneous Neumann BCs everywhere. Add
# ability to use more complicated BCs

import numpy as np
from dolfinx.common import Timer
from dolfinx.cpp.fem.petsc import create_discrete_gradient
from dolfinx.fem import Expression, Function, FunctionSpace, form, petsc
from dolfinx.mesh import Mesh
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, as_vector, curl, dx, inner
from ufl.core.expr import Expr
from typing import Dict
from util import project


def solve_problem(mesh: Mesh, k: int, mu: np.float64, T_0: Expr,
                  preconditioner: str = "ams", jit_params: Dict = None,
                  form_compiler_params: Dict = None):
    """Solves a magnetostatic problem.
    Args:
        mesh: the mesh
        k: order of space
        mu: Permability
        T_0: impressed magnetic field
        preconditioner: "ams" or "gamg"
        form_compiler_params: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_params:See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
    Returns:
        A: Magnetic vector potential
        ndofs: number of degrees of freedom
        solve_time: time taked for solver alone
        iterations: number of solver iterations
    """
    if form_compiler_params is None:
        form_compiler_params = {}
    if jit_params is None:
        jit_params = {}

    V = FunctionSpace(mesh, ("N1curl", k))

    A = TrialFunction(V)
    v = TestFunction(V)

    a = form(inner(1 / mu * curl(A), curl(v)) * dx,
             form_compiler_params=form_compiler_params, jit_params=jit_params)

    L = form(inner(T_0, curl(v)) * dx,
             form_compiler_params=form_compiler_params, jit_params=jit_params)

    A = Function(V)

    # TODO More steps needed here for Dirichlet boundaries
    mat = petsc.assemble_matrix(a)
    mat.assemble()
    vec = petsc.assemble_vector(L)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Create solver
    ksp = PETSc.KSP().create(mesh.comm)

    # Set solver options
    ksp.setType("cg")
    ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

    pc = ksp.getPC()
    if preconditioner == "ams":
        # Based on: https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/curl-curl/demo_curl-curl.py # noqa: E501
        pc.setType("hypre")
        pc.setHYPREType("ams")

        # Build discrete gradient
        V_CG = FunctionSpace(mesh, ("CG", k))._cpp_object
        G = create_discrete_gradient(V._cpp_object, V_CG)

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
    A.x.scatter_forward()
    ksp.view()
    t.stop()
    return {"A": A,
            "ndofs": A.vector.getSize(),
            "solve_time": t.elapsed()[0],
            "iterations": ksp.its}


def compute_B(A: Function):
    """Computes the magnetic field (using interpolation).
    Args:
        A: Magnetic vector potential
    Returns:
        B: The magnetic flux density
    """
    mesh = A.function_space.mesh
    k = A.function_space.ufl_element().degree()
    V = FunctionSpace(mesh, ("RT", k))
    B = Function(V)
    curl_A = Expression(curl(A), V.element.interpolation_points)
    B.interpolate(curl_A)
    return B
