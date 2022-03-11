# Solver for the problem details in [1]. NOTE: If beta = 0, problem is
# only semi-definite and right hand side must satisfy compatability
# conditions. This can be ensured by prescribing an impressed magnetic
# field, see[2]

# References:
# [1] https://hypre.readthedocs.io/en/latest/solvers-ams.html
# [2] Oszkar Biro, "Edge element formulations of eddy current problems"

import numpy as np
from dolfinx.common import Timer
from dolfinx.cpp.fem.petsc import create_discrete_gradient
from dolfinx.fem import (Expression, Function, FunctionSpace, form, petsc,
                         locate_dofs_topological, dirichletbc, Constant)
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, curl, dx, inner
from ufl.core.expr import Expr
from typing import Dict


def solve_problem(mesh: Mesh, k: int, alpha: np.float64, beta: np.float64,
                  f: Expr, boundary_marker, u_bc_ufl,
                  preconditioner: str = "ams", jit_params: Dict = None,
                  form_compiler_params: Dict = None):
    """Solves a magnetostatic problem.
    Args:
        mesh: the mesh
        k: order of space
        alpha: Coefficient (see [1])
        beta: Coefficient (see [1])
        f: impressed magnetic field
        preconditioner: "ams" or "gamg"
        form_compiler_params: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_params:See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
    Returns:
        (u: The computed solution
        {ndofs: number of degrees of freedom
         solve_time: time taked for solver alone
         iterations: number of solver iterations})
    """
    if form_compiler_params is None:
        form_compiler_params = {}
    if jit_params is None:
        jit_params = {}

    V = FunctionSpace(mesh, ("N1curl", k))

    u = TrialFunction(V)
    v = TestFunction(V)

    alpha = Constant(mesh, alpha)
    beta = Constant(mesh, beta)
    a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx,
             form_compiler_params=form_compiler_params, jit_params=jit_params)

    L = form(inner(f, v) * dx,
             form_compiler_params=form_compiler_params, jit_params=jit_params)

    u = Function(V)

    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(
        mesh, dim=tdim - 1, marker=boundary_marker)
    boundary_dofs = locate_dofs_topological(
        V, entity_dim=tdim - 1, entities=boundary_facets)
    u_bc_expr = Expression(u_bc_ufl, V.element.interpolation_points)
    u_bc = Function(V)
    u_bc.interpolate(u_bc_expr)
    bc = dirichletbc(u_bc, boundary_dofs)

    # TODO More steps needed here for Dirichlet boundaries
    A = petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    # Create solver
    ksp = PETSc.KSP().create(mesh.comm)

    # Set solver options
    ksp.setType("cg")
    ksp.setTolerances(rtol=1.0e-8)

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

        cvec_0 = Function(V)
        cvec_0.interpolate(lambda x: np.vstack((np.ones_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_1 = Function(V)
        cvec_1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.ones_like(x[0]),
                                                np.zeros_like(x[0]))))
        cvec_2 = Function(V)
        cvec_2.interpolate(lambda x: np.vstack((np.zeros_like(x[0]),
                                                np.zeros_like(x[0]),
                                                np.ones_like(x[0]))))
        pc.setHYPRESetEdgeConstantVectors(cvec_0.vector,
                                          cvec_1.vector,
                                          cvec_2.vector)

        # If we are dealing with a zero conductivity problem (no mass
        # term),need to tell the preconditioner
        if np.isclose(beta.value, 0):
            pc.setHYPRESetBetaPoissonMatrix(None)

        # NOTE Can set more ams options like:
        # opts = PETSc.Options()
        # opts["pc_hypre_ams_cycle_type"] = 13
    elif preconditioner == "gamg":
        pc.setType("gamg")

    # Set matrix operator
    ksp.setOperators(A)

    ksp.setMonitor(lambda ksp, its, rnorm: print(
        "Iteration: {}, rel. residual: {}".format(its, rnorm)))

    ksp.setFromOptions()

    # Compute solution
    t = Timer()
    t.start()
    ksp.solve(b, u.vector)
    u.x.scatter_forward()
    ksp.view()
    t.stop()
    return (u, {"ndofs": u.vector.getSize(),
                "solve_time": t.elapsed()[0],
                "iterations": ksp.its})
