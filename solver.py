# Solver for the problem details in [1]. NOTE: If beta = 0, problem is
# only semi-definite and right hand side must satisfy compatability
# conditions. This can be ensured by prescribing an impressed magnetic
# field, see[2]

# TODO Could construct alpha and beta poisson matrices and pass to AMS,
# see [1] for details.

# References:
# [1] https://hypre.readthedocs.io/en/latest/solvers-ams.html
# [2] Oszkar Biro, "Edge element formulations of eddy current problems"

from typing import Dict
from util import save_function
import numpy as np
from dolfinx.common import Timer, timing
from dolfinx.cpp.fem.petsc import (discrete_gradient,
                                   interpolation_matrix)
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         dirichletbc, form, locate_dofs_topological, petsc)
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, VectorElement, curl, dx, inner
from ufl.core.expr import Expr


def solve_problem(mesh: Mesh, k: int, alpha: np.float64, beta: np.float64,
                  f: Expr, boundary_marker, u_bc_ufl,
                  preconditioner: str = "ams", jit_options: Dict = None,
                  form_compiler_options: Dict = None, petsc_options: Dict = None):
    """Solves a magnetostatic problem.
    Args:
        mesh: the mesh
        k: order of space
        alpha: Coefficient (see [1])
        beta: Coefficient (see [1])
        f: impressed magnetic field
        preconditioner: "ams" or "gamg"
        form_compiler_options: See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        jit_options:See :func:`ffcx_jit <dolfinx.jit.ffcx_jit>`
        petsc_options: Parameters that is passed to the linear algebra backend
          PETSc. For available choices for the 'petsc_options' kwarg, see the `PETSc-documentation
          <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`
    Returns:
        (u: The computed solution
        {ndofs: number of degrees of freedom
         solve_time: time taked for solver alone
         iterations: number of solver iterations})
    """
    if form_compiler_options is None:
        form_compiler_options = {}
    if jit_options is None:
        jit_options = {}
    if petsc_options is None:
        petsc_options = {}

    V = FunctionSpace(mesh, ("N1curl", k))
    ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    u = TrialFunction(V)
    v = TestFunction(V)

    alpha = Constant(mesh, alpha)
    beta = Constant(mesh, beta)
    a = form(inner(alpha * curl(u), curl(v)) * dx + inner(beta * u, v) * dx,
             form_compiler_options=form_compiler_options, jit_options=jit_options)

    L = form(inner(f, v) * dx,
             form_compiler_options=form_compiler_options, jit_options=jit_options)

    u = Function(V)

    tdim = mesh.topology.dim
    boundary_facets = locate_entities_boundary(
        mesh, dim=tdim - 1, marker=boundary_marker)
    boundary_dofs = locate_dofs_topological(
        V, entity_dim=tdim - 1, entities=boundary_facets)
    u_bc_expr = Expression(u_bc_ufl, V.element.interpolation_points())
    with Timer(f"~{k}, {ndofs}: BC interpolation"):
        u_bc = Function(V)
        u_bc.interpolate(u_bc_expr)
    bc = dirichletbc(u_bc, boundary_dofs)

    # TODO More steps needed here for Dirichlet boundaries
    with Timer(f"~{k}, {ndofs}: Assemble LHS and RHS"):
        A = petsc.assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

    # Create solver
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOptionsPrefix(f"ksp_{id(ksp)}")
    # ksp.setNormType(ksp.NormType.NORM_PRECONDITIONED)

    pc = ksp.getPC()
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for option, value in petsc_options.items():
        opts[option] = value
    opts.prefixPop()
    if preconditioner == "ams":
        # Based on: https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/curl-curl/demo_curl-curl.py # noqa: E501
        pc.setType("hypre")
        pc.setHYPREType("ams")

        # Build discrete gradient
        with Timer(f"~{k}, {ndofs}: Build discrete gradient"):
            V_CG = FunctionSpace(mesh, ("CG", k))._cpp_object
            G = discrete_gradient(V_CG, V._cpp_object)
            G.assemble()
            pc.setHYPREDiscreteGradient(G)

        if k == 1:
            with Timer(f"~{k}, {ndofs}: Build EdgeConstantVectors"):
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
        else:
            # Create interpolation operator
            with Timer(f"~{k}, {ndofs}: Build interpolation matrix"):
                Vec_CG = FunctionSpace(mesh, VectorElement("CG", mesh.ufl_cell(), k))
                Pi = interpolation_matrix(Vec_CG._cpp_object, V._cpp_object)
                Pi.assemble()

                # Attach discrete gradient to preconditioner
                pc.setHYPRESetInterpolations(mesh.geometry.dim, None, None, Pi, None)

        # If we are dealing with a zero conductivity problem (no mass
        # term),need to tell the preconditioner
        if np.isclose(beta.value, 0):
            pc.setHYPRESetBetaPoissonMatrix(None)

    elif preconditioner == "gamg":
        pc.setType("gamg")

    # Set matrix operator
    ksp.setOperators(A)

    def monitor(ksp, its, rnorm):
        if mesh.comm.rank == 0:
            print("Iteration: {}, rel. residual: {}".format(its, rnorm))
    ksp.setMonitor(monitor)
    ksp.setFromOptions()
    # Compute solution
    with Timer(f"~{k}, {ndofs}: Solve Problem"):
        ksp.solve(b, u.vector)
        u.x.scatter_forward()

    reason = ksp.getConvergedReason()
    print(f"Convergence reason {reason}")
    if reason < 0:
        u.name = "A"
        save_function(u, "error.bp")
        raise RuntimeError("Solver did not converge. Output at error.bp")
    # ksp.view()
    return (u, {"ndofs": ndofs,
                "solve_time": timing(f"~{k}, {ndofs}: Solve Problem")[1],
                "iterations": ksp.its})
