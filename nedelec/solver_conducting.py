from dolfinx import Function,  FunctionSpace, solve, DirichletBC
from ufl import (TrialFunctions, TestFunctions, inner, dx, curl,
                 FiniteElement, grad, as_vector, pi, sin, SpatialCoordinate)
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological
import numpy as np


def solve_problem(problem):
    A_fe = FiniteElement("N1curl", problem.mesh.ufl_cell(), 1)
    V_fe = FiniteElement("Lagrange", problem.mesh.ufl_cell(), 1)
    W = FunctionSpace(problem.mesh, A_fe * V_fe)

    W0 = W.sub(0).collapse()
    A_d = Function(W0)
    A_d.vector.set(0.0)
    d = problem.mesh.topology.dim
    facets = locate_entities_boundary(problem.mesh, d - 1, problem.boundary)
    dofs = locate_dofs_topological((W.sub(0), W0), d - 1, facets)
    bc1 = DirichletBC(A_d, dofs,  W.sub(0))

    W1 = W.sub(1).collapse()
    V_bc = Function(W1)
    # V_bc.vector.set(0.0)
    V_bc.interpolate(lambda x: x[0])
    facets = locate_entities_boundary(problem.mesh, d - 1, problem.boundary)
    dofs = locate_dofs_topological((W.sub(1), W1), d - 1, facets)
    bc2 = DirichletBC(V_bc, dofs, W.sub(1))

    (A, V) = TrialFunctions(W)
    (w0, w1) = TestFunctions(W)

    mu = problem.mu
    sigma = problem.sigma
    omega = problem.omega
    a_1 = inner(1 / mu * curl(A), curl(w0)) * dx + \
        inner(1j * omega * sigma * A, w0) * dx + \
        inner(1j * omega * sigma * grad(V), w0) * dx
    a_2 = inner(1j * omega * sigma * A, grad(w1)) * dx + \
        inner(1j * omega * sigma * grad(V), grad(w1)) * dx
    a = a_1 + a_2

    T_0 = problem.T_0
    # TODO REMOVE
    x = SpatialCoordinate(problem.mesh)
    J_0 = as_vector((((1j*mu*omega*sigma*(sin(x[1]*pi)*sin(x[2]*pi) + 1) + 2*pi**2*sin(x[1]*pi)*sin(x[2]*pi))/mu),
                    0,
                    0))
    # L = inner(T_0, curl(w0)) * dx
    L = inner(J_0, w0) * dx

    X = Function(W)
    # NOTE That A is not unique because of the nullspace of the curl operator
    # i.e. curl(grad(\phi)) = 0 for any \phi, so for any A that is a solution,
    # A + grad(\phi) is also a solution. Hence, must use an iterative solver.
    solve(a == L, X, [bc1, bc2], petsc_options={"ksp_type": "cg",
                                                "pc_type": "icc",
                                                "ksp_rtol": 1e-8,
                                                "ksp_monitor": None})
    A = X.sub(0).collapse()
    V = X.sub(1).collapse()
    return A, V
