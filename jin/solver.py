# Solves the generic equation given on pg 131 of [1] but
# assuming \alpha_x = \alpha_y = \alpha

# References:
# [1] The Finite Element Method in Electromagnetics by Jin

from problems import Problem
from meshes import create_unit_square_mesh
import dolfinx
from dolfinx import FunctionSpace, Function
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, DirichletBC
from ufl import TrialFunction, TestFunction, inner, grad, dx
import numpy as np


# TODO Remove
def boundary_marker(x):
    left = np.isclose(x[0], 0)
    right = np.isclose(x[0], 1)
    bottom = np.isclose(x[1], 0)
    top = np.isclose(x[1], 1.1)

    l_r = np.logical_or(left, right)
    b_t = np.logical_or(bottom, top)
    l_r_b_t = np.logical_or(l_r, b_t)
    return l_r_b_t


# TODO Remove
def bound_cond(x):
    # TODO Make this use meshtags
    values = np.zeros((1, x.shape[1]))
    return values


def solve(problem):
    V = FunctionSpace(problem.get_mesh(), ("Lagrange", problem.get_k()))

    u = TrialFunction(V)
    v = TestFunction(V)

    # TODO Use meshtags
    u_bc = Function(V)
    u_bc.interpolate(bound_cond)
    facets = locate_entities_boundary(mesh, 1, boundary_marker)
    bdofs = locate_dofs_topological(V, 1, facets)
    bc = DirichletBC(u_bc, bdofs)

    a = inner(1 / problem.get_alpha() * grad(u), grad(v)) * dx \
        + inner(problem.get_beta() * u, v) * dx
    L = inner(problem.get_f(), v) * dx

    u = Function(V)
    dolfinx.solve(a == L, u, bc, petsc_options={"ksp_type": "preonly",
                                                "pc_type": "lu"})
    return u


h = 0.1
mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
k = 1
alpha_dict = {1: 1}
beta_dict = {1: 0}
f = 1
problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f)
u = solve(problem)
