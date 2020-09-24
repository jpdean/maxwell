# Solves the generic equation given on pg 131 of [1] but
# assuming \alpha_x = \alpha_y = \alpha

# References:
# [1] The Finite Element Method in Electromagnetics by Jin

from problems import Problem
from meshes import create_unit_square_mesh
import dolfinx
from dolfinx import FunctionSpace, Function
from dolfinx.fem import locate_dofs_topological, DirichletBC
from ufl import TrialFunction, TestFunction, inner, grad, dx
import numpy as np
from postprocessing import save

# TODO Add Neumann BCs


def solve(problem):
    V = FunctionSpace(problem.get_mesh(), ("Lagrange", problem.get_k()))

    u = TrialFunction(V)
    v = TestFunction(V)

    # FIXME There is almost certainly a much better/more efficient way
    # to do this
    bcs = []
    for tag, u_d in problem.get_bc_dict().items():
        u_bc = Function(V)
        u_bc.interpolate(u_d)
        facet_mt = problem.get_facet_mt()
        facets = []
        for i in range(len(facet_mt.indices)):
            if facet_mt.values[i] == tag:
                facets.append(facet_mt.indices[i])
        bdofs = locate_dofs_topological(V, 1, facets)
        bcs.append(DirichletBC(u_bc, bdofs))

    a = inner(1 / problem.get_alpha() * grad(u), grad(v)) * dx \
        + inner(problem.get_beta() * u, v) * dx
    L = inner(problem.get_f(), v) * dx

    u = Function(V)
    dolfinx.solve(a == L, u, bcs, petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
    return u


h = 0.1
mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
k = 1
alpha_dict = {1: 1}
beta_dict = {1: 0}
f = 1
bc_dict = {2: lambda x: np.zeros((1, x.shape[1])),
           3: lambda x: np.zeros((1, x.shape[1])),
           4: lambda x: np.zeros((1, x.shape[1])),
           5: lambda x: np.zeros((1, x.shape[1]))}
problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f,
                  bc_dict)
u = solve(problem)
save(u, mesh, "phi.xdmf")
