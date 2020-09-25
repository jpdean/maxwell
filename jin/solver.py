# Solves the generic equation given on pg 131 of [1] but
# assuming \alpha_x = \alpha_y = \alpha. See residual form
# on pg 152. Note that they integrate over all element
# boundaries but mention later that only boundary elements
# contribute (at least for Lagrange elements).

# References:
# [1] The Finite Element Method in Electromagnetics by Jin

from problems import ProblemFactory
import dolfinx
from dolfinx import FunctionSpace, Function
from dolfinx.fem import locate_dofs_topological, DirichletBC
from ufl import TrialFunction, TestFunction, inner, grad, dx, Measure
from postprocessing import save

# TODO Add Robin BCs


def solve(problem):
    V = FunctionSpace(problem.get_mesh(), ("Lagrange", problem.get_k()))

    u = TrialFunction(V)
    v = TestFunction(V)

    # FIXME There is almost certainly a much better/more efficient way
    # to do this.
    bcs = []
    for tag, u_d in problem.get_bc_dict()["dirichlet"].items():
        u_bc = Function(V)
        u_bc.interpolate(u_d)
        facet_mt = problem.get_facet_mt()
        facets = []
        for i in range(len(facet_mt.indices)):
            if facet_mt.values[i] == tag:
                facets.append(facet_mt.indices[i])
        bdofs = locate_dofs_topological(V, 1, facets)
        bcs.append(DirichletBC(u_bc, bdofs))

    a = inner(problem.get_alpha() * grad(u), grad(v)) * dx \
        + inner(problem.get_beta() * u, v) * dx
    L = inner(problem.get_f(), v) * dx

    ds_mt = Measure("ds", subdomain_data=problem.get_facet_mt())
    for tag, g in problem.get_bc_dict()["neumann"].items():
        L += inner(g, v) * ds_mt(tag)

    u = Function(V)
    dolfinx.solve(a == L, u, bcs, petsc_options={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
    return u


# problem = ProblemFactory.create_Poisson_problem_2()
# u = solve(problem)
# save(u, problem.get_mesh(), "phi.xdmf")
