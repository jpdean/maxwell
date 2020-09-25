# TODO Reference Jorgen's post on his website

# References:
# [1] The Finite Element Method in Electromagnetics by Jin

from dolfinx import FunctionSpace, Function, Constant
from meshes import (create_unit_square_mesh,
                    create_shielded_microstrip_line_mesh)
import numpy as np
from ufl import sin, cos, div, grad, pi, SpatialCoordinate


class Problem:
    def __init__(self, mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f,
                 bc_dict):
        self._mesh = mesh
        self._cell_mt = cell_mt
        self._facet_mt = facet_mt
        self._k = k
        self._alpha = self._create_dg0_func(alpha_dict)
        self._beta = self._create_dg0_func(beta_dict)
        self._f = f
        self._bc_dict = bc_dict

    def _create_dg0_func(self, dict):
        V = FunctionSpace(self._mesh, ("Discontinuous Lagrange", 0))
        v = Function(V)

        for i in range(V.dim):
            v.vector.setValueLocal(i, dict[self._cell_mt.values[i]])
        return v

    def get_mesh(self):
        return self._mesh

    def get_k(self):
        return self._k

    def get_alpha(self):
        return self._alpha

    def get_beta(self):
        return self._beta

    def get_f(self):
        return self._f

    def get_bc_dict(self):
        return self._bc_dict

    def get_facet_mt(self):
        return self._facet_mt


class ProblemFactory:
    @staticmethod
    def create_Poisson_problem_1(h=0.1, k=1):
        mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
        alpha_dict = {1: 1}
        beta_dict = {1: 0}
        f = 1
        bc_dict = {}
        bc_dict["dirichlet"] = {4: lambda x: np.zeros((1, x.shape[1])),
                                5: lambda x: np.zeros((1, x.shape[1]))}
        bc_dict["neumann"] = {2: 0.5, 3: -0.25}
        problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f,
                          bc_dict)
        return problem

    @staticmethod
    def create_Poisson_problem_2(h=0.1, k=1):
        mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
        alpha_dict = {1: 1}
        beta_dict = {1: 0}
        x = SpatialCoordinate(mesh)
        u = sin(pi * x[0]) * cos(pi * x[1])
        f = - div(grad(u))
        bc_dict = {}
        bc_dict["dirichlet"] = {2: lambda x: np.sin(np.pi * x[0]),
                                3: lambda x: np.zeros((1, x.shape[1])),
                                4: lambda x: - np.sin(np.pi * x[0]),
                                5: lambda x: np.zeros((1, x.shape[1]))}
        bc_dict["neumann"] = {}
        problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f,
                          bc_dict)
        return problem

    # Problem based on pg 272 of [1]
    @staticmethod
    def create_shielded_microstrip_line_problem(h=0.1, k=1):
        mesh, cell_mt, facet_mt = create_shielded_microstrip_line_mesh(h)
        eps_r = 8.8
        eps_0 = 8.85418782e-12
        alpha_dict = {1: eps_r * eps_0, 2: eps_0}
        beta_dict = {1: 0, 2: 0}
        rho = Constant(mesh, 0)  # FIXME Use constant
        f = rho / eps_0
        bc_dict = {}
        bc_dict["dirichlet"] = {1: lambda x: np.zeros((1, x.shape[1])),
                                2: lambda x: np.ones((1, x.shape[1]))}
        bc_dict["neumann"] = {3: Constant(mesh, 0)}  # FIXME Use constant
        problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f,
                          bc_dict)
        return problem
