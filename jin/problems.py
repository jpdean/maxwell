# TODO Reference Jorgen's post on his website


from dolfinx import FunctionSpace, Function
from meshes import create_unit_square_mesh
import numpy as np


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
    def create_Poisson_problem_1():
        h = 0.1
        mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
        k = 1
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
