# TODO Reference Jorgen's post


from dolfinx import FunctionSpace, Function


class Problem:
    def __init__(self, mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict, f):
        self._mesh = mesh
        self._cell_mt = cell_mt
        self._facet_mt = facet_mt
        self._k = k
        self._alpha = self._create_dg0_func(alpha_dict)
        self._beta = self._create_dg0_func(beta_dict)
        self._f = f

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
