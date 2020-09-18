# TODO Reference Jorgen's post


from dolfinx import FunctionSpace, Function


class Problem:
    def __init__(self, mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict):
        self._mesh = mesh
        self._cell_mt = cell_mt
        self._facet_mt = facet_mt
        self._k = k
        self._alpha = self._create_dg0_func(alpha_dict)
        self._beta = self._create_dg0_func(beta_dict)

    def _create_dg0_func(self, dict):
        V = FunctionSpace(self._mesh, ("Discontinuous Lagrange", 0))
        v = Function(V)

        for i in range(V.dim):
            v.setValueLocal(i, dict[self._cell_mt[i]])
        return v
