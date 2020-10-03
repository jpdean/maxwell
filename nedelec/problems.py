from ufl import SpatialCoordinate, as_vector, pi, cos
from dolfinx import UnitCubeMesh, FunctionSpace, Function
from mpi4py import MPI
from solver import solve_problem, compute_B
from util import save_function, L2_norm
import pygmsh
from dolfinx.mesh import create_mesh
from dolfinx.io import ufl_mesh_from_gmsh
from util import entity_from_pygmsh_mesh, entity_mesh_tags
import numpy as np


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory


class Problem():
    def __init__(self, mesh, k, mu, T_0, B_e, cell_mt, facet_mt):
        self.mesh = mesh
        self.k = k
        self.mu = mu
        self.T_0 = T_0
        self.B_e = B_e
        self.cell_mt = cell_mt
        self.facet_mt = facet_mt


def create_problem_1(h, k, mu):
    n = round(1 / h)
    mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    T_0 = as_vector((- pi * cos(x[2] * pi) / mu,
                     - pi * cos(x[0] * pi) / mu,
                     - pi * cos(x[1] * pi) / mu))
    B_e = as_vector((- pi * cos(x[2] * pi),
                     - pi * cos(x[0] * pi),
                     - pi * cos(x[1] * pi)))
    return Problem(mesh, k, mu, T_0, B_e)


def create_problem_2(h, k):
    geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
    domain = geom.add_box([0, 0, 0], [1, 1, 1])
    magnet = geom.add_cylinder([0.5, 0.5, 0.3], [0, 0, 0.4], 0.1)
    geom.boolean_fragments([domain], [magnet])
    geom.add_raw_code("Physical Volume(1) = {14};")
    geom.add_raw_code("Physical Volume(2) = {13};")
    geom.add_raw_code("Physical Surface(1) = {10,11,12,13,14,15};")
    pygmsh_mesh = pygmsh.generate_mesh(geom, geo_filename="geom.geo")

    x = pygmsh_mesh.points
    cells = entity_from_pygmsh_mesh(pygmsh_mesh, "tetra")
    mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                       ufl_mesh_from_gmsh("tetra", 3))
    mesh.topology.create_connectivity_all()
    cell_mt = entity_mesh_tags(mesh, pygmsh_mesh, "tetra")
    facet_mt = entity_mesh_tags(mesh, pygmsh_mesh, "triangle")

    V = FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
    mu_0 = 4 * np.pi * 1e-7
    mu_magnet = 10 * mu_0
    mu = Function(V)
    mu_dict = {1: mu_0, 2: mu_magnet}
    # FIXME In newer versions of FEnicS-X, use V.dim not V.dim()
    for i in range(V.dim()):
        mu.vector.setValueLocal(i, mu_dict[cell_mt.values[i]])

    x = SpatialCoordinate(mesh)
    T_0 = as_vector((0, 0, 1))
    B_e = None
    return Problem(mesh, k, mu, T_0, B_e, cell_mt, facet_mt)


if __name__ == "__main__":
    # k = 1
    # h = 1 / 32
    # mu = 1
    # problem = create_problem_1(h, k, mu)
    # A = solve_problem(problem)
    # save_function(A, problem.mesh, "A.xdmf")
    # B = compute_B(A, k - 1, problem.mesh)
    # save_function(B, problem.mesh, "B.xdmf")
    # e = L2_norm(B - problem.B_e)
    # print(f"L2-norm of error in B = {e}")

    problem = create_problem_2(1 / 64, 1)
    A = solve_problem(problem)
    save_function(A, problem.mesh, "A.xdmf")
    B = compute_B(A, problem.k - 1, problem.mesh)
    save_function(B, problem.mesh, "B.xdmf")
