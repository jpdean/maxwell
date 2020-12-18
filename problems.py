from ufl import SpatialCoordinate, as_vector, pi, cos
from dolfinx import UnitCubeMesh
from mpi4py import MPI
from solver import solve_problem, compute_B
from util import save_function, L2_norm


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory
if __name__ == "__main__":
    # Space degree
    k = 1
    # Number of elements in each direction
    n = 4
    # Permeability
    mu = 1

    mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    T_0 = as_vector((- pi * cos(x[2] * pi) / mu,
                     - pi * cos(x[0] * pi) / mu,
                     - pi * cos(x[1] * pi) / mu))
    # TODO Compute from T_0 with UFL
    B_e = as_vector((- pi * cos(x[2] * pi),
                     - pi * cos(x[0] * pi),
                     - pi * cos(x[1] * pi)))

    A = solve_problem(mesh, k, mu, T_0)
    save_function(A, mesh, "A.xdmf")
    B = compute_B(A, k - 1, mesh)
    save_function(B, mesh, "B.xdmf")
    e = L2_norm(B - B_e)
    print(f"L2-norm of error in B = {e}")
