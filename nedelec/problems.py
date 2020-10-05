from ufl import SpatialCoordinate, as_vector, pi, cos, sin
from dolfinx import UnitCubeMesh, Constant
from mpi4py import MPI
from solver_conducting import solve_problem
from solver import compute_B
from util import save_function, L2_norm
import numpy as np


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory


class Problem():
    def __init__(self, mesh, k, mu, sigma, omega, T_0, B_e, boundary):
        self.mesh = mesh
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.omega = omega
        self.T_0 = T_0
        self.B_e = B_e
        self.boundary = boundary


def cube_bound_marker(x):
    left = np.isclose(x[0], 0)
    right = np.isclose(x[0], 1)
    bottom = np.isclose(x[1], 0)
    top = np.isclose(x[1], 1)
    back = np.isclose(x[2], 0)
    front = np.isclose(x[2], 1)

    l_r = np.logical_or(left, right)
    b_t = np.logical_or(bottom, top)
    b_f = np.logical_or(back, front)

    l_r_b_t = np.logical_or(l_r, b_t)

    return np.logical_or(l_r_b_t, b_f)


def create_problem_1(h, k, mu, sigma, omega):
    n = round(1 / h)
    mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    # FIXME This isn't currently being used!
    T_0 = as_vector((- pi * cos(x[2] * pi) / mu,
                     - pi * cos(x[0] * pi) / mu,
                     - pi * cos(x[1] * pi) / mu))
    B_e = None
    return Problem(mesh, k, mu, sigma, omega, T_0, B_e, cube_bound_marker)


if __name__ == "__main__":
    k = 1
    h = 1 / 16
    mu = 1
    sigma = 1
    omega = 1
    problem = create_problem_1(h, k, mu, sigma, omega)
    A, V = solve_problem(problem)
    save_function(A, problem.mesh, "A.xdmf")
    save_function(V, problem.mesh, "V.xdmf")
    B = compute_B(A, k - 1, problem.mesh)
    save_function(B, problem.mesh, "B.xdmf")
    e = L2_norm(B - problem.B_e)
    print(f"L2-norm of error in B = {e}")
    x = SpatialCoordinate(problem.mesh)
    e = L2_norm(V - x[0])
    print(f"L2-norm of error in V = {e}")
