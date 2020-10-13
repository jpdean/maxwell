from ufl import SpatialCoordinate, as_vector, pi, cos
from dolfinx import UnitCubeMesh
from mpi4py import MPI
from solver import solve_problem, compute_B
from util import save_function, L2_norm


# FIXME Get ufl to compute f and B from A for checking solution
# TODO Make problem factory


class Problem():
    """Simple class representing a problem.
    Args:
        mesh: The mesh
        k: Degree of the function space
        mu: Permeability
        T_0: Impressed magnetic field (due to impressed current
             density)
        B_e: Exact solution for the B field (None if no exact
             solution)
        """
    def __init__(self, mesh, k, mu, T_0, B_e):
        self.mesh = mesh
        self.k = k
        self.mu = mu
        self.T_0 = T_0
        self.B_e = B_e


def create_problem_1(h, k, mu):
    """Create problem 1 from man_sol.py
    Args:
        h: Characteristic cell size
        k: Degree of function space
        mu: Permeability
    """
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


if __name__ == "__main__":
    k = 1
    h = 1 / 32
    mu = 1
    problem = create_problem_1(h, k, mu)
    A = solve_problem(problem)
    save_function(A, problem.mesh, "A.xdmf")
    B = compute_B(A, k - 1, problem.mesh)
    save_function(B, problem.mesh, "B.xdmf")
    e = L2_norm(B - problem.B_e)
    print(f"L2-norm of error in B = {e}")
