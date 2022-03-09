from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Tuple

import numpy as np
from dolfinx.mesh import Mesh, create_unit_cube
from mpi4py import MPI
from ufl import SpatialCoordinate, as_vector, cos, pi
from ufl.core.expr import Expr

from solver import compute_B, solve_problem
from util import L2_norm, save_function


def create_problem_1(h: np.float64, mu: np.float64) -> Tuple[Mesh, Expr, Expr]:
    """Create setup for Maxwell problem

    Args:
        h: Diameter of cells in the mesh
        mu: Permability

    Returns:
        Tuple with the mesh, the impressed magnetic field and the exact magnetic field.
        The two last outputs are ufl-expressions.
    """
    n = int(round(1 / h))
    mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    T_0 = as_vector((- pi * cos(x[2] * pi) / mu,
                     - pi * cos(x[0] * pi) / mu,
                     - pi * cos(x[1] * pi) / mu))
    # FIXME Get ufl to compute f and B from A for checking solution
    B_e = as_vector((- pi * cos(x[2] * pi),
                     - pi * cos(x[0] * pi),
                     - pi * cos(x[1] * pi)))
    return mesh, T_0, B_e


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h", default=1./16, type=np.float64, dest="h",
                        help="Resolution of Mesh")
    parser.add_argument("--k", default=1, type=int, dest="k",
                        help="Degree of H(curl) function space")
    parser.add_argument("--prec", default="ams", type=str, dest="prec",
                        help="Preconditioner used for solving the Maxwell problem",
                        choices=["ams", "gamg"])
    parser.add_argument("--mu", default=1., type=np.float64, dest="mu",
                        help="Permability")
    args = parser.parse_args()

    k = args.k
    h = args.h
    mu = args.mu
    prec = args.prec

    mesh, T_0, B_e = create_problem_1(h, mu)

    results = solve_problem(mesh, k, mu, T_0, prec)
    A = results["A"]
    A.name = "A"
    save_function(A, "A.xdmf")
    B = compute_B(A, k - 1)
    B.name = "B"
    save_function(B, "B.xdmf")
    e = L2_norm(B - B_e)
    print(f"L2-norm of error in B = {e}")
