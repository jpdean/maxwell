from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Tuple

import numpy as np
from dolfinx.mesh import Mesh, create_unit_cube
from mpi4py import MPI
from ufl import SpatialCoordinate, as_vector, cos, pi, curl
from ufl.core.expr import Expr

from solver import solve_problem
from util import L2_norm, save_function


def create_problem_1(h: np.float64, mu: np.float64) -> Tuple[Mesh, Expr, Expr]:
    """Create setup for Maxwell problem

    Args:
        h: Diameter of cells in the mesh
        mu: Permability

    Returns:
        Tuple with the mesh, the impressed magnetic field and the exact
        magnetic field. The two last outputs are ufl-expressions.
    """
    n = int(round(1 / h))
    mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    A_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
    f = curl(curl(A_e)) + A_e

    def boundary_marker(x):
        """Marker function for the boundary of a unit cube"""
        # Collect boundaries perpendicular to each coordinate axis
        boundaries = [
            np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
            for i in range(3)]
        return np.logical_or(np.logical_or(boundaries[0],
                                           boundaries[1]),
                             boundaries[2])
    return mesh, A_e, f, boundary_marker


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h", default=1. / 16, type=np.float64, dest="h",
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

    mesh, A_e, f, boundary_marker = create_problem_1(h, mu)

    A = solve_problem(mesh, k, mu, f, boundary_marker, A_e, prec)[0]
    A.name = "A"
    save_function(A, "A.bp")
    e = L2_norm(A - A_e)
    print(f"L2-norm of error in A = {e}")
