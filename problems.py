# References:
# [1] https://hypre.readthedocs.io/en/latest/solvers-ams.html

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Tuple

import numpy as np
from dolfinx.mesh import Mesh, create_unit_cube
from mpi4py import MPI
from ufl import SpatialCoordinate, as_vector, cos, pi, curl
from ufl.core.expr import Expr

from solver import solve_problem
from util import L2_norm, save_function


def create_problem_0(h: np.float64,
                     alpha: np.float64,
                     beta: np.float64) -> Tuple[Mesh, Expr, Expr]:
    """Create setup for Maxwell problem

    Args:
        h: Diameter of cells in the mesh
        alpha: Coefficient (see [1])
        beta: Coefficient (see [1])

    Returns:
        Tuple containing the mesh, exact solution, right hand side, and
        a marker for the boundary.
    """
    n = int(round(1 / h))
    mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n)
    x = SpatialCoordinate(mesh)
    u_e = as_vector((cos(pi * x[1]), cos(pi * x[2]), cos(pi * x[0])))
    f = curl(curl(u_e)) + u_e

    def boundary_marker(x):
        """Marker function for the boundary of a unit cube"""
        # Collect boundaries perpendicular to each coordinate axis
        boundaries = [
            np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0))
            for i in range(3)]
        return np.logical_or(np.logical_or(boundaries[0],
                                           boundaries[1]),
                             boundaries[2])
    return mesh, u_e, f, boundary_marker


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h", default=1. / 16, type=np.float64, dest="h",
                        help="Resolution of Mesh")
    parser.add_argument("--k", default=1, type=int, dest="k",
                        help="Degree of H(curl) function space")
    parser.add_argument("--prec", default="ams", type=str, dest="prec",
                        help="Preconditioner used for solving the Maxwell problem",
                        choices=["ams", "gamg"])
    parser.add_argument("--alpha", default=1., type=np.float64, dest="alpha",
                        help="Alpha coefficient")
    parser.add_argument("--beta", default=1., type=np.float64, dest="beta",
                        help="Beta coefficient")
    args = parser.parse_args()

    k = args.k
    h = args.h
    alpha = args.alpha
    beta = args.beta
    prec = args.prec

    mesh, u_e, f, boundary_marker = create_problem_0(h, alpha, beta)
    petsc_options = {"pc_hypre_ams_cycle_type": 7,
                     "pc_hypre_ams_tol": 1e-8,
                     "ksp_atol": 1e-8, "ksp_rtol": 1e-8,
                     "ksp_type": "gmres"}
    u = solve_problem(mesh, k, alpha, beta, f, boundary_marker, u_e, prec, petsc_options=petsc_options)[0]
    u.name = "A"
    save_function(u, "u.bp")
    e = L2_norm(u - u_e)
    print(f"||u - u_e||_L^2(Omega) = {e}")
