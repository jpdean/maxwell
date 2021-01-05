from dolfinx import Function
import numpy as np
from mpi4py import MPI
from dolfinx.fem import assemble_scalar, LinearProblem
from ufl import (TrialFunction, TestFunction, inner, dx)
from dolfinx.io import XDMFFile


def project(f, V):
    """Projects the function f onto the space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    problem = LinearProblem(a, L, petsc_options={"ksp_type": "cg"})
    u_h = problem.solve()
    return u_h


def save_function(v, mesh, filename):
    """Save a function v to xdmf"""
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as f:
        f.write_mesh(mesh)
        f.write_function(v)


def L2_norm(v):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(assemble_scalar(inner(v, v) * dx),
                                            op=MPI.SUM))
