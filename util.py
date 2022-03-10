import numpy as np
from dolfinx.fem import assemble_scalar, petsc, form
from dolfinx.io import XDMFFile
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, inner
from ufl.core.expr import Expr


def project(f, V):
    """Projects the function f onto the space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    problem = petsc.LinearProblem(a, L, petsc_options={"ksp_type": "cg"})
    u_h = problem.solve()
    return u_h


def save_function(v, filename):
    """Save a function v to xdmf"""
    mesh = v.function_space.mesh
    with XDMFFile(mesh.comm, filename, "w") as f:
        f.write_mesh(mesh)
        f.write_function(v)


def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    integral = form(inner(v, v) * dx)
    return np.sqrt(MPI.COMM_WORLD.allreduce(assemble_scalar(integral),
                                            op=MPI.SUM))
