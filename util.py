import numpy as np
from dolfinx.fem import (assemble_scalar, form, VectorFunctionSpace,
                         Function)
from mpi4py import MPI
from ufl import dx, inner
from ufl.core.expr import Expr
from dolfinx.io import VTXWriter


def save_function(v, filename):
    """Save a function v to file. The function is interpolated into a
    discontinuous Lagrange space so that functions in Nedelec and
    Raviart-Thomas spaces can be visualised exactly"""
    mesh = v.function_space.mesh
    k = v.function_space.ufl_element().degree()
    # NOTE: Alternatively could pass this into function so it doesn't need
    # to be created each time
    W = VectorFunctionSpace(mesh, ("Discontinuous Lagrange", k))
    w = Function(W)
    w.name = v.name
    w.interpolate(v)
    with VTXWriter(mesh.comm, filename, [w]) as file:
        file.write(0.0)


def L2_norm(v: Expr):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(
        assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM))
