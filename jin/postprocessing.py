from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
from ufl import inner, dx
from dolfinx.fem.assemble import assemble_scalar


def save(v, mesh, file_name):
    with XDMFFile(MPI.COMM_WORLD, file_name, "w") as file:
        file.write_mesh(mesh)
        file.write_function(v)


# TODO Add test for this function
def L2_norm(v):
    return np.sqrt(MPI.COMM_WORLD.allreduce(assemble_scalar(inner(v, v) * dx),
                                            op=MPI.SUM))
