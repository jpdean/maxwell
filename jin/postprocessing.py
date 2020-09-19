from dolfinx.io import XDMFFile
from mpi4py import MPI


def save(v, mesh, file_name):
    with XDMFFile(MPI.COMM_WORLD, file_name, "w") as file:
        file.write_mesh(mesh)
        file.write_function(v)
