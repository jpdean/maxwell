from dolfinx import (Function, solve)
import numpy as np
from mpi4py import MPI
from dolfinx.fem import assemble_scalar
from ufl import (TrialFunction, TestFunction, inner, dx)
from dolfinx.io import XDMFFile
from dolfinx.cpp.io import extract_local_entities
from dolfinx.cpp.graph import AdjacencyList_int32
from dolfinx.mesh import create_meshtags


def project(f, V):
    """Projects the function f onto the space V
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = inner(f, v) * dx

    u = Function(V)
    solve(a == L, u, [], petsc_options={"ksp_type": "cg"})
    return u


def save_function(v, mesh, filename):
    with XDMFFile(MPI.COMM_WORLD, filename, "w") as f:
        f.write_mesh(mesh)
        f.write_function(v)


def L2_norm(v):
    """Computes the L2-norm of v
    """
    return np.sqrt(MPI.COMM_WORLD.allreduce(assemble_scalar(inner(v, v) * dx),
                                            op=MPI.SUM))


def entity_from_pygmsh_mesh(pygmsh_mesh, entity_type):
    entities = np.vstack(
        [cell.data for cell in pygmsh_mesh.cells
         if cell.type == entity_type])
    return entities


# Simplify this method
def entity_mesh_tags(mesh, pygmsh_mesh, entity_type):
    entities = entity_from_pygmsh_mesh(pygmsh_mesh, entity_type)
    entity_values = np.hstack(
        [pygmsh_mesh.cell_data_dict["gmsh:physical"][key] for key in
         pygmsh_mesh.cell_data_dict["gmsh:physical"].keys()
         if key == entity_type])
    if entity_type == "tetra":
        entity_dim = 3
    elif entity_type == "triangle":
        entity_dim = 2
    local_entities, local_values = \
        extract_local_entities(mesh, entity_dim, entities, entity_values)
    mt = create_meshtags(mesh, entity_dim,
                         AdjacencyList_int32(local_entities),
                         np.int32(local_values))
    return mt
