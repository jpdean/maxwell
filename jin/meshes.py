# References:
# [1] The Finite Element Method in Electromagnetics by Jin

import pygmsh
import numpy
from dolfinx.io import ufl_mesh_from_gmsh
from dolfinx.cpp.io import extract_local_entities
from mpi4py import MPI
from dolfinx.mesh import (create_mesh, create_meshtags)
from dolfinx.cpp.graph import AdjacencyList_int32
import numpy as np

# TODO Make into class to hide helper methods


def create_unit_square_mesh(h):
    geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
    domain = geom.add_rectangle([0, 0, 0], 1, 1)
    geom.add_physical(domain, 1)
    geom.add_raw_code("Physical Curve(2) = {1};")
    geom.add_raw_code("Physical Curve(3) = {2};")
    geom.add_raw_code("Physical Curve(4) = {3};")
    geom.add_raw_code("Physical Curve(5) = {4};")
    pygmsh_mesh = pygmsh.generate_mesh(geom, prune_z_0=True)

    mesh = mesh_from_pygmsh_mesh(pygmsh_mesh)
    # Could just do create_connectivity(1, 0)
    mesh.topology.create_connectivity_all()

    cell_mt = entity_mesh_tags(mesh, pygmsh_mesh, "triangle")
    facet_mt = entity_mesh_tags(mesh, pygmsh_mesh, "line")
    return mesh, cell_mt, facet_mt


# Mesh for shielded microstrip line on pg 159 of [1]
def create_shielded_microstrip_line_mesh(h):
    geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
    domain = geom.add_rectangle([0, 0, 0], 1, 1)
    microstrip = geom.add_rectangle([0, 0.2, 0], 0.2, 0.05)
    domain_minus_microstrip = geom.boolean_difference([domain], [microstrip])
    dielectric = geom.add_rectangle([0, 0, 0], 1, 0.2)
    geom.boolean_fragments([domain_minus_microstrip], [dielectric])
    # Lower part
    geom.add_raw_code("Physical Surface(1) = {13};")
    # Upper part
    geom.add_raw_code("Physical Surface(2) = {14};")
    # Outer conductor boundary
    geom.add_raw_code("Physical Curve(1) = {4, 5, 9};")
    # Inner conductor boundary
    geom.add_raw_code("Physical Curve(2) = {1, 2, 7};")
    # Symmetry boundary
    geom.add_raw_code("Physical Curve(3) = {3, 10};")
    pygmsh_mesh = pygmsh.generate_mesh(geom, prune_z_0=True)

    mesh = mesh_from_pygmsh_mesh(pygmsh_mesh)
    # Could just do create_connectivity(1, 0)
    mesh.topology.create_connectivity_all()

    cell_mt = entity_mesh_tags(mesh, pygmsh_mesh, "triangle")
    facet_mt = entity_mesh_tags(mesh, pygmsh_mesh, "line")
    return mesh, cell_mt, facet_mt


def mesh_from_pygmsh_mesh(pygmsh_mesh):
    x = pygmsh_mesh.points
    cells = entity_from_pygmsh_mesh(pygmsh_mesh, "triangle")
    mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                       ufl_mesh_from_gmsh("triangle", 2))
    return mesh


def entity_from_pygmsh_mesh(pygmsh_mesh, entity_type):
    entities = numpy.vstack(
        [cell.data for cell in pygmsh_mesh.cells
         if cell.type == entity_type])
    return entities


# Simplify this method
def entity_mesh_tags(mesh, pygmsh_mesh, entity_type):
    entities = entity_from_pygmsh_mesh(pygmsh_mesh, entity_type)
    entity_values = numpy.hstack(
        [pygmsh_mesh.cell_data_dict["gmsh:physical"][key] for key in
         pygmsh_mesh.cell_data_dict["gmsh:physical"].keys()
         if key == entity_type])
    if entity_type == "triangle":
        entity_dim = 2
    elif entity_type == "line":
        entity_dim = 1
    local_entities, local_values = \
        extract_local_entities(mesh, entity_dim, entities, entity_values)
    mt = create_meshtags(mesh, entity_dim,
                         AdjacencyList_int32(local_entities),
                         np.int32(local_values))
    return mt
