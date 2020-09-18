# TODO Reference Jorgen's post


import pygmsh
import numpy
from dolfinx.io import ufl_mesh_from_gmsh
from dolfinx.cpp.io import extract_local_entities
from mpi4py import MPI
from dolfinx.mesh import (create_mesh, create_meshtags)
from dolfinx.cpp.graph import AdjacencyList_int32
import numpy as np
from dolfinx import FunctionSpace


class Problem:
    def __init__(self, h, k, mesh, cell_mt, facet_mt, alpha, beta):
        self.h = h
        self.k = k
        self.mesh = mesh
        self.cell_mt = cell_mt
        self.facet_mt = facet_mt
        self.alpha = alpha
        self.beta = beta


class ProblemFactory:
    @staticmethod
    def create_poisson_problem(h, k):
        geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
        domain = geom.add_rectangle([0, 0, 0], 1, 1)
        geom.add_physical(domain, 1)
        geom.add_raw_code("Physical Curve(2) = {1};")
        geom.add_raw_code("Physical Curve(3) = {2};")
        geom.add_raw_code("Physical Curve(4) = {3};")
        geom.add_raw_code("Physical Curve(5) = {4};")
        pygmsh_mesh = pygmsh.generate_mesh(geom,
                                           prune_z_0=True,
                                           geo_filename="mesh.geo")
        x = pygmsh_mesh.points
        cells = numpy.vstack(
            [cell.data for cell in pygmsh_mesh.cells
             if cell.type == "triangle"])
        mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                           ufl_mesh_from_gmsh("triangle", 2))
        # TODO Simplify (e.g. create functions)
        cell_values = numpy.hstack(
            [pygmsh_mesh.cell_data_dict["gmsh:physical"][key] for key in
             pygmsh_mesh.cell_data_dict["gmsh:physical"].keys()
             if key == "triangle"])
        local_entities, local_values = \
            extract_local_entities(mesh, 2, cells, cell_values)
        cell_mt = create_meshtags(mesh, 2,
                                  AdjacencyList_int32(local_entities),
                                  np.int32(local_values))
        boundary_facets = numpy.vstack(
            [cell.data for cell in pygmsh_mesh.cells
             if cell.type == "line"])
        facet_values = numpy.hstack(
            [pygmsh_mesh.cell_data_dict["gmsh:physical"][key] for key in
             pygmsh_mesh.cell_data_dict["gmsh:physical"].keys()
             if key == "line"])
        local_entities, local_values = \
            extract_local_entities(mesh, 1, boundary_facets, facet_values)
        # Needed to create_meshtags, otherwise raises exception
        mesh.topology.create_connectivity(1, 0)
        facet_mt = create_meshtags(mesh, 1,
                                   AdjacencyList_int32(local_entities),
                                   np.int32(local_values))

        # \alpha and \beta
        V = FunctionSpace(self.mesh, ("Discontinuous Lagrange", 0))
        problem = Problem(h, k, mesh, cell_mt, facet_mt)


ProblemFactory.create_poisson_problem(0.1)
