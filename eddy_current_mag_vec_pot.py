import pygmsh
import numpy as np
from dolfinx.mesh import (create_mesh, create_meshtags,
                          locate_entities_boundary)
from dolfinx.cpp.io import extract_local_entities
from dolfinx.cpp.graph import AdjacencyList_int32
from mpi4py import MPI
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.fem import DirichletBC, locate_dofs_topological
from dolfinx import (FunctionSpace, Function, Constant, solve,
                     VectorFunctionSpace)
import ufl
from ufl import grad, TrialFunction, TestFunction, inner, Measure, as_vector

h = 0.025

geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
domain = geom.add_rectangle([0, 0, 0], 1, 1.1)
lower_c = geom.add_rectangle([0, 0, 0], 0.5, 0.2)
upper_c = geom.add_rectangle([0, 0.8, 0], 0.5, 0.2)
mid_c = geom.add_rectangle([0, 0.1, 0], 0.2, 0.8)
c = geom.boolean_union([lower_c, upper_c, mid_c])
coil = geom.add_rectangle([0.2, 0.2, 0], 0.2, 0.6)
airgap = geom.boolean_difference([domain], [c, coil], delete_other=False)
frags = geom.boolean_fragments([airgap], [c, coil])
geom.add_physical(airgap, 1)
geom.add_physical(c, 2)
geom.add_physical(coil, 3)
# print(geom.get_code())
pygmsh_mesh = pygmsh.generate_mesh(geom)

# Prune z, probably a better way to do this (see generate mesh options)
x = np.array([pt[0:2] for pt in pygmsh_mesh.points])

cells = np.vstack([cells.data for cells in pygmsh_mesh.cells])
values = np.hstack([cell_data for cell_data in
                    pygmsh_mesh.cell_data["gmsh:physical"]])

mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                   ufl_mesh_from_gmsh("triangle", 2))
local_entities, local_values = \
    extract_local_entities(mesh, 2, cells, values)
# TODO Create connectivity?

mat_mt = create_meshtags(mesh, 2,
                         AdjacencyList_int32(local_entities),
                         np.int32(local_values))

V = FunctionSpace(mesh, ("Lagrange", 2))


# TODO Replace with meshtag
def bound_marker(x):
    l = np.isclose(x[0], 0)
    r = np.isclose(x[0], 1)
    b = np.isclose(x[1], 0)
    t = np.isclose(x[1], 1.1)

    l_r = np.logical_or(l, r)
    b_t = np.logical_or(b, t)
    l_r_b_t = np.logical_or(l_r, b_t)

    return l_r_b_t


u_bc = Function(V)
u_bc.vector.set(0.0)
facets = locate_entities_boundary(mesh, 1, bound_marker)
bdofs = locate_dofs_topological(V, 1, facets)
bc = DirichletBC(u_bc, bdofs)

mu_0 = 4 * np.pi * 1e-7
mu_r_iron = 5000
J = Constant(mesh, 1.0)
A_z = TrialFunction(V)
v = TestFunction(V)

dx = Measure("dx", subdomain_data=mat_mt)

a = (1 / mu_0) * inner(grad(A_z), grad(v)) * dx(1) \
    + (1 / mu_0) * inner(grad(A_z), grad(v)) * dx(3) \
    + (1 / (mu_r_iron * mu_0)) * inner(grad(A_z), grad(v)) * dx(2)
L = inner(J, v) * dx(3)

A_z = Function(V)
solve(a == L, A_z, bc, petsc_options={"ksp_type": "preonly",
                                      "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "A_z.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(A_z)

# TODO Is there a better way? Is the project needed? What space?
W = VectorFunctionSpace(mesh, ("DG", 1))
B = TrialFunction(W)
v = TestFunction(W)
f = as_vector((A_z.dx(1), -A_z.dx(0)))

a = inner(B, v) * ufl.dx
L = inner(f, v) * ufl.dx

B = Function(W)
solve(a == L, B, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "B.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(B)
