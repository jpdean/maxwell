# Maxwell problem based on the FEniCS tutorial:
# https://fenicsproject.org/pub/tutorial/html/._ftut1015.html#___sec104
# Updated for dolfinx.

# Solves for the magnetic field of a copper wire in a vacuum

# FIXME Why does B not go to zero at the boundary?

import pygmsh
import numpy as np
import pygmsh
from mpi4py import MPI
from dolfinx import cpp, solve
from dolfinx.cpp.io import perm_gmsh, extract_local_entities
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.mesh import (create_mesh, create_meshtags, MeshTags,
                          locate_entities_boundary)
import meshio
from dolfinx import FunctionSpace, Function, Constant, VectorFunctionSpace
from dolfinx.fem import DirichletBC, locate_dofs_topological
import dolfinx
from ufl import (Measure, dot, grad, TrialFunction, TestFunction, inner,
                 as_vector)
import ufl

# FIXME This will generate on each process
geom = pygmsh.opencascade.Geometry()
outer_disk = geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=0.25)
inner_disk = geom.add_disk([0.0, 0.0, 0.0], 0.05, char_length=0.01)
frags = geom.boolean_fragments([outer_disk], [inner_disk])
# Outer
geom.add_raw_code("Physical Surface(1) = {3};")
# Inner
geom.add_raw_code("Physical Surface(2) = {2};")
# print(geom.get_code())
pygmsh_mesh = pygmsh.generate_mesh(geom)

# Prune z, probably a better way to do this...
x = np.array([pt[0:2] for pt in pygmsh_mesh.points])

# Cells
inner_cells = pygmsh_mesh.cells[0].data
outer_cells = pygmsh_mesh.cells[1].data
cells = np.vstack((inner_cells, outer_cells))

# Cell values
inner_cell_values = pygmsh_mesh.cell_data["gmsh:physical"][0]
outer_cell_values = pygmsh_mesh.cell_data["gmsh:physical"][1]
values = np.hstack((inner_cell_values, outer_cell_values))

mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh("triangle", 2))
mesh.name = "wire"

local_entities, local_values = extract_local_entities(mesh, 2, cells, values)
# TODO Create connectivity?

mat_mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mat_mt.name = "material"

# with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
#     file.write_mesh(mesh)
#     file.write_meshtags(mat_mt) # TODO Path needed?

# meshio.write("subdomains.vtu", pygmsh_mesh)

V = FunctionSpace(mesh, ("Lagrange", 1))


# TODO Replace with meshtag
def bound_marker(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, 1.0) # TODO Don't hardcode

u_bc = Function(V)
u_bc.vector.set(0.0)

facets = locate_entities_boundary(mesh, 1, bound_marker)
bdofs = locate_dofs_topological(V, 1, facets)
bc = DirichletBC(u_bc, bdofs)

mu = [1.26e-6, 4 * np.pi * 1e-7]

dx = Measure("dx", subdomain_data=mat_mt)

J = Constant(mesh, 1.0)
A_z = TrialFunction(V)
v = TestFunction(V)

# TODO Write as function / list comprehension / use loop
a = (1 / mu[0]) * inner(grad(A_z), grad(v)) * dx(1) + \
    (1 / mu[1]) * inner(grad(A_z), grad(v)) * dx(2)
L = inner(J, v) * dx(1)


A_z = Function(V)
solve(a == L, A_z, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "A_z.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(A_z)


# TODO Is there a better way? Is the project needed?
W = VectorFunctionSpace(mesh, ("Lagrange", 1))
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
