# Maxwell problem based on the FEniCS tutorial:
# https://fenicsproject.org/pub/tutorial/html/._ftut1015.html#___sec104
# Updated for dolfinx.

# Solves for the magnetic field of a copper wire in a vacuum

# TODO H(div) elements for normal jumps?

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

a = 1.0   # inner radius of iron cylinder
b = 1.2   # outer radius of iron cylinder
c_1 = 0.8 # radius for inner circle of copper wires
c_2 = 1.4 # radius for outer circle of copper wires
r = 0.1   # radius of copper wires
R = 5.0   # radius of domain
n = 10    # number of windings

# FIXME This will generate on each process
geom = pygmsh.opencascade.Geometry()
domain = geom.add_disk([0.0, 0.0, 0.0], R, char_length=0.25)
inner_iron = geom.add_disk([0.0, 0.0, 0.0], a, char_length=0.1)
outer_iron = geom.add_disk([0.0, 0.0, 0.0], b, char_length=0.1)
iron = geom.boolean_difference([outer_iron], [inner_iron])
thetas_up = [i * 2 * np.pi / n for i in range(n)]
wires_up = [geom.add_disk([c_1 * np.cos(theta), c_1 * np.sin(theta), 0.0],
            r, char_length=0.05) for theta in thetas_up]
thetas_down = [(i + 0.5) * 2 * np.pi / n for i in range(n)]
wires_down = [geom.add_disk([c_2 * np.cos(theta), c_2 * np.sin(theta), 0.0],
              r, char_length=0.05) for theta in thetas_down]
# Concatenate arrays
wires = wires_up + wires_down
frags = geom.boolean_fragments([domain], wires + [iron])
# Calculate GMSH surface ID number for vacuum region.
# Formula found by outputting gmsh code (geom.get_code()) and
# running interactively with gmsh to see how it numbers things.
vac_surf_id = 2 * n + 4
geom.add_raw_code("Physical Surface(1) = {" +
                   str(vac_surf_id) + ", " +
                   str(vac_surf_id + 1) + "};")
# For the wires and iron, we can just use pygmsh's add_physical method
geom.add_physical(wires_up, 2)
geom.add_physical(wires_down, 3)
geom.add_physical(iron, 4)
# print(geom.get_code())
pygmsh_mesh = pygmsh.generate_mesh(geom)

# Prune z, probably a better way to do this...
x = np.array([pt[0:2] for pt in pygmsh_mesh.points])

# Cells
# pygmsh_mesh.cells[i].data contains list of cells in the ith surface
# (i.e. for n turns, we have 2n circles and the vacuum space, 
# so len(pygmsh_mesh.cells) = 2n + 1).
# Need to collect these all together to create a dolfin mesh.
cells = np.vstack([cells.data for cells in pygmsh_mesh.cells])

# Cell values
# Do the same as above for the cell values, combining into one.
values = np.hstack([cell_data for cell_data in 
                    pygmsh_mesh.cell_data["gmsh:physical"]])

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
    return np.isclose(r, R)

u_bc = Function(V)
u_bc.vector.set(0.0)

facets = locate_entities_boundary(mesh, 1, bound_marker)
bdofs = locate_dofs_topological(V, 1, facets)
bc = DirichletBC(u_bc, bdofs)

mu_vacuum = 4 * np.pi * 1e-7
mu_copper = 1.26e-6
# Using lower value than iron's actual value of 6.3e-3 as is done
# in FEniCS tutorial. This is so the field in the iron doesn't
# dominate the solution as much, making it more interesting.
mu_iron = 1e-5

J = Constant(mesh, 1.0)
A_z = TrialFunction(V)
v = TestFunction(V)

# Vacuum marked as 1, wires up marked as 2, wires down marked as 3
dx = Measure("dx", subdomain_data=mat_mt)

# TODO Write as function / list comprehension / use loop
a = (1 / mu_vacuum) * inner(grad(A_z), grad(v)) * dx(1) + \
    (1 / mu_copper) * inner(grad(A_z), grad(v)) * dx(2) + \
    (1 / mu_copper) * inner(grad(A_z), grad(v)) * dx(3) + \
    (1 / mu_iron) * inner(grad(A_z), grad(v)) * dx(4)
L = inner(J, v) * dx(2) - inner(J, v) * dx(3)


A_z = Function(V)
solve(a == L, A_z, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "A_z.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(A_z)


# TODO Is there a better way? Is the project needed? What space?
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
