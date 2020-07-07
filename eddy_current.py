# Solves the very simple 1D eddy current problem on page 39 of "Numerical
# methods in Electromagnetism" by Chari and Salon. Problem consists of a
# material occupying a half-space x >= 0, subjected to an applied sinusoidal
# magnetic field in the y-direction with angular frequency omega. Thus, this
# H field is given by H_y = Re(e^(i * omega * t)), with H_x and H_z both
# zero. This results in current flowing only at the surface due to the
# induced eddys (see nice diagram on Wikipedia page for the skin effect).
# Since the only non-zero component of H is the y-component, the flux
# density B only has a non-zero y-component, and the electric field E and
# current density J only have a z-component.

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import solve
from dolfinx import (DirichletBC, Function, FunctionSpace, UnitIntervalMesh,
                     Constant)
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, inner, grad, TrialFunction, TestFunction
from dolfinx.io import XDMFFile

# Angular frequency
omega = 2 * np.pi * 1
# Permiability (of copper)
mu = 1.256629e-6
# Conductivity (of copper)
sigma = 5.96e7
# Surface (x = 0) flux density
B_0 = 1.0
# Polynomial order
p = 1
# Number of cells
n = 100

# Create 1D mesh of unit length
mesh = UnitIntervalMesh(MPI.COMM_WORLD, n)

# Create function space and trial / test functions
V = FunctionSpace(mesh, ("Lagrange", p))
# y-component of the B field
B = TrialFunction(V)
v = TestFunction(V)
# Zero forcing term
f = Constant(mesh, 0)

# Define bilinear and linear forms. The weak formulation was derived from the
# strong formulation given in the above mentioned book.
a = inner(grad(B), grad(v)) * dx + 1j * omega * mu * sigma * inner(B, v) * dx
L = inner(f, v) * dx

# Boundary condition array
bcs = []

# Boundary condition at x = 0 (B = B_0)
left_B_bc = Function(V)
left_B_bc.vector.set(B_0)
left_point = locate_entities_boundary(mesh, 0,
                                      lambda x: np.isclose(x[0], 0.0))
left_bdofs = locate_dofs_topological(V, 0, left_point)
left_bc = DirichletBC(left_B_bc, left_bdofs)
bcs.append(left_bc)

# Boundary condition at x = 1 (B decays exponentially as a function of x, so
# set the RHS to zero. Should be OK if domain is much larger than the skin
# depth)
right_B_bc = Function(V)
right_B_bc.vector.set(0.0)
right_point = locate_entities_boundary(mesh, 0,
                                       lambda x: np.isclose(x[0], 1.0))
right_bdofs = locate_dofs_topological(V, 0, right_point)
right_bc = DirichletBC(right_B_bc, right_bdofs)
bcs.append(right_bc)

# Make space for solution and solve
B = Function(V)
solve(a == L, B, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "B.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(B)

# Skin depth (for exact solution)
delta = np.sqrt(2 / (omega * mu * sigma))


# Function for exact solution, given in above book
def B_exact(x):
    B_e = B_0 * np.exp(-x[0] / delta) * np.exp(- 1j * x[0] / delta)
    return B_e


V_exact = FunctionSpace(mesh, ("Lagrange", p + 3))
B_e = Function(V_exact)
B_e.interpolate(B_exact)

with XDMFFile(MPI.COMM_WORLD, "B_e.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(B_e)

# z-component of current density. This is computed from the magnetic flux
# density (see book). Here, it is projected into a Lagrange space.
J = TrialFunction(V)
v = TestFunction(V)
f = 1 / mu * B.dx(0)

# Projection
a = inner(J, v) * ufl.dx
L = inner(f, v) * ufl.dx

J = Function(V)
solve(a == L, J, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

with XDMFFile(MPI.COMM_WORLD, "J.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(J)


# Exact solution for z-component of current density
def J_exact(x):
    J_e = - B_0 / mu * (1 + 1j) / delta * np.exp(-(1 + 1j) * x[0] / delta)
    return J_e


J_e = Function(V_exact)
J_e.interpolate(J_exact)

with XDMFFile(MPI.COMM_WORLD, "J_e.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(J_e)
