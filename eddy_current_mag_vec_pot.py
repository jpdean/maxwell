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
from dolfinx.fem.assemble import assemble_scalar


# TODO Add references/page numbers for equations


class Problem:
    def __init__(self, h, freq, J_s, k):
        self.h = h
        # Frequency (Hz)
        self.freq = freq
        self.J_s = J_s
        self.k = k
        self.mesh, self.mat_mt = self.create_mesh()

    def create_mesh(self):
        return None

    def boundary_marker(self):
        return None

    def bound_cond(self, x):
        return None


class Prob1(Problem):
    def __init__(self, h, freq, J_s, k):
        super().__init__(h, freq, J_s, k)

    def create_mesh(self):
        # TODO Remove magic numbers
        geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
        domain = geom.add_rectangle([0, 0, 0], 1, 1.1)
        lower_c = geom.add_rectangle([0, 0, 0], 0.5, 0.2)
        upper_c = geom.add_rectangle([0, 0.8, 0], 0.5, 0.2)
        mid_c = geom.add_rectangle([0, 0.1, 0], 0.2, 0.8)
        c = geom.boolean_union([lower_c, upper_c, mid_c])
        coil = geom.add_rectangle([0.2, 0.2, 0], 0.2, 0.6)
        lower_c2 = geom.add_rectangle([0.55, 0, 0], 0.35, 0.2)
        upper_c2 = geom.add_rectangle([0.55, 0.8, 0], 0.35, 0.2)
        mid_c2 = geom.add_rectangle([0.7, 0, 0], 0.2, 1.0)
        c2 = geom.boolean_union([lower_c2, upper_c2, mid_c2])
        airgap = geom.boolean_difference([domain], [c, c2, coil], delete_other=False)
        geom.boolean_fragments([airgap], [c, c2, coil])
        geom.add_physical(airgap, 1)
        geom.add_physical(c, 2)
        geom.add_physical(coil, 3)
        geom.add_physical(c2, 4)
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
        return mesh, mat_mt

    def boundary_marker(self, x):
        l = np.isclose(x[0], 0)
        r = np.isclose(x[0], 1)
        b = np.isclose(x[1], 0)
        t = np.isclose(x[1], 1.1)

        l_r = np.logical_or(l, r)
        b_t = np.logical_or(b, t)
        l_r_b_t = np.logical_or(l_r, b_t)
        return l_r_b_t

    def bound_cond(self, x):
        # TODO Make this use meshtags
        values = np.zeros((1, x.shape[1]))
        return values


def solver(problem):
    mesh = problem.mesh
    mat_mt = problem.mat_mt
    k = problem.k
    omega = 2 * np.pi * problem.freq
    J_s = problem.J_s

    mu_0 = 4 * np.pi * 1e-7
    mu_r_iron = 5000
    sigma_iron = 1e7

    V = FunctionSpace(mesh, ("Lagrange", k))

    A_bc = Function(V)
    A_bc.interpolate(problem.bound_cond)
    facets = locate_entities_boundary(mesh, 1, problem.boundary_marker)
    bdofs = locate_dofs_topological(V, 1, facets)
    bc = DirichletBC(A_bc, bdofs)

    # TODO Could remove assumption of J being constant.
    J = Constant(mesh, J_s)

    A = TrialFunction(V)
    v = TestFunction(V)

    dx = Measure("dx", subdomain_data=mat_mt)

    a = (1 / mu_0) * inner(grad(A), grad(v)) * dx(1) \
        + (1 / mu_0) * inner(grad(A), grad(v)) * dx(3) \
        + (1 / (mu_r_iron * mu_0)) * inner(grad(A), grad(v)) * dx(2) \
        + (1 / (mu_r_iron * mu_0)) * inner(grad(A), grad(v)) * dx(4) \
        - sigma_iron * 1j * omega * inner(A, v) * dx(4)
    L = inner(J, v) * dx(3)

    A = Function(V)
    solve(a == L, A, bc, petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu"})
    return A


# TODO Make a time dependent save option (i.e. to save field with
# e^{i omega t})
def save(mesh, v, file_name):
    with XDMFFile(MPI.COMM_WORLD, file_name, "w") as file:
        file.write_mesh(mesh)
        file.write_function(v)


def compute_B_from_A(A, problem):
    # TODO Is there a better way?
    V = VectorFunctionSpace(problem.mesh,
                            ("Discontinuous Lagrange", problem.k - 1))
    B = TrialFunction(V)
    v = TestFunction(V)
    f = as_vector((A.dx(1), - A.dx(0)))

    a = inner(B, v) * ufl.dx
    L = inner(f, v) * ufl.dx

    B = Function(V)
    solve(a == L, B, [], petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu"})
    return B

    # with XDMFFile(MPI.COMM_WORLD, "B.xdmf", "w") as file:
    #     file.write_mesh(mesh)
    #     file.write_function(B)


    # # Output file
    # file = XDMFFile(MPI.COMM_WORLD, "J.xdmf", "w")
    # file.write_mesh(mesh)

    # X = FunctionSpace(mesh, ("CG", 1))
    # t = 0
    # T = 1 / freq
    # # NOTE J_e_sol needed so that it all outputs to the same name in paraview
    # J_e_sol = Function(X)
    # while t < T:
    #     t += T / 100
    #     # TODO This should be the real part. Find out how to get this
    #     f = - sigma_iron * 1j * omega * A_z * ufl.exp(1j * omega * t)
    #     J_e = TrialFunction(X)
    #     v = TestFunction(X)

    #     a = inner(J_e, v) * ufl.dx
    #     L = inner(f, v) * dx(4)

    #     J_e = Function(X)
    #     solve(a == L, J_e, [], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    #     J_e.vector.copy(result=J_e_sol.vector)
    #     file.write_function(J_e_sol, t)

    # # NOTE Inner takes complex conjugate of secon argument, so no need to e.g. get magnitude
    # # pg 244 of Bastos and Sadowski
    # ave_power = omega**2 / 2 * mesh.mpi_comm().allreduce(assemble_scalar(sigma_iron * inner(A_z, A_z) * dx(4)), op=MPI.SUM)
    # print(f"Average power loss = {ave_power} W")

if __name__ == "__main__":
    h = 0.01
    freq = 0.01
    J_s = 100
    k = 2

    problem = Prob1(h, freq, J_s, k)
    A = solver(problem)
    save(problem.mesh, A, "A.xdmf")

    B = compute_B_from_A(A, problem)
    save(problem.mesh, B, "B.xdmf")
