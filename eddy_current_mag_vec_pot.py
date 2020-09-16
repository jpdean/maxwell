# 2D solver for the magnetic vector potential. See [1] and [2] for
# more information
# NOTE 2D solver therefore magnetic vector potential only has a single
# component in the z direction, called A in the code. The B field lies
# in the 2D plane. The current density has only a z component.
# References:
# [1] "Electromagnetic Modeling by Finite Element Methods"
#     by Bastos and Sadowski
# [2] https://fenicsproject.org/pub/tutorial/html/._ftut1015.html#___sec104

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
from ufl import (grad, TrialFunction, TestFunction, inner, Measure,
                 as_vector, real)
from dolfinx.fem.assemble import assemble_scalar
import physical_properties as props


class Problem:
    def __init__(self, h, freq, k):
        self.h = h
        # Frequency (Hz)
        self.freq = freq
        self.k = k
        self.mesh, self.mat_mt = self.create_mesh()
        self.mu = self.create_mu()
        self.sigma = self.create_sigma()

    def create_mesh(self):
        return None

    def boundary_marker(self):
        return None

    def bound_cond(self, x):
        return None

    def calc_omega(self):
        return 2 * np.pi * self.freq

    def create_mu(self):
        return None

    def create_sigma(self):
        return None

    def get_J_s_dict(self):
        return None


# Similar to the problem on pg 282
class Prob1(Problem):
    def __init__(self, h=0.01, freq=0.001, k=2):
        # This problem has large geometry (approx 1 m) so only need a low
        # frequency
        super().__init__(h, freq, k)

    def create_mesh(self):
        # TODO Remove magic numbers
        h = self.h
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
        airgap = geom.boolean_difference([domain], [c, c2, coil],
                                         delete_other=False)
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
        left = np.isclose(x[0], 0)
        right = np.isclose(x[0], 1)
        bottom = np.isclose(x[1], 0)
        top = np.isclose(x[1], 1.1)

        l_r = np.logical_or(left, right)
        b_t = np.logical_or(bottom, top)
        l_r_b_t = np.logical_or(l_r, b_t)
        return l_r_b_t

    def bound_cond(self, x):
        # TODO Make this use meshtags
        values = np.zeros((1, x.shape[1]))
        return values

    def create_mu(self):
        # Based on https://fenicsproject.discourse.group/t/dolfinx-discontinous-expression/2582/3
        V = FunctionSpace(self.mesh, ("Discontinuous Lagrange", 0))
        mu = Function(V)
        mat_mt = self.mat_mt

        for i in range(V.dim):
            if mat_mt.values[i] == 1:
                mu.vector.setValueLocal(i, props.permiability_air)
            elif mat_mt.values[i] == 2:
                mu.vector.setValueLocal(i, props.permiability_iron)
            elif mat_mt.values[i] == 3:
                mu.vector.setValueLocal(i, props.permiability_copper)
            elif mat_mt.values[i] == 4:
                mu.vector.setValueLocal(i, props.permiability_iron)
        return mu

    def create_sigma(self):
        V = FunctionSpace(self.mesh, ("Discontinuous Lagrange", 0))
        sigma = Function(V)
        mat_mt = self.mat_mt

        for i in range(V.dim):
            if mat_mt.values[i] == 1:
                sigma.vector.setValueLocal(i, props.conductivity_air)
            elif mat_mt.values[i] == 2:
                # Iron in this region is laminated so assume zero
                # conductivity
                sigma.vector.setValueLocal(i, 0)
            elif mat_mt.values[i] == 3:
                # J_s prescribed in coil, so set sigma to 0
                sigma.vector.setValueLocal(i, 0)
            elif mat_mt.values[i] == 4:
                sigma.vector.setValueLocal(i, props.conductivity_iron)
        return sigma

    def get_J_s_dict(self):
        # Prescribe J_s in left and right coils (numbered 5 and 6
        # respectively)
        return {3: 10000}


# Problem similar to RR single phase induction motor. Increase
# frequency to e.g. 500 Hz to see skin effect
class Prob2(Problem):
    def __init__(self, h=0.001, freq=50, k=2):
        # This problem has large geometry (approx 1 m) so only need a low
        # frequency
        super().__init__(h, freq, k)

    def create_mesh(self):
        h = self.h

        r1 = 0.02
        r2 = 0.03
        r3 = 0.032
        r4 = 0.052
        r5 = 0.057

        geom = pygmsh.opencascade.Geometry(characteristic_length_max=h)
        outer_stator = geom.add_disk([0.0, 0.0, 0.0], r5)
        inner_stator = geom.add_disk([0.0, 0.0, 0.0], r4)
        stator = geom.boolean_difference([outer_stator], [inner_stator],
                                         delete_other=False)

        coil_height = r1
        coil_width = r5 - r1
        right_coil = geom.add_rectangle([r1, - coil_height / 2, 0],
                                        coil_width, coil_height)
        right_coil = geom.boolean_intersection([inner_stator, right_coil],
                                               delete_first=False)
        geom.boolean_fragments([right_coil], [stator])
        air_gap_circ = geom.add_disk([0.0, 0.0, 0.0], r3)
        right_coil = geom.boolean_difference([right_coil], [air_gap_circ],
                                             delete_other=False)

        left_coil = geom.add_rectangle([- r1, - coil_height / 2, 0],
                                       - coil_width, coil_height)
        left_coil = geom.boolean_intersection([inner_stator, left_coil],
                                              delete_first=False)
        geom.boolean_fragments([left_coil], [stator])
        left_coil = geom.boolean_difference([left_coil], [air_gap_circ])

        outer_rotor = geom.add_disk([0.0, 0.0, 0.0], r2, char_length=h/2)
        rotor_steel = geom.add_disk([0.0, 0.0, 0.0], r1, char_length=h/2)
        rotor_al = geom.boolean_difference([outer_rotor], [rotor_steel],
                                           delete_other=False)
        air = geom.boolean_difference(
            [inner_stator],
            [left_coil, right_coil, rotor_al, rotor_steel], delete_other=False)

        geom.add_physical(air, 1)
        geom.add_physical(stator, 2)
        geom.add_physical(rotor_al, 3)
        geom.add_physical(rotor_steel, 4)
        geom.add_physical(left_coil, 5)
        geom.add_physical(right_coil, 6)

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
        # TODO Don't hardcode radius
        r5 = 0.057
        r = np.sqrt(x[0]**2 + x[1]**2)
        return np.isclose(r, r5)

    def bound_cond(self, x):
        # TODO Make this use meshtags
        values = np.zeros((1, x.shape[1]))
        return values

    def create_mu(self):
        # Based on https://fenicsproject.discourse.group/t/dolfinx-discontinous-expression/2582/3
        V = FunctionSpace(self.mesh, ("Discontinuous Lagrange", 0))
        mu = Function(V)
        mat_mt = self.mat_mt

        for i in range(V.dim):
            if mat_mt.values[i] == 1:
                mu.vector.setValueLocal(i, props.permiability_air)
            elif mat_mt.values[i] == 2:
                mu.vector.setValueLocal(i, props.permiability_iron)
            elif mat_mt.values[i] == 3:
                mu.vector.setValueLocal(i, props.permiability_aluminium)
            elif mat_mt.values[i] == 4:
                mu.vector.setValueLocal(i, props.permiability_iron)
            elif mat_mt.values[i] == 5:
                mu.vector.setValueLocal(i, props.permiability_copper)
            elif mat_mt.values[i] == 6:
                mu.vector.setValueLocal(i, props.permiability_copper)
        return mu

    def create_sigma(self):
        V = FunctionSpace(self.mesh, ("Discontinuous Lagrange", 0))
        sigma = Function(V)
        mat_mt = self.mat_mt

        for i in range(V.dim):
            if mat_mt.values[i] == 1:
                sigma.vector.setValueLocal(i, props.conductivity_air)
            elif mat_mt.values[i] == 2:
                # Iron in this region is laminated so assume zero
                # conductivity
                sigma.vector.setValueLocal(i, 0)
            elif mat_mt.values[i] == 3:
                sigma.vector.setValueLocal(i, props.conductivity_aluminium)
            elif mat_mt.values[i] == 4:
                sigma.vector.setValueLocal(i, props.conductivity_iron)
            elif mat_mt.values[i] == 5:
                # J_s prescribed in coil, so set sigma to 0
                sigma.vector.setValueLocal(i, 0)
            elif mat_mt.values[i] == 6:
                # J_s prescribed in coil, so set sigma to 0
                sigma.vector.setValueLocal(i, 0)
        return sigma

    def get_J_s_dict(self):
        # Prescribe J_s in left and right coils (numbered 5 and 6
        # respectively)
        return {5: -3.1e6, 6: 3.1e6}


def solver(problem):
    mesh = problem.mesh
    mat_mt = problem.mat_mt
    k = problem.k
    omega = problem.calc_omega()

    V = FunctionSpace(mesh, ("Lagrange", k))

    A_bc = Function(V)
    A_bc.interpolate(problem.bound_cond)
    facets = locate_entities_boundary(mesh, 1, problem.boundary_marker)
    bdofs = locate_dofs_topological(V, 1, facets)
    bc = DirichletBC(A_bc, bdofs)

    A = TrialFunction(V)
    v = TestFunction(V)

    a = (1 / problem.mu) * inner(grad(A), grad(v)) * ufl.dx \
        - problem.sigma * 1j * omega * inner(A, v) * ufl.dx

    # NOTE Could handle J_s in a similar way as mu and sigma (i.e. DG0
    # functions defined on whole domain), but I think this way makes
    # it easier to cope with non-constant current densities
    dx_mt = Measure("dx", subdomain_data=mat_mt)
    L_integral_list = []
    for index, J_s in problem.get_J_s_dict().items():
        # TODO Could easily remove assumption of J being constant.
        J_s = Constant(mesh, J_s)
        L_integral_list.append(inner(J_s, v) * dx_mt(index))
    L = sum(L_integral_list)

    A = Function(V)
    solve(a == L, A, bc, petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu"})
    return A


def project(f, V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * ufl.dx
    L = inner(f, v) * ufl.dx

    u = Function(V)
    solve(a == L, u, [], petsc_options={"ksp_type": "preonly",
                                        "pc_type": "lu"})
    return u


def compute_B_from_A(A, problem):
    # TODO Is there a better way?
    V = VectorFunctionSpace(problem.mesh,
                            ("Discontinuous Lagrange", problem.k - 1))
    # See [2]
    f = as_vector((A.dx(1), - A.dx(0)))
    B = project(f, V)
    return B


def compute_J_e_from_A(A, problem):
    V = FunctionSpace(problem.mesh,
                      ("Lagrange", problem.k))
    omega = problem.calc_omega()
    # The eddy current "phasor" J_e = - \sigma * i * \omega A^~
    # (see [1] pgs 235 and 242). In the code, I use A to represent A^~
    f = - problem.sigma * 1j * omega * A

    J_e = project(f, V)
    return J_e


# The actual source current is given by Re(J_s e^{i \omega t}) (see [1] pg
# 242). Other quantities can be obtained from A, B, J_e etc. in the same
# manner. Note that the phase angle of J_s it taken to be zero (i.e. the
# reference).
# NOTE Only the real part of the time_series field is of physical
# significance
def save(v, problem, file_name, n=100, time_series=False):
    """Saves results. If time_series=False, the saved results is a complex
       phasor. If time_series=True, then the real part of the saved date
       is the time dependent field.
    """
    if not time_series:
        with XDMFFile(MPI.COMM_WORLD, file_name, "w") as file:
            file.write_mesh(problem.mesh)
            file.write_function(v)
    else:
        out_file = XDMFFile(MPI.COMM_WORLD, file_name, "w")
        out_file.write_mesh(problem.mesh)

        t = 0
        T = 1 / problem.freq
        omega = problem.calc_omega()
        delta_t = T / n

        V = v.function_space
        # Need to always write out the same vector, otherwise when opened in
        # Paraview there are lots of different fields which is annoying for
        # playback
        v_out = Function(V)

        while t < T:
            # NOTE Originally I used ufl.real to get the real part only, but
            # for some reason this breaks project for vector fields. Not
            # sure why.
            # NOTE See [1] pg 242
            f = v * ufl.exp(1j * omega * t)
            v_eval_at_t = project(f, V)
            v_eval_at_t.vector.copy(result=v_out.vector)
            out_file.write_function(v_out, t)
            t += delta_t


def compute_ave_power_loss(A, problem):
    omega = problem.calc_omega()

    # From pg 244 of [1]
    # NOTE Inner takes complex conjugate of secon argument, so no need
    # to e.g. get magnitude
    L = problem.sigma * inner(A, A) * ufl.dx
    ave_power_loss = omega**2 / 2 * problem.mesh.mpi_comm().allreduce(
        assemble_scalar(L), op=MPI.SUM)
    # This will only have a real component so take that
    return real(ave_power_loss)


if __name__ == "__main__":
    print("A")
    problem = Prob1()
    A = solver(problem)
    save(A, problem, "A.xdmf", time_series=False)

    print("B")
    B = compute_B_from_A(A, problem)
    save(B, problem, "B.xdmf", time_series=False)

    print("J_e")
    J_e = compute_J_e_from_A(A, problem)
    save(J_e, problem, "J_e.xdmf", time_series=False)

    ave_power_loss = compute_ave_power_loss(A, problem)
    print(f"Average power loss = {ave_power_loss} W")
