# Maxwell Dirichlet problem based on Matthew's RR convergence demo.
# FIXME Does not give correct convergence results in parallel.

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import (DirichletBC, Function, FunctionSpace, UnitCubeMesh, cpp,
                     VectorFunctionSpace)
from dolfinx.fem import (apply_lifting, locate_dofs_topological, set_bc,
                         assemble_matrix, assemble_vector, assemble_scalar)
from ufl import dx, inner, curl, div
import matplotlib.pylab as plt


def l2_projection(space, f, f_order=1):
    cg = VectorFunctionSpace(space.mesh, ("CG", f_order))
    f_fun = Function(cg)
    f_fun.interpolate(f)

    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)
    mass = inner(u, v) * dx
    rhs = inner(f_fun, v) * dx

    a = assemble_matrix(mass, [])
    a.assemble()

    b = assemble_vector(rhs)
    # apply_lifting(b, [a], [[]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    u = Function(space)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(a)
    solver.solve(b, u.vector)
    return u


def f_sol(x):
    out = np.zeros(x.shape, dtype=np.complex64)
    out[2, :] = np.exp((1j * k * (x[0] + x[1])) / np.sqrt(2))
    return out


def solve(mesh_n, k, order):
    mesh = UnitCubeMesh(MPI.COMM_WORLD, mesh_n, mesh_n, mesh_n)
    mesh.topology.create_connectivity_all()

    # V = FunctionSpace(mesh, ("N1curl", order))
    V = VectorFunctionSpace(mesh, ("CG", order))

    sol_V = VectorFunctionSpace(mesh, ("CG", order + 3))
    sol = Function(sol_V)
    sol.interpolate(f_sol)

    u0 = l2_projection(V, f_sol, order)
    bndry_facets = np.where(
        np.array(cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    bc = DirichletBC(u0, locate_dofs_topological(V, 2, bndry_facets))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = Function(V)
    f.vector.set(0.0)

    a_form = inner(curl(u), curl(v)) * dx - k ** 2 * inner(u, v) * dx + 10 * inner(div(u), div(v)) * dx
    b_form = inner(f, v) * dx

    a = assemble_matrix(a_form, [bc])
    a.assemble()

    b = assemble_vector(b_form)
    apply_lifting(b, [a_form], [[bc]])

    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Create solution function
    u = Function(V)

    # Set solver options
    opts = PETSc.Options()
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_package"] = "mumps"
    opts["options_suppress_deprecated_warnings"] = True

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(a)

    # Compute solution
    solver.solve(b, u.vector)
    # solver.view()

    L2_error = np.real(np.sqrt(mesh.mpi_comm().allreduce(assemble_scalar(
        inner(u - sol, u - sol) * dx), op=MPI.SUM)))
    div_error = np.real(np.sqrt(mesh.mpi_comm().allreduce(assemble_scalar(
        inner(div(u), div(u)) * dx), op=MPI.SUM)))
    return L2_error, div_error


def convergence(k, order):
    xs = []
    ys = []
    for i in range(2, 8):
        if i % 2 == 0:
            n = 2 ** (i // 2)
        else:
            n = 2 ** (i // 2) + 2 ** (i // 2 - 1)
        xs.append(1 / n)
        L2_error, div_error = solve(n, k, order)
        ys.append(L2_error)
        print_info(xs[-1], ys[-1], div_error)

    r = np.log(ys[-1] / ys[-2]) / np.log(xs[-1] / xs[-2])
    print(f"r = {r}")

    plt.plot(xs, ys, "ro-")
    plt.xscale("log")
    plt.xlabel("$h$")
    plt.yscale("log")
    plt.ylabel("Error (L2 norm)")
    plt.axis("equal")
    plt.xlim(plt.xlim()[::-1])
    plt.savefig("convergence.png")
    plt.show()


def problem(n, k, order):
    L2_error, div_error = solve(n, k, order)
    print_info(1 / n, L2_error, div_error)


def print_info(h, L2_error, div_error):
    print(f"h = {h}, L2-error = {L2_error}, Div-error = {div_error}")

if __name__ == "__main__":
    # Wave number
    k = 2
    # Order (div error zero for first order due to divergence of
    # basis functions being zero)
    order = 1
    # Number of elements in each direction
    n = 5
    convergence(k, order)
    # problem(n, k, order)
