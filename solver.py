# Solver for the magnetostatic A formulation from [1]

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

# TODO Solver currently assumes homogeneous Neumann BCs everywhere. Add
# ability to use more complicated BCs

from dolfinx import Function,  FunctionSpace, VectorFunctionSpace
from ufl import TrialFunction, TestFunction, inner, dx, curl
from util import project
from dolfinx.fem import assemble_matrix, assemble_vector
from petsc4py import PETSc


def solve_problem(problem):
    """Solves a magnetostatic problem.
    Args:
        problem: A problem created by problems.py
    Returns:
        A: The magnetic vector potential. Note that A is not unique because
           of the nullspace of the curl operator i.e. curl(grad(\phi)) = 0 for
           any \phi, so for any A that is a solution, A + grad(\phi) is also a
           solution.
    """
    V = FunctionSpace(problem.mesh, ("N1curl", problem.k))

    A = TrialFunction(V)
    v = TestFunction(V)

    mu = problem.mu
    T_0 = problem.T_0
    a = inner(1 / mu * curl(A), curl(v)) * dx

    L = inner(T_0, curl(v)) * dx

    A = Function(V)

    # TODO More steps needed here for Dirichlet boundaries
    mat = assemble_matrix(a, [])
    mat.assemble()
    vec = assemble_vector(L)

    # TODO Use AMS
    # NOTE Need to use iterative solver due to nullspace of curl operator
    # Set solver options
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["pc_type"] = "icc"
    opts["ksp_rtol"] = 1e-12
    opts["ksp_monitor"] = None

    # Create solver
    solver = PETSc.KSP().create(problem.mesh.mpi_comm())
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(mat)

    # Compute solution
    solver.solve(vec, A.vector)
    solver.view()
    return A


def compute_B(A, k, mesh):
    """Computes the magnetic field.
    Args:
        A: Magnetic vector potential
        k: Degree of DG space for B
        mesh: The mesh
    Returns:
        B: The magnetic flux density
    """
    # TODO Get k from A somehow and use k - 1 for degree of V
    V = VectorFunctionSpace(mesh, ("DG", k))
    B = project(curl(A), V)
    return B
