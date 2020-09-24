from problems import ProblemFactory
from postprocessing import L2_norm
from solver import solve
from postprocessing import L2_norm
import ufl


def test_convergence_rate():
    for k in range(1, 3):
        errors = []
        for h in [0.05, 0.025]:
            problem = ProblemFactory.create_Poisson_problem_2(h=h, k=k)
            x = ufl.SpatialCoordinate(problem.get_mesh())
            u = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
            u_h = solve(problem)
            errors.append(L2_norm(u - u_h))
        assert round(errors[0] / errors[1]) == 2**(k + 1)
