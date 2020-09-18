# Solves the generic equation given on pg 131 of [1] but
# assuming \alpha_x = \alpha_y = \alpha

# References:
# [1] The Finite Element Method in Electromagnetics by Jin

from problems import Problem
from meshes import create_unit_square_mesh


def solve(problem):
    pass


h = 0.1
mesh, cell_mt, facet_mt = create_unit_square_mesh(h)
k = 1
alpha_dict = {1: 1}
beta_dict = {1: 0}
problem = Problem(mesh, cell_mt, facet_mt, k, alpha_dict, beta_dict)
u = solve(problem)
