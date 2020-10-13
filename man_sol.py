# Computes manufactured solutions to problem (1)-(2) in [1] with
# constitutive equations (7) and impressed field (9).

# References
# [1] Oszkar Biro, "Edge element formulations of eddy current problems"

from sympy import symbols
from sympy.vector import CoordSys3D, curl
from sympy import sin, cos, pi, simplify


R = CoordSys3D("R")
mu = symbols("mu")

# Problem 1
A = sin(pi * R.y) * R.i + sin(pi * R.z) * R.j + sin(pi * R.x) * R.k

B = simplify(curl(A))
print(f"B = {B}")

H = simplify(1 / mu * B)
print(f"H = {H}")

# From (1) and (9), curl(H) = curl(T_0), therefore one option is to just
# choose T_0 = H
T_0 = H
print(f"T_0 = {T_0}")

J_0 = simplify(curl(T_0))
print(f"J_0 = {J_0}")
