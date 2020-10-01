# Computes manufactured solutions to curl(curl(A)) = f

import sympy

from sympy import symbols
from sympy.vector import CoordSys3D, laplacian, gradient, divergence, curl
from sympy import sin, cos, pi, simplify


R = CoordSys3D("R")

A = R.y * (1 - R.y) * R.i

f = simplify(curl(curl(A)))
print(f"f = {f}")

B = simplify(curl(A))
print(f"B = {B}")
