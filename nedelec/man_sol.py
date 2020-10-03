# Computes manufactured solutions to curl(curl(A)) = f

import sympy

from sympy import symbols
from sympy.vector import CoordSys3D, laplacian, gradient, divergence, curl
from sympy import sin, cos, pi, simplify


R = CoordSys3D("R")

# 2D problems
# Problem 1
A = 0.5 * R.y * (1 - R.y) * R.i

# Problem 2
# A = sin(pi * R.y) * R.i

# 3D problems
# Problem 3
# A = R.y * (1 - R.y) * R.z * (1 - R.z) * R.i

# Problem 4
# A = sin(pi * R.y) * sin(pi * R.z) * R.i

f = simplify(curl(curl(A)))
print(f"f = {f}")

B = simplify(curl(A))
print(f"B = {B}")
