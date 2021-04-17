
## Muelu Trilinos Demo

To compile, first install FEniCS-X dolfinx (real or complex)
and Trilinos (an example build script is in builder.sh)

For real

`ffcx --scalar_type=double maxwell.ufl`

or for complex

`ffcx --scalar_type="double complex" maxwell.ufl`

cmake .
make
