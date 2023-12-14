
## Muelu Trilinos Demo

To compile, first install FEniCS-X dolfinx (real or complex)
and [Trilinos](https://github.com/trilinos/Trilinos.git)  (an example build script is in builder.sh)

1. Clone trilinos from their github and checkout the correct version:
```bash
git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout trilinos-release-13-0-1
```
2. Buld trilinos
```bash
cmake -B build-dir-trilinos \
      -DTPL_ENABLE_MPI=ON \
      -DMPI_BASE_DIR=/usr/x86_64-linux/ \
      -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
      -DTrilinos_ENABLE_MueLu=ON \
      -DTrilinos_ENABLE_TrilinosCouplings=ON \
      -DTrilinos_ENABLE_Teko=OFF \
      -DTpetra_INST_INT_LONG=ON \
      -DTpetra_INST_INT_LONG_LONG=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DTrilinos_ENABLE_PyTrilinos=OFF \
      -DTPL_ENABLE_Netcdf=OFF \
      .
cd build-dir-trilinos
make -j 2 install
```

3. Compile integration kernels
For real

```bash
python3 -m ffcx --scalar_type=double maxwell.ufl
```

or for complex

```bash
python3 -m ffcx --scalar_type="double complex" maxwell.ufl
```

4. Compile deom

```bash
cmake .
make
```
