#!/bin/bash
rm -r build
mkdir -p build
cd build
cmake -DTPL_ENABLE_MPI=ON \
      -DMPI_BASE_DIR=/usr/x86_64-linux/ \
      -DTrilinos_ENABLE_COMPLEX_DOUBLE=ON \
      -DTrilinos_ENABLE_MueLu=ON \
      -DTrilinos_ENABLE_TrilinosCouplings=ON \
      -DTpetra_INST_INT_LONG_LONG=OFF \
      -DTpetra_INST_INT_LONG=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DTrilinos_ENABLE_PyTrilinos=OFF \
      -DTPL_ENABLE_Netcdf=OFF \
      -DCMAKE_INSTALL_PREFIX=/home/chris/packages/trilinos ..
