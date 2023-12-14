#!/bin/bash
# Based on trilinos released: trilinos-release-13.4.1
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
      -DCMAKE_INSTALL_PREFIX=/home/chris/packages/trilinos ..
      .
