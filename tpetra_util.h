
#include <memory>
#include <mpi.h>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrixFactory.hpp>
#include <Xpetra_CrsMatrixWrap_fwd.hpp>
#include <Xpetra_Matrix_fwd.hpp>

using Node = Kokkos::Compat::KokkosSerialWrapperNode;

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/la/SparsityPattern.h>

/// Helper functions to create Tpetra::CrsMatrix from
/// dolfin::la::SparsityPattern or dolfin::common::IndexMap (for a matrix with
/// only diagonal entries)

template <typename T>
Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
create_tpetra_matrix(MPI_Comm mpi_comm,
                     const dolfinx::la::SparsityPattern &pattern);

template <typename T>
Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
create_tpetra_diagonal_matrix(
    std::shared_ptr<const dolfinx::common::IndexMap> index_map);

template <typename T>
Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
create_tpetra_matrix(MPI_Comm mpi_comm,
                     const dolfinx::la::SparsityPattern &pattern) {
  Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));

  std::cout << "Sparsity = " << pattern.index_map(0)->size_global() << "x"
            << pattern.index_map(1)->size_global() << "\n";

  const dolfinx::graph::AdjacencyList<std::int32_t> &diagonal_pattern =
      pattern.diagonal_pattern();
  const dolfinx::graph::AdjacencyList<std::int32_t> &off_diagonal_pattern =
      pattern.off_diagonal_pattern();

  std::vector<std::size_t> nnz(diagonal_pattern.num_nodes());
  for (int i = 0; i < diagonal_pattern.num_nodes(); ++i)
    nnz[i] = diagonal_pattern.num_links(i) + off_diagonal_pattern.num_links(i);

  dolfinx::common::Timer tcre("Trilinos: create sparsity");
  std::vector<std::int64_t> global_indices1 = pattern.column_indices();

  const Teuchos::ArrayView<const std::int64_t> global_index_view1(
      global_indices1.data(), global_indices1.size());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t, Node>> colMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t, Node>(
          pattern.index_map(1)->size_global(), global_index_view1, 0, comm));

  // Column map with no ghosts = domain map (needed for rectangular matrix)
  const Teuchos::ArrayView<const std::int64_t> global_index_view1_domain(
      global_indices1.data(), pattern.index_map(1)->size_local());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t, Node>> domainMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t, Node>(
          pattern.index_map(1)->size_global(), global_index_view1_domain, 0,
          comm));

  std::vector<std::int64_t> global_indices0 =
      pattern.index_map(0)->global_indices();
  const Teuchos::ArrayView<const std::int64_t> global_index_view0(
      global_indices0.data(), pattern.index_map(0)->size_local());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t, Node>> vecMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t, Node>(
          pattern.index_map(0)->size_global(), global_index_view0, 0, comm));

  Teuchos::ArrayView<std::size_t> _nnz(nnz.data(), nnz.size());
  Teuchos::RCP<Tpetra::CrsGraph<std::int32_t, std::int64_t, Node>> crs_graph(
      new Tpetra::CrsGraph<std::int32_t, std::int64_t, Node>(vecMap, colMap,
                                                             _nnz));

  const std::int64_t nlocalrows = pattern.index_map(0)->size_local();
  for (std::size_t i = 0; i != diagonal_pattern.num_nodes(); ++i) {
    std::vector<std::int32_t> indices(diagonal_pattern.links(i).begin(),
                                      diagonal_pattern.links(i).end());
    for (std::int32_t q : off_diagonal_pattern.links(i))
      indices.push_back(q);

    Teuchos::ArrayView<std::int32_t> _indices(indices.data(), indices.size());
    crs_graph->insertLocalIndices(i, _indices);
  }

  crs_graph->fillComplete(domainMap, vecMap);
  tcre.stop();

  Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>(
              crs_graph));
  return A_Tpetra;
}

template <typename T>
Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
create_tpetra_diagonal_matrix(
    std::shared_ptr<const dolfinx::common::IndexMap> index_map) {

  Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Teuchos::rcp(new Teuchos::MpiComm<int>(index_map->comm()));

  // Get non-ghost global indices only
  std::vector<std::int64_t> global_indices = index_map->global_indices();
  global_indices.resize(index_map->size_local());

  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t, Node>> vecMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t, Node>(
          index_map->size_global(), global_indices, 0, comm));

  Teuchos::RCP<Tpetra::CrsGraph<std::int32_t, std::int64_t, Node>> crs_graph(
      new Tpetra::CrsGraph<std::int32_t, std::int64_t, Node>(vecMap, vecMap,
                                                             1));

  for (std::size_t i = 0; i != index_map->size_local(); ++i) {
    std::vector<std::int32_t> indices(1, i);
    crs_graph->insertLocalIndices(i, indices);
  }

  crs_graph->fillComplete();

  Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>(
              crs_graph));
  return A_Tpetra;
}

/// Convert Tpetra to Xpetra...
template <typename T>
Teuchos::RCP<Xpetra::Matrix<T, std::int32_t, std::int64_t, Node>>
tpetra_to_xpetra(
    Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>> mat) {
  Teuchos::RCP<Xpetra::CrsMatrix<T, std::int32_t, std::int64_t, Node>> mat_X =
      Teuchos::rcp(
          new Xpetra::TpetraCrsMatrix<T, std::int32_t, std::int64_t, Node>(
              mat));

  Teuchos::RCP<Xpetra::Matrix<T, std::int32_t, std::int64_t, Node>> A_mat =
      Teuchos::rcp(
          new Xpetra::CrsMatrixWrap<T, std::int32_t, std::int64_t, Node>(
              mat_X));
  return A_mat;
}

template <typename T>
Teuchos::RCP<Xpetra::MultiVector<T, std::int32_t, std::int64_t, Node>>
tpetra_to_xpetra(
    Teuchos::RCP<Tpetra::MultiVector<T, std::int32_t, std::int64_t, Node>> mv) {
  return Teuchos::rcp(
      new Xpetra::TpetraMultiVector<T, std::int32_t, std::int64_t, Node>(mv));
}