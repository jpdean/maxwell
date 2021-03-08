
#include "maxwell.h"
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <dolfinx.h>

Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>>
create_tpetra_matrix(MPI_Comm mpi_comm, const fem::Form<PetscScalar> &a) {
  Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));

  dolfinx::la::SparsityPattern pattern =
      dolfinx::fem::create_sparsity_pattern(a);
  pattern.assemble();
  const dolfinx::graph::AdjacencyList<std::int32_t> &diagonal_pattern =
      pattern.diagonal_pattern();
  const dolfinx::graph::AdjacencyList<std::int32_t> &off_diagonal_pattern =
      pattern.off_diagonal_pattern();

  std::vector<std::size_t> nnz(diagonal_pattern.num_nodes());
  for (int i = 0; i < diagonal_pattern.num_nodes(); ++i)
    nnz[i] = diagonal_pattern.num_links(i) + off_diagonal_pattern.num_links(i);

  dolfinx::common::Timer tcre("Trilinos: create sparsity");
  std::vector<std::int64_t> global_indices = pattern.column_indices();

  const std::shared_ptr<const fem::FunctionSpace> V = a.function_spaces()[0];
  const Teuchos::ArrayView<const std::int64_t> global_index_view(
      global_indices.data(), global_indices.size());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> colMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global(), global_index_view, 0, comm));

  const Teuchos::ArrayView<const std::int64_t> global_index_vec_view(
      global_indices.data(), V->dofmap()->index_map->size_local());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> vecMap =
      Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global(), global_index_vec_view, 0,
          comm));

  Teuchos::ArrayView<std::size_t> _nnz(nnz.data(), nnz.size());
  Teuchos::RCP<Tpetra::CrsGraph<std::int32_t, std::int64_t>> crs_graph(
      new Tpetra::CrsGraph<std::int32_t, std::int64_t>(vecMap, colMap, _nnz));

  const std::int64_t nlocalrows = V->dofmap()->index_map->size_local();
  for (std::size_t i = 0; i != diagonal_pattern.num_nodes(); ++i) {
    std::vector<std::int32_t> indices(diagonal_pattern.links(i).begin(),
                                      diagonal_pattern.links(i).end());
    for (std::int32_t q : off_diagonal_pattern.links(i))
      indices.push_back(q);

    Teuchos::ArrayView<std::int32_t> _indices(indices.data(), indices.size());
    crs_graph->insertLocalIndices(i, _indices);
  }

  crs_graph->fillComplete();
  tcre.stop();

  Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>(
              crs_graph));
  return A_Tpetra;
}

int main(int argc, char **argv) {
  common::subsystem::init_mpi(argc, argv);
  common::subsystem::init_logging(argc, argv);

  std::size_t n = 12;
  auto cmap = fem::create_coordinate_map(create_coordinate_map_maxwell);
  std::shared_ptr<mesh::Mesh> mesh =
      std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
          MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n}, cmap,
          mesh::GhostMode::none));

  auto V = fem::create_functionspace(create_functionspace_form_maxwell_Mc, "A",
                                     mesh);

  auto Q = fem::create_functionspace(create_functionspace_form_maxwell_Mg, "u",
                                     mesh);

  auto Mg =
      fem::create_form<PetscScalar>(create_form_maxwell_Mg, {Q, Q}, {}, {}, {});

  auto Mg_mat = create_tpetra_matrix(mesh->mpi_comm(), *Mg);

  // Lump mass matrix of Mg into diagonal vector
  la::Vector<PetscScalar> Mg_vec(Q->dofmap()->index_map, 1);
  std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                    const std::int32_t *, const PetscScalar *)>
      lumper = [&Mg_vec](int nr, const int *rows, int nc, const int *cols,
                         const PetscScalar *vals) {
        std::vector<PetscScalar> &Mg_data = Mg_vec.mutable_array();
        for (int i = 0; i < nr; ++i) {
          for (int j = 0; j < nc; ++j) {
            Mg_data[rows[i]] += vals[i * nc + j];
          }
        }
        return 0;
      };

  fem::assemble_matrix(lumper, *Mg, {});
  // TODO: 'export' values to other processes

  return 0;
}