
#include "maxwell.h"
#include "tpetra_util.h"

#include <MatrixMarket_Tpetra.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_RefMaxwell.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <dolfinx.h>

#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrixFactory.hpp>
#include <Xpetra_IO.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_MultiVector.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_Parameters.hpp>
#include <Xpetra_Vector.hpp>

#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosXpetraAdapter.hpp>

void tpetra_assemble(Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t,
                                                    std::int64_t, Node>>
                         A_Tpetra,
                     const fem::Form<PetscScalar> &form) {

  std::vector<std::int64_t> global_cols; // temp for columns
  const std::shared_ptr<const fem::FunctionSpace> V = form.function_spaces()[0];
  const std::int64_t nlocalrows = V->dofmap()->index_map->size_local();
  std::vector<std::int64_t> global_indices =
      V->dofmap()->index_map->global_indices();

  std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                    const std::int32_t *, const PetscScalar *)>
      tpetra_insert = [&A_Tpetra, &global_indices, &global_cols, &nlocalrows](
                          std::int32_t nr, const std::int32_t *rows,
                          const std::int32_t nc, const std::int32_t *cols,
                          const PetscScalar *data) {
        for (std::int32_t i = 0; i < nr; ++i) {
          Teuchos::ArrayView<const PetscScalar> data_view(data + i * nc, nc);
          if (rows[i] < nlocalrows) {
            Teuchos::ArrayView<const int> col_view(cols, nc);
            int nvalid =
                A_Tpetra->sumIntoLocalValues(rows[i], col_view, data_view);
            if (nvalid != nc)
              throw std::runtime_error("Inserted " + std::to_string(nvalid) +
                                       "/" + std::to_string(nc) + " on row:" +
                                       std::to_string(global_indices[rows[i]]));
          } else {
            global_cols.resize(nc);
            for (int j = 0; j < nc; ++j)
              global_cols[j] = global_indices[cols[j]];
            int nvalid = A_Tpetra->sumIntoGlobalValues(global_indices[rows[i]],
                                                       global_cols, data_view);
            if (nvalid != nc)
              throw std::runtime_error("Inserted " + std::to_string(nvalid) +
                                       "/" + std::to_string(nc) + " on row:" +
                                       std::to_string(global_indices[rows[i]]));
          }
        }
        return 0;
      };

  fem::assemble_matrix(tpetra_insert, form, {});
}

int main(int argc, char **argv) {
  common::subsystem::init_mpi(argc, argv);
  common::subsystem::init_logging(argc, argv);

  std::size_t n = 10;
  auto cmap = fem::create_coordinate_map(create_coordinate_map_maxwell);
  std::shared_ptr<mesh::Mesh> mesh =
      std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
          MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n}, cmap,
          mesh::GhostMode::none));

  // N1curl space for Kc and Mc
  auto V = fem::create_functionspace(create_functionspace_form_maxwell_Mc, "A",
                                     mesh);

  // Lagrange space for Mg
  auto Q = fem::create_functionspace(create_functionspace_form_maxwell_Mg, "u",
                                     mesh);

  common::Timer tcreate("Tpetra: create matrices");
  // Hcurl stiffness matrix
  auto Kc =
      fem::create_form<PetscScalar>(create_form_maxwell_Kc, {V, V}, {}, {}, {});
  dolfinx::la::SparsityPattern Kc_pattern =
      dolfinx::fem::create_sparsity_pattern(*Kc);
  Kc_pattern.assemble();
  auto Kc_mat = create_tpetra_matrix<PetscScalar>(mesh->mpi_comm(), Kc_pattern);
  tpetra_assemble(Kc_mat, *Kc);
  Kc_mat->fillComplete();

  // Hcurl mass matrix
  auto Mc =
      fem::create_form<PetscScalar>(create_form_maxwell_Mc, {V, V}, {}, {}, {});
  dolfinx::la::SparsityPattern Mc_pattern =
      dolfinx::fem::create_sparsity_pattern(*Mc);
  Mc_pattern.assemble();
  auto Mc_mat = create_tpetra_matrix<PetscScalar>(mesh->mpi_comm(), Mc_pattern);
  tpetra_assemble(Mc_mat, *Mc);
  Mc_mat->fillComplete();

  // Inverse lumped Hgrad mass matrix
  auto Mg =
      fem::create_form<PetscScalar>(create_form_maxwell_Mg, {Q, Q}, {}, {}, {});
  std::shared_ptr<const common::IndexMap> qmap = Q->dofmap()->index_map;
  // Lump mass matrix into diagonal vector
  la::Vector<PetscScalar> Mg_vec(qmap, 1);
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
  // Gather and add ghost entries
  la::scatter_rev(Mg_vec, common::IndexMap::Mode::add);

  // Invert local values and insert into the diagonal of a matrix
  const std::vector<PetscScalar> &vals = Mg_vec.array();
  auto Mg_mat = create_tpetra_diagonal_matrix<PetscScalar>(qmap);
  std::vector<std::int32_t> col(1);
  std::vector<PetscScalar> val(1);
  for (int i = 0; i < qmap->size_local(); ++i) {
    col[0] = i;
    val[0] = 1.0 / vals[i];
    Mg_mat->replaceLocalValues(i, col, val);
  }
  Mg_mat->fillComplete();

  // Discrete gradient matrix
  la::SparsityPattern D0_sp = fem::create_sparsity_discrete_gradient(*V, *Q);
  auto D0_mat = create_tpetra_matrix<PetscScalar>(mesh->mpi_comm(), D0_sp);

  std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                    const std::int32_t *, const PetscScalar *)>
      mat_set_dg = [&D0_mat](int nr, const int *rows, int nc, const int *cols,
                             const PetscScalar *data) {
        for (int i = 0; i < nr; ++i) {
          Teuchos::ArrayView<const PetscScalar> data_view(data + i * nc, nc);
          Teuchos::ArrayView<const int> col_view(cols, nc);
          D0_mat->replaceLocalValues(rows[i], col_view, data_view);
        }
        return 0;
      };

  fem::assemble_discrete_gradient(mat_set_dg, *V, *Q);
  D0_mat->fillComplete();

  // Get nodal coordinates
  Teuchos::RCP<Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>>
      coords = Teuchos::rcp(
          new Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>(
              Mg_mat->getRowMap(), 3));
  fem::Function<double> xcoord(Q);
  for (int j = 0; j < 3; ++j) {
    common::Timer time_interpolate("Interpolate x");
    xcoord.interpolate([&j](auto &x) {
      return std::vector<PetscScalar>(x.row(j).begin(), x.row(j).end());
    });
    time_interpolate.stop();
    for (int i = 0; i < Q->dofmap()->index_map->size_local(); ++i)
      coords->replaceLocalValue(i, j, xcoord.x()->array()[i]);
  }

  tcreate.stop();

  common::Timer tw("Tpetra: write files");
  Tpetra::MatrixMarket::Writer<
      Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>::
      writeSparseFile("D0.mat", *D0_mat, "D0", "Edge-based discrete gradient");

  Tpetra::MatrixMarket::Writer<
      Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>::
      writeSparseFile("Mg.mat", *Mg_mat, "Mg",
                      "Lumped inverse Hgrad mass matrix");

  Tpetra::MatrixMarket::Writer<
      Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>::
      writeSparseFile("Mc.mat", *Mc_mat, "Mc", "Hcurl mass matrix");

  Tpetra::MatrixMarket::Writer<
      Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>::
      writeSparseFile("Kc.mat", *Kc_mat, "Kc", "Hcurl stiffness matrix");

  Tpetra::MatrixMarket::Writer<
      Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>>::
      writeDenseFile("coords.mat", *coords, "coords", "Nodal coordinates");
  tw.stop();

  common::Timer t0("refMaxwell::create");
  auto A_Kc = tpetra_to_xpetra(Kc_mat);
  auto A_Mc = tpetra_to_xpetra(Mc_mat);
  auto A_Mg = tpetra_to_xpetra(Mg_mat);
  auto A_D0 = tpetra_to_xpetra(D0_mat);
  auto A_coords = tpetra_to_xpetra(coords);

  Teuchos::RCP<Teuchos::ParameterList> MLList =
      Teuchos::getParametersFromXmlFile("Maxwell.xml");
  Teuchos::RCP<MueLu::RefMaxwell<PetscScalar, std::int32_t, std::int64_t, Node>>
      refMaxwell = rcp(
          new MueLu::RefMaxwell<PetscScalar, std::int32_t, std::int64_t, Node>(
              A_Kc, A_D0, A_Mg, A_Mc, Teuchos::null, A_coords, *MLList));

  // Create linear problem solver
  using MV = Xpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t, Node>;
  Teuchos::RCP<Belos::OperatorT<MV>> belosOp = Teuchos::rcp(
      new Belos::XpetraOp<PetscScalar, std::int32_t, std::int64_t, Node>(
          A_Kc)); // Turns a Xpetra::Matrix object into a Belos operator
  t0.stop();

  dolfinx::common::Timer t1("Belos::SetOperator");
  Teuchos::RCP<Belos::LinearProblem<PetscScalar, MV, Belos::OperatorT<MV>>>
      problem = rcp(
          new Belos::LinearProblem<PetscScalar, MV, Belos::OperatorT<MV>>());
  problem->setOperator(belosOp);
  t1.stop();

  common::Timer t2("Belos::problem::setProblem");
  Teuchos::RCP<Belos::OperatorT<MV>> belosPrecOp = Teuchos::rcp(
      new Belos::XpetraOp<PetscScalar, std::int32_t, std::int64_t, Node>(
          refMaxwell));
  problem->setRightPrec(belosPrecOp);

  // Solution and RHS vectors
  Teuchos::RCP<Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>>
      x_tp = Teuchos::rcp(
          new Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>(
              Kc_mat->getRowMap(), 1));
  Teuchos::RCP<MV> x = Teuchos::rcp(
      new Xpetra::TpetraMultiVector<double, std::int32_t, std::int64_t, Node>(
          x_tp));
  x->putScalar(Teuchos::ScalarTraits<PetscScalar>::zero());

  Teuchos::RCP<Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>>
      b_tp = Teuchos::rcp(
          new Tpetra::MultiVector<double, std::int32_t, std::int64_t, Node>(
              Kc_mat->getRowMap(), 1));
  Teuchos::RCP<MV> b = Teuchos::rcp(
      new Xpetra::TpetraMultiVector<double, std::int32_t, std::int64_t, Node>(
          b_tp));

  // Hcurl RHS vector assemble
  auto Lform =
      fem::create_form<PetscScalar>(create_form_maxwell_L, {V}, {}, {}, {});
  const int vec_size = V->dofmap()->index_map->size_local() +
                       V->dofmap()->index_map->num_ghosts();
  dolfinx::la::Vector<PetscScalar> _b(V->dofmap()->index_map, 1);
  dolfinx::fem::assemble_vector(tcb::span<PetscScalar>(_b.mutable_array()),
                                *Lform);
  dolfinx::la::scatter_rev(_b, dolfinx::common::IndexMap::Mode::add);
  std::copy(_b.array().begin(),
            _b.array().begin() + V->dofmap()->index_map->size_local(),
            b->getDataNonConst(0).get());

  problem->setProblem(x, b);

  if (!problem->setProblem())
    throw std::runtime_error(
        "Belos::LinearProblem failed to set up correctly!");
  t2.stop();

  common::Timer t3("Belos::solver::setProblem");
  // Belos solver
  Teuchos::RCP<Teuchos::ParameterList> solver_params =
      Teuchos::getParametersFromXmlFile("Belos.xml");
  Teuchos::RCP<Belos::SolverFactory<PetscScalar, MV, Belos::OperatorT<MV>>>
      factory = Teuchos::rcp(
          new Belos::SolverFactory<PetscScalar, MV, Belos::OperatorT<MV>>());
  Teuchos::RCP<Belos::SolverManager<PetscScalar, MV, Belos::OperatorT<MV>>>
      solver = factory->create("Block CG", solver_params);
  solver->setProblem(problem);
  t3.stop();

  common::Timer t5("Belos::solve");
  Belos::ReturnType status = solver->solve();
  t5.stop();
  int iters = solver->getNumIters();
  bool success = (iters < 500 && status == Belos::Converged);
  if (success)
    std::cout << "SUCCESS! Belos converged in " << iters << " iterations."
              << std::endl;
  else
    std::cout << "FAILURE! Belos did not converge fast enough." << std::endl;

  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  return 0;
}
