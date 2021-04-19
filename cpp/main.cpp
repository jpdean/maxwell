
#include "lumping.h"
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

void tpetra_assemble(
    Teuchos::RCP<
        Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>
        A_Tpetra,
    const fem::Form<PetscScalar> &form,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>> &bcs) {

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

  fem::assemble_matrix(tpetra_insert, form, bcs);
}

int main(int argc, char **argv) {
  common::subsystem::init_mpi(argc, argv);
  common::subsystem::init_logging(argc, argv);

  std::size_t n = 20;
  std::shared_ptr<mesh::Mesh> mesh =
      std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
          MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {n, n, n},
          mesh::CellType::tetrahedron, mesh::GhostMode::none));

  // N1curl space for Kc and Mc
  auto V = fem::create_functionspace(functionspace_form_maxwell_Mc, "A", mesh);

  // Lagrange space for Mg
  auto Q = fem::create_functionspace(functionspace_form_maxwell_Mg, "u", mesh);

  const int tdim = mesh->topology().dim();
  // Find facets with bc applied
  const std::vector<std::int32_t> bc_facets = dolfinx::mesh::locate_entities(
      *mesh, tdim - 1,
      [](const xt::xtensor<double, 2> &x) -> xt::xtensor<bool, 1> {
        return xt::isclose(xt::row(x, 0), 0.0) or
               xt::isclose(xt::row(x, 0), 1.0) or
               xt::isclose(xt::row(x, 1), 0.0) or
               xt::isclose(xt::row(x, 1), 1.0) or
               xt::isclose(xt::row(x, 2), 0.0) or
               xt::isclose(xt::row(x, 2), 1.0);
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs =
      dolfinx::fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

  std::cout << "Number of boundary dofs = " << bdofs.size() << std::endl;

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);
  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  common::Timer tcreate("Tpetra: create matrices");
  // Hcurl stiffness matrix
  auto Kc =
      std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_maxwell_Kc, {V, V},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));
  auto Kc_mat = create_tpetra_matrix<PetscScalar>(mesh->mpi_comm(), *Kc);
  tpetra_assemble(Kc_mat, *Kc, {bc});

  std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                    const std::int32_t *, const PetscScalar *)>
      Kc_set = [Kc_mat](std::int32_t nr, const std::int32_t *rows,
                        std::int32_t nc, const std::int32_t *cols,
                        const PetscScalar *data) -> int {
    if (nr > 1 or nc > 1)
      throw std::runtime_error("Error setting diagonal");
    Teuchos::ArrayView<const int> col_view(cols, 1);
    Teuchos::ArrayView<const PetscScalar> data_view(data, 1);
    Kc_mat->replaceLocalValues(*rows, col_view, data_view);
    return 0;
  };

  fem::set_diagonal(Kc_set, *V, {bc});
  Kc_mat->fillComplete();

  // Hcurl mass matrix
  auto Mc =
      std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_maxwell_Mc, {V, V},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));
  auto Mc_mat = create_tpetra_matrix<PetscScalar>(mesh->mpi_comm(), *Mc);
  tpetra_assemble(Mc_mat, *Mc, {});
  Mc_mat->fillComplete();

  // Inverse lumped Hgrad mass matrix
  auto Mg =
      std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_maxwell_Mg, {Q, Q},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));
  la::Vector<PetscScalar> Mg_vec = create_lumped_diagonal(*Mg);

  // Invert local values and insert into the diagonal of a matrix
  const std::vector<PetscScalar> &vals = Mg_vec.array();
  auto Mg_mat =
      create_tpetra_diagonal_matrix<PetscScalar>(Q->dofmap()->index_map);
  std::vector<std::int32_t> col(1);
  std::vector<PetscScalar> val(1);
  for (int i = 0; i < Q->dofmap()->index_map->size_local(); ++i) {
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
    xcoord.interpolate(
        [&j](const xt::xtensor<double, 2> &x) -> xt::xarray<double> {
          return xt::row(x, j);
        });
    time_interpolate.stop();
    for (int i = 0; i < Q->dofmap()->index_map->size_local(); ++i)
      coords->replaceLocalValue(i, j, xcoord.x()->array()[i]);
  }

  tcreate.stop();

  bool write_files = false;

  if (write_files) {
    common::Timer tw("Tpetra: write files");
    Tpetra::MatrixMarket::Writer<
        Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t, Node>>::
        writeSparseFile("D0.mat", *D0_mat, "D0",
                        "Edge-based discrete gradient");

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
  }

  common::Timer t0("refMaxwell::create");
  auto A_Kc = tpetra_to_xpetra(Kc_mat);
  auto A_Mc = tpetra_to_xpetra(Mc_mat);
  auto A_Mg = tpetra_to_xpetra(Mg_mat);
  auto A_D0 = tpetra_to_xpetra(D0_mat);
  auto A_coords = tpetra_to_xpetra(coords);

  Teuchos::RCP<Teuchos::ParameterList> MLList =
      Teuchos::getParametersFromXmlFile("maxwell_solver_settings.xml");
      // Teuchos::getParametersFromXmlFile("Maxwell.xml");
  Teuchos::RCP<MueLu::RefMaxwell<PetscScalar, std::int32_t, std::int64_t, Node>>
      refMaxwell = rcp(
          new MueLu::RefMaxwell<PetscScalar, std::int32_t, std::int64_t, Node>(
              A_Kc, A_D0, A_Mg, A_Mc, Teuchos::null, A_coords, *MLList));
  t0.stop();

  // Create linear problem solver with operator and preconditioner
  auto problem = create_belos_problem<PetscScalar>(A_Kc, refMaxwell);
  using MV = Xpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t, Node>;

  // Solution vector
  Teuchos::RCP<
      Tpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t, Node>>
      x_tp = Teuchos::rcp(
          new Tpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t,
                                  Node>(Kc_mat->getRowMap(), 1));
  Teuchos::RCP<MV> x =
      Teuchos::rcp(new Xpetra::TpetraMultiVector<PetscScalar, std::int32_t,
                                                 std::int64_t, Node>(x_tp));
  x->putScalar(Teuchos::ScalarTraits<PetscScalar>::zero());

  // RHS vector
  Teuchos::RCP<
      Tpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t, Node>>
      b_tp = Teuchos::rcp(
          new Tpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t,
                                  Node>(Kc_mat->getRowMap(), 1));
  Teuchos::RCP<MV> b =
      Teuchos::rcp(new Xpetra::TpetraMultiVector<PetscScalar, std::int32_t,
                                                 std::int64_t, Node>(b_tp));

  // Hcurl RHS vector assemble
  auto Lform =
      std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_maxwell_L, {V},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));
  const int vec_size = V->dofmap()->index_map->size_local() +
                       V->dofmap()->index_map->num_ghosts();
  dolfinx::la::Vector<PetscScalar> _b(V->dofmap()->index_map, 1);
  dolfinx::fem::assemble_vector(xtl::span<PetscScalar>(_b.mutable_array()),
                                *Lform);
  dolfinx::fem::apply_lifting(xtl::span<PetscScalar>(_b.mutable_array()), {Kc},
                              {{bc}}, {}, 1.0);

  dolfinx::la::scatter_rev(_b, dolfinx::common::IndexMap::Mode::add);
  fem::set_bc(xtl::span<PetscScalar>(_b.mutable_array()), {bc});

  std::copy(_b.array().begin(),
            _b.array().begin() + V->dofmap()->index_map->size_local(),
            b->getDataNonConst(0).get());

  problem->setProblem(x, b);

  if (!problem->setProblem())
    throw std::runtime_error(
        "Belos::LinearProblem failed to set up correctly!");

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

  // Copy solution to Function and write to file
  // FIXME There is probably a much better way
  auto x_func = std::make_shared<fem::Function<PetscScalar>>(V);
  std::vector<PetscScalar>& x_func_vec = x_func->x()->mutable_array();
  for(int i = 0; i< vec_size; i++)
  {
    x_func_vec[i] = x->getData(0)[i];
  }

  io::VTKFile file("x.pvd");
  file.write(*x_func);

  return 0;
}
