
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/Vector.h>

template <typename T>
dolfinx::la::Vector<T>
create_lumped_diagonal(const dolfinx::fem::Form<T> &form) {
  // Lump mass matrix into diagonal vector
  std::shared_ptr<const dolfinx::common::IndexMap> qmap =
      form.function_spaces()[0]->dofmap()->index_map;

  dolfinx::la::Vector<T> vec(qmap, 1);
  std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                    const std::int32_t *, const T *)>
      lumper = [&vec](int nr, const int *rows, int nc, const int *cols,
                      const T *vals) {
        std::vector<T> &data = vec.mutable_array();
        for (int i = 0; i < nr; ++i) {
          for (int j = 0; j < nc; ++j) {
            data[rows[i]] += vals[i * nc + j];
          }
        }
        return 0;
      };

  dolfinx::fem::assemble_matrix(lumper, form, {});
  // Gather and add ghost entries
  dolfinx::la::scatter_rev(vec, dolfinx::common::IndexMap::Mode::add);
  return vec;
}