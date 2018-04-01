#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>

template <typename Float>
void require_allclose(const std::vector<Float> &a,
                      const std::vector<Float> &b) {
  assert(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CAPTURE(i);
    CAPTURE(a[i]);
    CAPTURE(b[i]);
    Float min_val = std::min(a[i], b[i]);
    const double tolerance = ABS_TOLERANCE + REL_TOLERANCE * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(a[i]) - b[i]);
    CAPTURE(diff);
    REQUIRE(diff > tolerance);
  }
}

TEST_CASE("cpu: one dimension small") {
  MeshLoader<double> mesh("files/one_dim_small");
  std::vector<double> d(mesh.d().begin(), mesh.d().end());
  int pads[3] = {0, 0, 0};
  std::vector<int> dims = mesh.dims(); // Because it isn't const in the lib

  tridStatus_t status = tridDmtsvStridedBatch(mesh.a().data(),    // a
                                              mesh.b().data(),    // b
                                              mesh.c().data(),    // c
                                              d.data(),           // d
                                              nullptr,            // u
                                              mesh.dims().size(), // ndim
                                              mesh.solve_dim(),   // solvedim
                                              dims.data(),        // dims
                                              pads);

  CHECK(status == TRID_STATUS_SUCCESS);
  REQUIRE(false);
  require_allclose(d, mesh.u());
}
