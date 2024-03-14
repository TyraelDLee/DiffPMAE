#include <string>

#include <torch/extension.h>

#include "R:\\Documents\\developing\\COMP5702_04\\models\\metrics\\pytorch_structural_losses\\pybind\\extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("ApproxMatch", &ApproxMatch);
  m.def("MatchCost", &MatchCost);
  m.def("MatchCostGrad", &MatchCostGrad);
  m.def("NNDistance", &NNDistance);
  m.def("NNDistanceGrad", &NNDistanceGrad);
}
