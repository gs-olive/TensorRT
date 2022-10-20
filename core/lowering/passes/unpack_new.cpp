#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackNewZeros(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string new_zeros_pattern = R"IR(
    graph(%input, %size, %dtype, %layout, %device, %pin_memory):
      %out: Tensor = aten::new_zeros(%input, %size, %dtype, %layout, %device, %pin_memory)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%input, %size, %dtype, %layout, %device, %pin_memory):
      %none: NoneType = prim::Constant()
      %dtypeIsNone: bool = aten::__is__(%dtype, %none)
      %deviceIsNone: bool = aten::__is__(%device, %none)
      %dtype_mod: int = prim::If(%dtypeIsNone)
        block0():
          # Compute original dtype of input tensor
          %dtype_input: int = prim::dtype(%input)
          -> (%dtype_input)
        block1():
          -> (%dtype)
      %device_mod: Device = prim::If(%deviceIsNone)
        block0():
          # Compute original device of input tensor
          %device_input: Device = prim::device(%input)
          -> (%device_input)
        block1():
          -> (%device)
      %out: Tensor = aten::zeros(%size, %dtype_mod, %layout, %device_mod, %pin_memory)
      return (%out))IR";

  torch::jit::SubgraphRewriter new_zeros_rewriter;
  new_zeros_rewriter.RegisterRewritePattern(new_zeros_pattern, unpacked_pattern);
  new_zeros_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack new_zeros: " << *graph);
}

void UnpackNewOnes(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string new_ones_pattern = R"IR(
    graph(%input, %size, %dtype, %layout, %device, %pin_memory):
      %out: Tensor = aten::new_ones(%input, %size, %dtype, %layout, %device, %pin_memory)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%input, %size, %dtype, %layout, %device, %pin_memory):
      %none: NoneType = prim::Constant()
      %dtypeIsNone: bool = aten::__is__(%dtype, %none)
      %deviceIsNone: bool = aten::__is__(%device, %none)
      %dtype_mod: int = prim::If(%dtypeIsNone)
        block0():
          # Compute original dtype of input tensor
          %dtype_input: int = prim::dtype(%input)
          -> (%dtype_input)
        block1():
          -> (%dtype)
      %device_mod: Device = prim::If(%deviceIsNone)
        block0():
          # Compute original device of input tensor
          %device_input: Device = prim::device(%input)
          -> (%device_input)
        block1():
          -> (%device)
      %out: Tensor = aten::ones(%size, %dtype_mod, %layout, %device_mod, %pin_memory)
      return (%out))IR";

  torch::jit::SubgraphRewriter new_ones_rewriter;
  new_ones_rewriter.RegisterRewritePattern(new_ones_pattern, unpacked_pattern);
  new_ones_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack new_ones: " << *graph);
}

void UnpackNewFull(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string new_full_pattern = R"IR(
    graph(%input, %size, %fill_value, %dtype, %layout, %device, %pin_memory):
      %out: Tensor = aten::new_full(%input, %size, %fill_value, %dtype, %layout, %device, %pin_memory)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%input, %size, %fill_value, %dtype, %layout, %device, %pin_memory):
      %none: NoneType = prim::Constant()
      %dtypeIsNone: bool = aten::__is__(%dtype, %none)
      %deviceIsNone: bool = aten::__is__(%device, %none)
      %dtype_mod: int = prim::If(%dtypeIsNone)
        block0():
          # Compute original dtype of input tensor
          %dtype_input: int = prim::dtype(%input)
          -> (%dtype_input)
        block1():
          -> (%dtype)
      %device_mod: Device = prim::If(%deviceIsNone)
        block0():
          # Compute original device of input tensor
          %device_input: Device = prim::device(%input)
          -> (%device_input)
        block1():
          -> (%device)
      %out: Tensor = aten::full(%size, %fill_value, %dtype_mod, %layout, %device_mod, %pin_memory)
      return (%out))IR";

  torch::jit::SubgraphRewriter new_full_rewriter;
  new_full_rewriter.RegisterRewritePattern(new_full_pattern, unpacked_pattern);
  new_full_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack new_full: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
