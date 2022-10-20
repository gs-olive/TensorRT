#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackMaskedFill_(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string masked_fill_pattern = R"IR(
    graph(%self, %mask, %value):
      %out: Tensor = aten::masked_fill_(%self, %mask, %value)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%self, %mask, %value):
      %out: Tensor = aten::masked_fill(%self, %mask, %value)
      return (%out))IR";

  torch::jit::SubgraphRewriter masked_fill_rewriter;
  masked_fill_rewriter.RegisterRewritePattern(masked_fill_pattern, unpacked_pattern);
  masked_fill_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack masked_fill_: " << *graph);
}

void UnpackFill_(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string fill_pattern = R"IR(
    graph(%self, %value):
      %out: Tensor = aten::fill_(%self, %value)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%self, %value):
      %size: int[] = aten::size(%self)
      %dtype: int = prim::dtype(%self)
      %device: Device = prim::device(%self)
      %pin_memory: None = prim::Constant()
      %layout: None = prim::Constant()
      %scalar: Scalar = aten::ScalarImplicit(%value)
      %out: Tensor = aten::full(%size, %scalar, %dtype, %layout, %device, %pin_memory)
      return (%out))IR";

  torch::jit::SubgraphRewriter fill_rewriter;
  fill_rewriter.RegisterRewritePattern(fill_pattern, unpacked_pattern);
  fill_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack fill_: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
