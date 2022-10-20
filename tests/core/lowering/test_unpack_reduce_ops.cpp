#include <string>
#include "core/compiler.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

TEST(LoweringPasses, UnpackVarLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackVarUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::var(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %5, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %one : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%3, %one)
        %7 : Tensor = aten::std(%x.1, %6, %5, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({1, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdUnbiasedLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %6 : int[] = prim::ListConstruct(%3)
        %7 : Tensor = aten::std(%x.1, %6, %4, %4) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackStdUnbiasedKeepDimsLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %5 : bool = prim::Constant[value=0]() # test_zeros.py:10:65
        %4 : bool = prim::Constant[value=1]() # test_zeros.py:10:50
        %3 : int = prim::Constant[value=0]() # test_zeros.py:10:39
        %one : int = prim::Constant[value=1]()
        %6 : int[] = prim::ListConstruct(%3, %one)
        %7 : Tensor = aten::std(%x.1, %6, %4, %5) # test_zeros.py:10:26
        return (%7))IR";

  auto in = at::randn({4, 3, 3}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackStd(g);
  torch_tensorrt::core::lowering::passes::UnpackVar(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewZerosEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant()
        %3 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_zeros(%x.1, %3, %2, %2, %2, %2)
        return (%5))IR";

  auto in = at::randint(-10, 10, {2, 4, 6, 8}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewZeros(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewZerosDataTypeEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=5]()
        %3 : None = prim::Constant()
        %4 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_zeros(%x.1, %4, %2, %3, %3, %3)
        return (%5))IR";

  auto in = at::rand({1, 3, 5, 7}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewZeros(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewOnesEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant()
        %3 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_ones(%x.1, %3, %2, %2, %2, %2)
        return (%5))IR";

  auto in = at::randint(-10, 10, {2, 4, 6, 8}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewOnes(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewOnesDataTypeEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=5]()
        %3 : None = prim::Constant()
        %4 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_ones(%x.1, %4, %2, %3, %3, %3)
        return (%5))IR";

  auto in = at::rand({1, 3, 5, 7}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewOnes(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewFullEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : None = prim::Constant()
        %fill : int = prim::Constant[value=-1]()
        %3 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_full(%x.1, %3, %fill, %2, %2, %2, %2)
        return (%5))IR";

  auto in = at::randint(-10, 10, {2, 4, 6, 8}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewFull(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(Evaluators, NewFullDataTypeEvaluatesCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : int = prim::Constant[value=5]()
        %fill : int = prim::Constant[value=2.4]()
        %3 : None = prim::Constant()
        %4 : int[] = aten::size(%x.1)
        %5 : Tensor = aten::new_full(%x.1, %4, %fill, %2, %3, %3, %3)
        return (%5))IR";

  auto in = at::rand({1, 3, 5, 7}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackNewFull(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackRsqrtLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : Tensor = aten::rsqrt(%x.1)
        return (%2))IR";

  // Make range [0.01, 1.01] to ensure positives / avoid NaN with negative sqrt
  auto in = at::rand({2, 3, 5, 7}, {at::kCUDA}) + 0.01;

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackRsqrt(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}

TEST(LoweringPasses, UnpackRsqrtIntLowersCorrectly) {
  const auto graph = R"IR(
      graph(%x.1 : Tensor):
        %2 : Tensor = aten::rsqrt(%x.1)
        return (%2))IR";

  // Make range of ints [1, 10]
  auto in = at::randint(1, 11, {2, 3, 5, 7}, {at::kCUDA});

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph, g.get());

  auto jit_pre_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});
  torch_tensorrt::core::lowering::passes::UnpackRsqrt(g);
  torch::jit::EliminateCommonSubexpression(g);
  auto jit_post_results = torch_tensorrt::tests::util::EvaluateGraphJIT(g, {in});

  ASSERT_TRUE(
      torch_tensorrt::tests::util::almostEqual(jit_pre_results[0].toTensor(), jit_post_results[0].toTensor(), 2e-6));
}
