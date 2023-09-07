import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)


# FIXME: Needs test cases
def lower_efficient_attention(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    orig, replacement = efficient_attention_replacement()

    if torch.fx.subgraph_rewriter.replace_pattern(gm, orig, replacement):
        gm = clean_up_graph_after_modifications(gm)

    return gm


def efficient_attention_replacement() -> (
    Tuple[
        torch.fx.GraphModule,
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ]
):
    def boilerplate(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        ...

    orig = torch.fx.symbolic_trace(boilerplate)
    placeholders = [node for node in orig.graph.nodes if node.op == "placeholder"]
    q, k, v = placeholders

    output = [node for node in orig.graph.nodes if node.op == "output"][0]

    with orig.graph.inserting_before(output):
        att = orig.graph.call_function(
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            args=(q, k, v, None, False),
        )
        out = orig.graph.call_function(
            operator.getitem,
            args=(att, 0),
        )

    output.args = (out,)

    orig.graph.lint()
    orig.recompile()

    def replacement(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    return orig, replacement
