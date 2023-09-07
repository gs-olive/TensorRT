import logging
from typing import Sequence

import torch
from torch.fx.passes.shape_prop import ShapeProp
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def fuse_prims_broadcast(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    modified_graph = False
    ShapeProp(gm).propagate(*sample_inputs)

    for node in gm.graph.nodes:
        # If the node is a sum or var prims operator
        if (
            node.target in (torch.ops.prims.sum.default, torch.ops.prims.var.default)
            and len(node.users) == 1
            and list(node.users)[0].target == torch.ops.prims.broadcast_in_dim.default
        ):
            broadcast_node = list(node.users)[0]
            broadcasted_shape = broadcast_node.args[1]
            reduced_dims = node.args[1]
            original_shape = node.args[0].meta["tensor_meta"].shape

            if (
                len(broadcasted_shape) == len(original_shape)
                and all(broadcasted_shape[i] == 1 for i in reduced_dims)
                and all(
                    broadcasted_shape[j] == original_shape[j]
                    for j in range(len(original_shape))
                    if j not in reduced_dims
                )
            ):
                with gm.graph.inserting_after(broadcast_node):
                    modified_graph = True

                    if node.target == torch.ops.prims.sum.default:
                        fused_node = gm.graph.call_function(
                            torch.ops.aten.sum.dim_IntList,
                            args=(node.args[0], reduced_dims, True),
                        )
                    elif node.target == torch.ops.prims.var.default:
                        fused_node = gm.graph.call_function(
                            torch.ops.aten.var.correction,
                            args=(node.args[0], reduced_dims, True),
                            kwargs={
                                "correction": node.kwargs.get("correction", None),
                                "keepdim": True,
                            },
                        )

                # Replace all uses of the placeholder except the cloned node
                # with the cloned placeholder
                broadcast_node.replace_all_uses_with(
                    fused_node,
                )

                gm.graph.erase_node(broadcast_node)
                gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Fused prims-broadcast paradigm:\n{gm.graph}")

    return gm
