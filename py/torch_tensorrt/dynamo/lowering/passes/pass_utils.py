import torch


def clean_up_graph_after_modifications(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place"""
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm
