from typing import Callable, Sequence

import torch

from .constant_folding import constant_fold
from .fuse_prims_broadcast import fuse_prims_broadcast
from .lower_efficient_attention import lower_efficient_attention

# Import and order lowering passes
from .pass_manager import DynamoPassManager
from .remove_input_alias_fixing_clones import remove_input_alias_fixing_clones
from .repair_input_as_output import repair_input_as_output

ATEN_LOWERING_PASSES = DynamoPassManager.build_from_passlist(
    [
        remove_input_alias_fixing_clones,
        constant_fold,
        repair_input_as_output,
        lower_efficient_attention,
        fuse_prims_broadcast,
    ]
)


def add_lowering_pass(
    lowering_pass: Callable[
        [torch.fx.GraphModule, Sequence[torch.Tensor]], torch.fx.GraphModule
    ]
) -> None:
    """Adds a lowering pass to the registry"""
    ATEN_LOWERING_PASSES.add_pass(lowering_pass)
    return


def apply_lowering_passes(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module, returns the modified GraphModule"""
    return ATEN_LOWERING_PASSES(gm, sample_inputs)
