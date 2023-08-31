from typing import Callable

import torch
from torch.fx.passes.pass_manager import PassManager

from .constant_folding import constant_fold

# Import and order lowering passes
from .remove_input_alias_fixing_clones import remove_input_alias_fixing_clones
from .repair_input_as_output import repair_input_as_output

ATEN_LOWERING_PASSES = PassManager.build_from_passlist(
    [
        remove_input_alias_fixing_clones,
        constant_fold,
        repair_input_as_output,
    ]
)


def add_lowering_pass(
    lowering_pass: Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
) -> None:
    """Adds a lowering pass to the registry"""
    ATEN_LOWERING_PASSES.add_pass(lowering_pass)
    return


def apply_lowering_passes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Applies the lowering passes to a graph module, returns the modified GraphModule"""
    return ATEN_LOWERING_PASSES(gm)
