import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._decomp import register_decomposition
from torch._ops import OpOverload

from ._decomposition_groups import (
    ENABLED_TORCH_DECOMPOSITIONS,
    TORCH_TRT_DECOMPOSITIONS,
    _core_aten_decompositions,
    aten,
    torch_disabled_decompositions,
    torch_enabled_decompositions,
)

logger = logging.getLogger(__name__)


def register_torch_trt_decomposition(
    aten_op: OpOverload, registry: Optional[Any] = None
) -> Callable[[Any], Any]:
    """Checks if the decomposition already exists in one of the sets
    Registers the decomposition via the Torch utility

    Alerts the user if the decomposition already exists, before registering
    Throws an AssertionError if the user attempts to register a decomposition
    which is present in the set of explicitly disabled decompositions
    """
    if aten_op in torch_enabled_decompositions:
        logger.warning(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in torch_enabled_decompositions. "
            "The custom implementation will take precedence."
        )
    elif aten_op in torch_disabled_decompositions:
        logger.info(
            f"Detected custom decomposition for {aten_op}, which is present "
            "in torch_disabled_decompositions."
        )

    # Conflicts with _core_aten_decompositions will only occur if
    # enable_experimental_decompositions is True in get_decompositions
    if aten_op in _core_aten_decompositions:
        logger.debug(
            f"Detected custom decomposition for {aten_op}, which conflicts "
            "with an existing Torch decomposition in core_aten_decompositions. "
            "The custom implementation will take precedence."
        )

    def register(fn: Callable[[Any], Any]) -> Any:
        return register_decomposition(aten_op=aten_op, registry=registry)(fn)

    return register


def replace_inplace_op(aten_op: OpOverload, outplace_op: OpOverload) -> Any:
    """Replace inplace operation with functional equivalent
    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_torch_trt_decomposition(aten_op, registry=TORCH_TRT_DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):  # type: ignore
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


replace_inplace_op(aten.add_, aten.add)
replace_inplace_op(aten.addbmm_, aten.addbmm)
replace_inplace_op(aten.addmm_, aten.addmm)
replace_inplace_op(aten.addmv_, aten.addmv)
replace_inplace_op(aten.baddbmm_, aten.baddbmm)
replace_inplace_op(aten.cumprod_, aten.cumprod)
replace_inplace_op(aten.index_put_, aten.index_put)
replace_inplace_op(aten.index_reduce_, aten.index_reduce)
replace_inplace_op(aten.relu_, aten.relu)
replace_inplace_op(aten.round_, aten.round)
replace_inplace_op(aten.scatter_, aten.scatter)
replace_inplace_op(aten.scatter_add_, aten.scatter_add)
replace_inplace_op(aten.scatter_reduce_, aten.scatter_reduce)


@register_torch_trt_decomposition(aten.rsqrt, registry=TORCH_TRT_DECOMPOSITIONS)
def rsqrt_replacement(*args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reciprocal(torch.sqrt(*args, **kwargs))


@register_torch_trt_decomposition(aten._unsafe_view, registry=TORCH_TRT_DECOMPOSITIONS)
def unsafe_view_replacement(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
    return torch.reshape(x, *args, **kwargs)


@register_torch_trt_decomposition(
    torch.ops.aten.lift_fresh_copy, registry=TORCH_TRT_DECOMPOSITIONS
)
def lift_fresh_copy_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(aten.alias, registry=TORCH_TRT_DECOMPOSITIONS)
def alias_replacement(x: torch.Tensor) -> torch.Tensor:
    return x


@register_torch_trt_decomposition(
    torch.ops.aten.addmm, registry=TORCH_TRT_DECOMPOSITIONS
)
def addmm_replacement(
    input_: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: int = 1,
    alpha: int = 1,
) -> torch.Tensor:
    return torch.add(
        torch.mul(input_, beta), torch.mul(torch.matmul(mat1, mat2), alpha)
    )


@register_torch_trt_decomposition(
    torch.ops.aten.reciprocal.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def reciprocal_replacement(
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.div(1, input_)


@register_torch_trt_decomposition(
    torch.ops.prims.var.default, registry=TORCH_TRT_DECOMPOSITIONS
)
def var_decomposition(
    input: torch.Tensor,
    dims: Optional[List[int]],
    correction: int,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dims is None:
        dims = []

    if isinstance(dims, (tuple, list)) and len(dims) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dims:
            n *= input.shape[dim_i]

    mean = torch.mean(input, dims, keepdim=True)
    sub = input - mean
    sq = sub * sub
    sum = torch.sum(sq, dims, keepdim=False)

    if correction is None:
        denom = float(n - 1)
    else:
        if isinstance(correction, int):
            denom = float(n - correction)
        elif isinstance(correction, float):
            denom = float(n) - correction
        else:
            raise RuntimeError("correction must be int or float")

    var = sum / max(0, denom)

    return var


def get_decompositions(
    enable_experimental_decompositions: bool = False,
) -> Dict[OpOverload, Callable[[Any], Any]]:
    if enable_experimental_decompositions:
        CORE_ATEN_DECOMPOSITIONS_FILTERED: Dict[OpOverload, Callable[[Any], Any]] = {
            decomp: _core_aten_decompositions[decomp]
            for decomp in _core_aten_decompositions
            if decomp not in torch_disabled_decompositions
        }
        return {**CORE_ATEN_DECOMPOSITIONS_FILTERED, **TORCH_TRT_DECOMPOSITIONS}
    else:
        return {**ENABLED_TORCH_DECOMPOSITIONS, **TORCH_TRT_DECOMPOSITIONS}
