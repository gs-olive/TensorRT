from typing import List, Optional

import tensorrt as trt
import torch
from torch.fx.node import Target, _get_qualified_name
from torch_tensorrt.fx.converters.converter_utils import (
    Frameworks,
    unified_dtype_converter,
)
from torch_tensorrt.fx.types import TRTDataType, TRTNetwork, TRTTensor

from .._SourceIR import SourceIR


def dynamic_unsupported(node: torch.fx.Node) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    # Check node value itself
    if getattr(node.meta["val"], "_has_symbolic_sizes_strides", False):
        return False

    # Check node arguments individually
    if any(
        getattr(arg.meta["val"], "_has_symbolic_sizes_strides", False)
        for arg in node.args
        if isinstance(arg, torch.fx.Node)
    ):
        return False

    # Check node keyword arguments individually
    if any(
        getattr(kwarg.meta["val"], "_has_symbolic_sizes_strides", False)
        for kwarg in node.kwargs.values()
        if isinstance(kwarg, torch.fx.Node)
    ):
        return False

    return True


def cast_trt_tensor(
    network: TRTNetwork,
    input_val: TRTTensor,
    dtype: TRTDataType,
    name: str,
    target: Target = "",
    source_ir: Optional[SourceIR] = None,
) -> TRTTensor:
    """
    Given a TRT Tensor, convert that Tensor to the specified dtype
    Adds an Identity layer to the network which performs the conversion
    Args:
        network (TRTNetwork): A TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType, torch.dtype, np.dtype): The data type to cast the input Tensor to
        name (str): Name of the calling layer
        target (Target): Target of calling node
        source_ir (SourceIR): SourceIR of calling converter
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    trt_dtype = unified_dtype_converter(dtype, Frameworks.TRT)

    if input_val.dtype != trt_dtype:
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_name = (
            f"{source_ir}_ops{'.' + target if target else ''}"
            if (isinstance(target, str))
            else f"{source_ir}_ops.{_get_qualified_name(target)}"
        )

        identity_layer = network.add_identity(input_val)
        identity_layer.set_output_type(0, trt_dtype)
        identity_layer.name = f"Cast ITensor {input_val.name} from {input_val.dtype} to {trt_dtype} -{name}-[{target_name}]-[{name}]"
        return identity_layer.get_output(0)
    else:
        return input_val


def cast_int_int_div_trt_tensor(
    network: TRTNetwork,
    lhs_val: TRTTensor,
    rhs_val: TRTTensor,
    name: str,
) -> List[TRTTensor]:
    """
    Given two `int` data type TRT Tensor to div operation, cast the TRT Tensor to float type
    Args:
        network (TRTNetwork): A TensorRT network
        lhs_val (TRTTensor): A TRT Tensor numerator
        rhs_val (TRTTensor): A TRT Tensor numerator
        name (str): Name of calling layer
    Returns:
        A list of lhs_val and rhs_val casted to the approriate datatype
    """
    if (lhs_val.dtype == trt.int8 or lhs_val.dtype == trt.int32) and (
        rhs_val.dtype == trt.int8 or rhs_val.dtype == trt.int32
    ):
        lhs_val = cast_trt_tensor(network, lhs_val, trt.float32, name)
        rhs_val = cast_trt_tensor(network, rhs_val, trt.float32, name)
    return [lhs_val, rhs_val]


def broadcastable(
    a: TRTTensor,
    b: TRTTensor,
) -> bool:
    "Check if two tensors are broadcastable according to torch rules"
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    # check from the trailing
    diff = len(a_shape) - len(b_shape)
    if diff == 0:
        return True
    if diff > 0:
        max = len(a_shape)
        min = len(b_shape)
        greater_tensor = a_shape
        lesser_tensor = b_shape
    elif diff < 0:
        max = len(b_shape)
        min = len(a_shape)
        greater_tensor = b_shape
        lesser_tensor = a_shape
    j = min - 1
    for i in range(max - 1, diff - 1, -1):
        if not (
            greater_tensor[i] != lesser_tensor[j]
            and (greater_tensor[i] == 1 or lesser_tensor[i] == 1)
        ):
            return False
    return True
