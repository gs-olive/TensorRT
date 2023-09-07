from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.fx.converters.converter_utils import SourceIR, get_trt_tensor
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def linear(
    network: TRTNetwork,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    weight: Union[TRTTensor, torch.Tensor, np.ndarray],
    bias: Optional[Union[TRTTensor, torch.Tensor, np.ndarray]],
) -> TRTTensor:
    # Process weight terms
    if not isinstance(weight, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has weight of type {type(weight)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(weight, (torch.Tensor, np.ndarray)):
        weight = get_trt_tensor(network, weight, f"{name}_weight")

    # Process bias terms
    if bias is not None and not isinstance(bias, (TRTTensor, torch.Tensor, np.ndarray)):
        raise RuntimeError(
            f"Linear layer {name} has bias of type {type(bias)}, Expect Union[TRTTensor, torch.Tensor, np.ndarray],"
        )
    elif isinstance(bias, (torch.Tensor, np.ndarray)):
        bias = get_trt_tensor(network, bias, f"{name}_bias")

    # add IMatrixMultiplyLayer
    out = impl.matmul.matrix_multiply(
        network,
        target,
        source_ir,
        name,
        input,
        weight,
        input_matrix_op=trt.MatrixOperation.NONE,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
    )

    if bias is not None:
        # add bias
        out = impl.elementwise.add(network, target, source_ir, name, out, bias)

    return out
