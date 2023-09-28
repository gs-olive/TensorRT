import math
from typing import Optional, Union

import tensorrt as trt
from torch.fx.node import Target
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion.converter_utils import SourceIR
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor


def scaled_dot_product_attention(
    network: TRTNetwork,
    target: Union[Target, str],
    source_ir: Optional[SourceIR],
    name: str,
    query: TRTTensor,
    key: TRTTensor,
    value: TRTTensor,
) -> TRTTensor:
    mm = impl.matmul.matrix_multiply(
        network,
        target,
        source_ir,
        name + "_mm",
        query,
        key,
        other_matrix_op=trt.MatrixOperation.TRANSPOSE,
    )
    div = impl.elementwise.div(
        network,
        target,
        source_ir,
        name + "_scale",
        mm,
        math.sqrt(query.shape[-1]),
    )
    softmax = impl.normalization.softmax(
        network, target, source_ir, name + "_softmax", div, -1
    )
    out = impl.matmul.matrix_multiply(
        network,
        target,
        source_ir,
        name + "_out",
        softmax,
        value,
    )

    return out
