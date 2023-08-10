from typing import Dict, Tuple
import torch
from torch._custom_op.impl import custom_op
from torch.fx.node import Argument, Target

from torch_tensorrt.dynamo.converter_registry import dynamo_tensorrt_converter
from torch_tensorrt.fx.converters import acc_ops_converters
from torch_tensorrt.fx.types import TRTNetwork, TRTTensor

from torch_tensorrt.dynamo.backend.lowering import register_substitution


@custom_op(
    qualname="tensorrt::batch_norm",
    manual_schema="(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> Tensor",
)
def batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps):
    # Defines operator schema, name, namespace, and function header
    ...


@batch_norm.impl("cpu")
@batch_norm.impl("cuda")
@batch_norm.impl_abstract()
def batch_norm_generic(
    *args,
    **kwargs,
):
    # Defines an implementation for AOT Autograd to use for shape analysis/propagation
    return torch.nn.functional.batch_norm(
        *args,
        **kwargs,
    )


@register_substitution(torch.nn.BatchNorm1d, torch.ops.tensorrt.batch_norm)
@register_substitution(torch.nn.BatchNorm2d, torch.ops.tensorrt.batch_norm)
@register_substitution(torch.nn.BatchNorm3d, torch.ops.tensorrt.batch_norm)
def batch_norm_insertion_fn(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    submodule: torch.nn.Module,
) -> torch.fx.Node:
    weight = gm.graph.get_attr(node.target + ".weight", torch.Tensor)
    bias = gm.graph.get_attr(node.target + ".bias", torch.Tensor)
    running_mean = gm.graph.get_attr(node.target + ".running_mean", torch.Tensor)
    running_var = gm.graph.get_attr(node.target + ".running_var", torch.Tensor)

    new_node = gm.graph.call_function(
        torch.ops.tensorrt.batch_norm,
        args=node.args,
        kwargs={
            "weight": weight,
            "bias": bias,
            "running_mean": running_mean,
            "running_var": running_var,
            "eps": submodule.eps,
            "momentum": submodule.momentum,
            "training": submodule.training,
        },
    )

    return new_node


@dynamo_tensorrt_converter(torch.ops.tensorrt.batch_norm.default)
def tensorrt_batch_norm(
    network: TRTNetwork,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> TRTTensor:
    import pdb

    pdb.set_trace()
    # Defines converter replacing the default operator for this function
    kwargs_new = {
        "input": args[0],
        "kernel_size": args[1],
        "stride": args[2],
        "padding": args[3],
        "dilation": args[4],
        "ceil_mode": False if len(args) < 6 else args[5],
    }

    return acc_ops_converters.acc_ops_max_pool1d(
        network, target, None, kwargs_new, name
    )
