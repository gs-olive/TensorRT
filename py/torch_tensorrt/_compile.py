from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Set, TypeGuard

import torch
import torch.fx
import torch_tensorrt.ts
from torch_tensorrt import logging
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.compile import compile as dynamo_compile
from torch_tensorrt.fx import InputTensorSpec
from torch_tensorrt.fx.lower import compile as fx_compile
from torch_tensorrt.fx.utils import LowerPrecision
from torch_tensorrt.ts._compiler import compile as torchscript_compile


def _non_fx_input_interface(
    inputs: Sequence[Input | torch.Tensor | InputTensorSpec],
) -> TypeGuard[List[Input | torch.Tensor]]:
    return all(isinstance(i, torch.Tensor | Input) for i in inputs)


def _fx_input_interface(
    inputs: Sequence[Input | torch.Tensor | InputTensorSpec],
) -> TypeGuard[List[InputTensorSpec | torch.Tensor]]:
    return all(isinstance(i, torch.Tensor | InputTensorSpec) for i in inputs)


class _IRType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout"""

    ts = 0
    fx = 1
    dynamo = 2
    torch_compile = 3


class _ModuleType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout"""

    nn = 0
    ts = 1
    fx = 2


def _parse_module_type(module: Any) -> _ModuleType:
    if any(
        isinstance(module, t)
        for t in [torch.jit.ScriptModule, torch.jit.ScriptFunction]
    ):
        return _ModuleType.ts
    elif isinstance(module, torch.fx.GraphModule):
        return _ModuleType.fx
    elif isinstance(module, torch.nn.Module):
        return _ModuleType.nn
    else:
        raise RuntimeError("Module is an unknown format")


def _get_target_ir(module_type: _ModuleType, ir: str) -> _IRType:
    module_is_tsable = any(module_type == t for t in [_ModuleType.nn, _ModuleType.ts])
    module_is_fxable = any(module_type == t for t in [_ModuleType.nn, _ModuleType.fx])

    ir_targets_torchscript = any(ir == opt for opt in ["torchscript", "ts"])
    ir_targets_fx = ir == "fx"
    ir_targets_dynamo = ir == "dynamo"
    ir_targets_torch_compile = ir == "torch_compile"

    if module_is_tsable and ir_targets_torchscript:
        return _IRType.ts
    elif module_is_fxable and ir_targets_fx:
        return _IRType.fx
    elif module_is_fxable and ir_targets_dynamo:
        return _IRType.dynamo
    elif module_is_fxable and ir_targets_torch_compile:
        return _IRType.torch_compile
    else:
        if ir == "default":
            # Options are listed in order of preference
            if module_is_fxable:
                logging.log(
                    logging.Level.Info, "ir was set to default, using dynamo as ir"
                )
                return _IRType.dynamo
            elif module_is_tsable:
                logging.log(
                    logging.Level.Warning,
                    "Input graph is a Torchscript module but the ir provided is default (dynamo). Please set ir=torchscript to suppress the warning. Compiling the module with ir=torchscript",
                )
                return _IRType.ts
            else:
                raise ValueError("Module was provided in an unsupported format")
        else:
            raise ValueError("Unknown ir was requested")


def compile(
    module: Any,
    ir: str = "default",
    inputs: Optional[Sequence[Input | torch.Tensor | InputTensorSpec]] = None,
    enabled_precisions: Optional[Set[torch.dtype | dtype]] = None,
    **kwargs: Any,
) -> (
    torch.nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule | Callable[..., Any]
):
    """Compile a PyTorch module for NVIDIA GPUs using TensorRT

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """
    input_list = inputs if inputs is not None else []
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else {torch.float}
    )

    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript",
            )
            ts_mod = torch.jit.script(module)
        assert _non_fx_input_interface(input_list)
        compiled_ts_module: torch.jit.ScriptModule = torchscript_compile(
            ts_mod,
            inputs=input_list,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
        return compiled_ts_module
    elif target_ir == _IRType.fx:
        if (
            torch.float16 in enabled_precisions_set
            or torch_tensorrt.dtype.half in enabled_precisions_set
        ):
            lower_precision = LowerPrecision.FP16
        elif (
            torch.float32 in enabled_precisions_set
            or torch_tensorrt.dtype.float in enabled_precisions_set
        ):
            lower_precision = LowerPrecision.FP32
        else:
            raise ValueError(f"Precision {enabled_precisions_set} not supported on FX")

        assert _fx_input_interface(input_list)
        compiled_fx_module: torch.nn.Module = fx_compile(
            module,
            input_list,
            lower_precision=lower_precision,
            explicit_batch_dimension=True,
            dynamic_batch=False,
            **kwargs,
        )
        return compiled_fx_module
    elif target_ir == _IRType.dynamo:
        import collections.abc

        from torch_tensorrt import Device
        from torch_tensorrt.dynamo.utils import prepare_inputs, to_torch_device

        if not isinstance(inputs, collections.abc.Sequence):
            inputs = [inputs]
        device = kwargs.get("device", Device._current_device())
        torchtrt_inputs, torch_inputs = prepare_inputs(inputs, to_torch_device(device))
        module = torch_tensorrt.dynamo.trace(module, torch_inputs, **kwargs)
        compiled_aten_module: torch.fx.GraphModule = dynamo_compile(
            module,
            inputs=input_list,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
        return compiled_aten_module
    elif target_ir == _IRType.torch_compile:
        return torch_compile(
            module, enabled_precisions=enabled_precisions_set, **kwargs
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def torch_compile(module: torch.nn.Module, **kwargs: Any) -> Any:
    """
    Returns a boxed model which is the output of torch.compile.
    This does not compile the model to TRT. Execute this model on
    sample inputs to compile the model to TRT.
    """
    from torch_tensorrt.dynamo.backend import torch_tensorrt_backend

    boxed_fn = torch.compile(module, backend=torch_tensorrt_backend, options={**kwargs})

    return boxed_fn


def convert_method_to_trt_engine(
    module: Any,
    method_name: str = "forward",
    inputs: Optional[Sequence[Input | torch.Tensor]] = None,
    ir: str = "default",
    enabled_precisions: Optional[Set[torch.dtype | dtype]] = None,
    **kwargs: Any,
) -> bytes:
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else {torch.float}
    )

    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript",
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.convert_method_to_trt_engine(  # type: ignore[no-any-return]
            ts_mod,
            inputs=inputs,
            method_name=method_name,
            enabled_precisions=enabled_precisions_set,
            **kwargs,
        )
    elif target_ir == _IRType.fx:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=fx"
        )
    elif target_ir == _IRType.dynamo:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=dynamo."
        )
    elif target_ir == _IRType.torch_compile:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=torch_compile"
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")
