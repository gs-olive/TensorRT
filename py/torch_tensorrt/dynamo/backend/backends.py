import logging
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch_tensorrt.dynamo import CompilationSettings
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.utils import parse_dynamo_kwargs

logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")  # type: ignore[misc]
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    compiled_mod: torch.nn.Module = DEFAULT_BACKEND(gm, sample_inputs, **kwargs)
    return compiled_mod


@td.register_backend(name="aot_torch_tensorrt_aten")  # type: ignore[misc]
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs: Any
) -> torch.nn.Module:
    settings = parse_dynamo_kwargs(kwargs)

    custom_backend = partial(
        _pretraced_backend,
        settings=settings,
    )

    # Perform Pre-AOT Lowering for Module-Level Replacement
    # gm = pre_aot_substitutions(gm)

    from torch._dynamo.utils import detect_fake_mode
    from torch._inductor.freezing import constant_fold
    import unittest

    fake_mode = detect_fake_mode(sample_inputs)

    # Place backend tracing within FakeTensor context allowing nonfake Tensors
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:

        # Invoke AOTAutograd to translate operators to aten
        graph_module = aot_export_joint_simple(
            gm,
            sample_inputs,
            trace_joint=False,
            decompositions=get_decompositions(),
        )
        constant_fold(graph_module)

        return _pretraced_backend(graph_module, sample_inputs, settings)

    # # Invoke AOTAutograd to translate operators to aten
    # return aot_module_simplified(
    #     gm,
    #     sample_inputs,
    #     fw_compiler=make_boxed_compiler(custom_backend),
    #     decompositions=get_decompositions(),
    # )


def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule | Callable[..., Any]:
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        logger.debug("Post-AOT Autograd graph:\n" + str(gm.graph))

        trt_compiled = _compile_module(
            gm,
            sample_inputs,
            settings=settings,
        )
        return trt_compiled
    except AssertionError:
        if not settings.pass_through_build_failures:
            logger.warning(
                "TRT conversion failed on the subgraph. See trace above. "
                + "Returning GraphModule forward instead.",
                exc_info=True,
            )
            return gm.forward
        else:
            logger.critical(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )
            raise


def _compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        gm, settings.debug, settings.torch_executed_ops
    )

    # If the number of supported operations is 0 or less than the block size, skip the subgraph
    # TODO: Add condition to second expression below when require_full_compilation is added
    if num_supported_ops == 0 or (num_supported_ops < settings.min_block_size):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    # Partition module into components that can be TRT-accelerated
    fast_partitioner_failed = False

    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            partitioned_module = partitioning.fast_partition(
                gm,
                verbose=settings.debug,
                min_block_size=settings.min_block_size,
                torch_executed_ops=settings.torch_executed_ops,
            )
        except torch.fx.passes.splitter_base.FxNetSplitterInternalError:
            logger.error(
                "Partitioning failed on the subgraph with fast partition. See trace above. "
                + "Retrying with global partition.",
                exc_info=True,
            )

            fast_partitioner_failed = True
            settings.use_fast_partitioner = False

    if not settings.use_fast_partitioner:
        partitioned_module = partitioning.global_partition(
            gm,
            verbose=settings.debug,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
        )

    # Store TRT replicas of Torch subgraphs
    trt_modules = {}

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            continue

        submodule = getattr(partitioned_module, name)

        # Get submodule inputs
        submodule_inputs = partitioning.get_submod_inputs(
            partitioned_module, submodule, sample_inputs
        )

        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            submodule_inputs = repair_long_or_double_inputs(
                partitioned_module, submodule, submodule_inputs, name
            )

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
            name=name,
        )

        trt_modules[name] = trt_mod

    # Replace all FX Modules with TRT Modules
    for name, trt_mod in trt_modules.items():
        setattr(partitioned_module, name, trt_mod)

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    return partitioned_module
