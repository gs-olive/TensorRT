import logging
import operator
from functools import partial
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as td
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch_tensorrt.dynamo import CompilationSettings, partitioning
from torch_tensorrt.dynamo.conversion import (
    convert_module,
    repair_long_or_double_inputs,
)
from torch_tensorrt.dynamo.conversion.converter_registry import ConverterRegistry
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
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

    import unittest

    from torch._dynamo.utils import detect_fake_mode

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

        orig, replacer = efficient_attention_replacement()

        torch.fx.subgraph_rewriter.replace_pattern(graph_module, orig, replacer)

        constant_fold(graph_module)

        return _pretraced_backend(graph_module, sample_inputs, settings)


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


@torch.utils._python_dispatch._disable_current_modes()  # type: ignore
def constant_fold(gm: torch.fx.GraphModule) -> None:
    from torch._inductor.freezing import ConstantFolder, replace_node_with_constant

    cf = ConstantFolder(gm, skip_constructors=False)
    cf.run()

    for node, constant in cf.node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def lower_efficient_attention(graph_module: torch.fx.GraphModule) -> None:
    eff_attn = ConverterRegistry.qualified_name_or_str(
        torch.ops.aten._scaled_dot_product_efficient_attention.default
    )

    import math

    for node in graph_module.graph.nodes:
        if (
            ConverterRegistry.qualified_name_or_str(node.target) == eff_attn
            and len(node.users) == 1
            and ConverterRegistry.qualified_name_or_str(list(node.users)[0].target)
            == "_operator.getitem"
            and list(node.users)[0].args[1] == 0
        ):
            # Replacement
            attention = node
            getitem = list(node.users)[0]

            q, k, v = node.args[:3]
            with graph_module.graph.inserting_before(node.next):
                transpose = graph_module.graph.call_function(
                    torch.ops.aten.transpose.int,
                    args=(k, -2, -1),
                )
                q_flat = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(q, [-1, *q.meta["val"].size()[-2:]]),
                )
                transpose_flat = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(q, [-1, *transpose.meta["val"].size()[-2:]]),
                )
                bmm = graph_module.graph.call_function(
                    torch.ops.aten.bmm.default,
                    args=(q_flat, transpose_flat),
                )
                mm = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(
                        bmm,
                        [*q.meta["val"].size()[:-2], *bmm.meta["val"].size()[-2:]],
                    ),
                )
                div = graph_module.graph.call_function(
                    torch.ops.aten.div.Scalar,
                    args=(mm, math.sqrt(q.meta["val"].size()[-1])),
                )
                softmax = graph_module.graph.call_function(
                    torch.ops.aten._softmax.default,
                    args=(div, -1, False),
                )
                softmax_flat = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(softmax, [-1, *softmax.meta["val"].size()[-2:]]),
                )
                v_flat = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(v, [-1, *v.meta["val"].size()[-2:]]),
                )
                out_flat = graph_module.graph.call_function(
                    torch.ops.aten.bmm.default,
                    args=(softmax_flat, v_flat),
                )
                out = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(
                        out_flat,
                        [*v.meta["val"].size()[:-2], *out_flat.meta["val"].size()[-2:]],
                    ),
                )

            getitem.replace_all_uses_with(out)
            graph_module.graph.erase_node(getitem)
            graph_module.graph.erase_node(attention)

    graph_module.graph.lint()
    graph_module.recompile()


def efficient_attention_replacement():
    def boilerplate(query, key, value):
        ...

    orig = torch.fx.symbolic_trace(boilerplate)
    placeholders = [node for node in orig.graph.nodes if node.op == "placeholder"]
    q, k, v = placeholders

    output = [node for node in orig.graph.nodes if node.op == "output"][0]

    with orig.graph.inserting_before(output):
        att = orig.graph.call_function(
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            args=(q, k, v, None, False),
        )
        out = orig.graph.call_function(
            operator.getitem,
            args=(att, 0),
        )

    output.args = (out,)

    orig.graph.lint()
    orig.recompile()

    def replacement(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    return orig, replacement


# def lower_efficient_attention(graph_module: torch.fx.GraphModule) -> None:
#     eff_attn = ConverterRegistry.qualified_name_or_str(torch.ops.aten._scaled_dot_product_efficient_attention.default)

#     import math

#     for node in graph_module.graph.nodes:
#         if (ConverterRegistry.qualified_name_or_str(node.target) == eff_attn
#             and len(node.users) == 1
#             and ConverterRegistry.qualified_name_or_str(list(node.users)[0].target) == "_operator.getitem"
#             and list(node.users)[0].args[1] == 0):
#             # Replacement
#             attention = node
#             getitem = list(node.users)[0]

#             q, k, v = node.args[:3]
#             with graph_module.graph.inserting_before(node.next):
#                 transpose = graph_module.graph.call_function(
#                     torch.ops.aten.transpose.int,
#                     args=(k, -2, -1),
#                 )
#                 q_flat = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(q, [-1, *q.meta["val"].size()[-2:]]),
#                 )
#                 transpose_flat = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(q, [-1, *transpose.meta["val"].size()[-2:]]),
#                 )
#                 bmm = graph_module.graph.call_function(
#                     torch.ops.aten.bmm.default,
#                     args=(q_flat, transpose_flat),
#                 )
#                 mm = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(bmm, [*q.meta["val"].size()[:-2], *bmm.meta["val"].size()[-2:]]),
#                 )
#                 div = graph_module.graph.call_function(
#                     torch.ops.aten.div.Scalar,
#                     args=(mm, math.sqrt(q.meta["val"].size()[-1])),
#                 )
#                 softmax = graph_module.graph.call_function(
#                     torch.ops.aten._softmax.default,
#                     args=(div, -1, False),
#                 )
#                 softmax_flat = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(softmax, [-1, *softmax.meta["val"].size()[-2:]]),
#                 )
#                 v_flat = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(v, [-1, *v.meta["val"].size()[-2:]]),
#                 )
#                 out_flat = graph_module.graph.call_function(
#                     torch.ops.aten.bmm.default,
#                     args=(softmax_flat, v_flat),
#                 )
#                 out = graph_module.graph.call_function(
#                     torch.ops.aten.view.default,
#                     args=(out_flat, [*v.meta["val"].size()[:-2], *out_flat.meta["val"].size()[-2:]]),
#                 )

#             getitem.replace_all_uses_with(out)
#             graph_module.graph.erase_node(getitem)
#             graph_module.graph.erase_node(attention)

#     graph_module.graph.lint()
#     graph_module.recompile()

# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
#     # Efficient implementation equivalent to the following:
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     out = attn_weight @ value
#     return out
