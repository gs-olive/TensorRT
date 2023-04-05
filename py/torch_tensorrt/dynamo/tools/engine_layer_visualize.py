import argparse
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pydot

_LOGGER: logging.Logger = logging.getLogger(__name__)
"""
log_file is generated by tensorrt verbose logger during building engine.
profile_file is generated by tensorrt profiler.

Curretnly we support processing multiple logs in one log_file, which
would generate multiple dot graphs. However, multiple engine profiles are not
supported.

Usage:
    python torch_tensorrt.fx/tools/engine_layer_visualize.py --log_file aaa --profile_file bbb
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_file",
    type=str,
    default="",
    help="TensorRT VERBOSE logging when building engines.",
)
parser.add_argument(
    "--profile_file",
    type=str,
    default="",
    help="TensorRT execution context profiler output.",
)
args = parser.parse_args()


class LayerInfo(NamedTuple):
    kernel_name: str
    layer_name: str
    tactic: str
    input_names: Optional[List[str]]
    input_types: Optional[List[str]]
    output_name: str
    output_type: str
    time: str

    @classmethod
    def from_string(cls, string, tactic_names, layer_times=None):
        input_names = []
        input_types = []
        kernel_name, layer_name, tactic, inputs, output_name, output_type = re.findall(
            "Layer\\((.+)\\): (.+), Tactic: (-?\\d+), (.+)? -> (.+)\\[(.+)\\]", string
        )[0]

        if kernel_name != "Constant":
            inputs = re.findall(
                "[, ]*(.+?)\\[([Half|Float|Int8]+\\(\\d[,\\d]*\\))\\]", inputs
            )
            for input_name, input_type in inputs:
                input_names.append(input_name)
                input_types.append(input_type)

            if layer_name in tactic_names:
                kernel_name = tactic_names[layer_name]
        else:
            input_names = input_types = None  # type:ignore[assignment]

        return cls(
            kernel_name,
            layer_name,
            tactic,
            input_names,
            input_types,
            output_name,
            output_type,
            layer_times[layer_name] if layer_times else "NA",
        )


def build_node(layer):
    layer_name = layer.layer_name.replace("|", "\\|")
    label = f"{{{layer_name}|kernel: {layer.kernel_name}\\l|tactic: {layer.tactic}\\l|time: {layer.time}\\l}}"
    label = label.replace(">", "\\>")
    return pydot.Node(layer.layer_name, label=label, **style)


def build_edge(layer, graph, reformat_layers, output_name2node, layer_name2node):
    if layer.input_names is None:
        return

    for input_name, input_type in zip(layer.input_names, layer.input_types):
        if input_name not in output_name2node:
            if input_name in reformat_layers:
                from_node = pydot.Node(
                    input_name,
                    label="{reformatter|kernel: Reformat\\l|tactic: 0\\l}",
                    **style,
                )
                graph.add_node(from_node)
                if reformat_layers[input_name][0] in output_name2node:
                    graph.add_edge(
                        pydot.Edge(
                            output_name2node[reformat_layers[input_name][0]],
                            from_node,
                            label=f"{reformat_layers[input_name][0]}\\l{reformat_layers[input_name][1]}\\l",
                        )
                    )
            else:
                _LOGGER.info(f"Missing node {input_name}")
                from_node = input_name
        else:
            from_node = output_name2node[input_name]

        edge_name = input_name.replace(">", "\\>")
        graph.add_edge(
            pydot.Edge(
                from_node,
                layer_name2node[layer.layer_name],
                label=f"{edge_name}\\l{input_type}\\l",
            )
        )


if args.profile_file != "":
    layer_times = {}
    with open(args.profile_file) as f:
        times = f.readlines()

    for t in times:
        t = t.strip("\n").split(": ")  # type: ignore[assignment]
        layer_times[": ".join(t[:-1])] = t[-1]
else:
    layer_times = None  # type: ignore[assignment]

if args.log_file != "":
    with open(args.log_file) as f:
        lines = f.readlines()

    graphs = []
    layers = []
    reformat_layers: Dict[str, Tuple[str, str]] = {}
    tactic_names: Dict[str, str] = {}
    layer_info_start = False
    tactic_name_start = False

    for line in lines:
        line = line.strip("\n")

        if layer_info_start:
            if "Layer(" in line:
                layers.append(LayerInfo.from_string(line, tactic_names, layer_times))
            else:
                layer_info_start = False
                graphs.append((layers, reformat_layers))
                layers = []
                reformat_layers = {}
                tactic_names = {}

        if tactic_name_start and "Set Tactic Name:" in line:
            layer_name, kernel_name, _ = re.findall(
                "VERBOSE: (.*) Set Tactic Name: (.*) Tactic: (.*)$", line
            )[0]
            tactic_names[layer_name] = kernel_name

        # Some reformat layers aren't displayed in Engine Layer Information
        if "Adding reformat layer" in line:
            output_name, input_name, from_type, to_type = re.findall(
                "reformat layer: (.+) \\((.+)\\) from (.+) to (.+)", line
            )[0]
            reformat_layers[output_name] = (input_name, from_type)

        if "Total Activation Memory:" in line:
            tactic_name_start = True

        if "Engine Layer Information" in line:
            layer_info_start = True
            tactic_name_start = False

    style = {
        "shape": "record",
        "fillcolor": "Salmon",
        "style": '"filled,rounded"',
        "fontcolor": "#000000",
    }

    dot_graphs: List[Any] = []
    i = 0
    for layers, reformat_layers in graphs:
        output_name2node = {}
        layer_name2node = {}
        dot_graph = pydot.Dot("Layer Graph")

        for layer in layers:
            node = build_node(layer)
            dot_graph.add_node(node)
            output_name2node[layer.output_name] = node
            layer_name2node[layer.layer_name] = node

        for layer in layers:
            build_edge(
                layer, dot_graph, reformat_layers, output_name2node, layer_name2node
            )

        dot_graph.write_raw(f"EngineLayers_{i}.dot")
        i += 1

if args.profile_file != "":
    est_reformat_time = 0.0
    est_total_time = 0.0

    for layer in layers:
        if layer.kernel_name == "Reformat":
            est_reformat_time += float(layer.time[:-2])
        est_total_time += float(layer.time[:-2])

    _LOGGER.info(f"Time Cost on Reformatting: {est_reformat_time} ms")
    _LOGGER.info(f"Total Time Cost: {est_total_time} ms")
