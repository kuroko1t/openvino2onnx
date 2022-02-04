# Copyright 2022 kurosawa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import xml.etree.ElementTree as ET

from onnx import (
    AttributeProto,
    GraphProto,
    TensorProto,
    checker,
    helper,
    shape_inference,
)

from .ops import operations


def get_inputNode(src_id, edges):
    dst_ids = []
    for edge in edges:
        if src_id == edge["to-layer"]:
            dst_ids.append(edge["from-layer"])
    return dst_ids


def get_edge(root):
    return [edge.attrib for edge in root.iter("edge")]


def get_layer(root):
    layers_info = []
    edges = get_edge(root)
    for layer in root.iter("layer"):
        layer_attr = layer.attrib
        layer_info = {}
        layer_info["id"] = layer_attr["id"]
        layer_info["type"] = layer_attr["type"]
        if layer_attr["type"] == "Parameter":
            input_shape = [
                int(dim) for dim in layer.find("data").attrib["shape"].split(",")
            ]
            layer_info["input_shape"] = input_shape
        elif layer_attr["type"] == "Const":
            data_attr = layer.find("data").attrib
            layer_info["element_type"] = data_attr["element_type"]
            layer_info["offset"] = int(data_attr["offset"])
            layer_info["size"] = int(data_attr["size"])
            if data_attr["shape"] == "":
                layer_info["shape"] = [1]
            else:
                layer_info["shape"] = [int(s) for s in data_attr["shape"].split(",")]
        else:
            layer_info["input_id"] = get_inputNode(layer_info["id"], edges)
        if layer_attr["type"] == "Result":
            output_dims = [
                port.findall("dim") for port in layer.find("input").iter("port")
            ][0]
            layer_info["output_shape"] = [int(dim.text) for dim in output_dims]
        elif layer_attr["type"] == "Multiply":
            input_dims = [
                port.findall("dim")
                for port in layer.find("input").iter("port")
                if port.attrib["id"] == "0"
            ][0]
            output_dims = [
                port.findall("dim") for port in layer.find("output").iter("port")
            ][0]
            input_feature = [int(dim.text) for dim in input_dims][-1]
            output_feature = [int(dim.text) for dim in input_dims][-1]
            layer_info["input_feature"] = input_feature
            layer_info["output_feature"] = output_feature
        elif (
            layer_attr["type"] == "Convolution"
            or layer_attr["type"] == "GroupConvolution"
        ):
            data_attr = layer.find("data").attrib

            # get kernel_size
            dims = [
                port.findall("dim") for port in layer.find("input").findall("port")
            ][1]
            dims = [int(dim.text) for dim in dims]
            kernel_size = [dims[2], dims[3]]
            layer_info["kernel_size"] = kernel_size
            if data_attr["auto_pad"] == "explicit":
                dilations = [int(d) for d in data_attr["dilations"].split(",")]
                pads_begin = [int(d) for d in data_attr["pads_begin"].split(",")]
                pads_end = [int(d) for d in data_attr["pads_end"].split(",")]
                strides = [int(d) for d in data_attr["strides"].split(",")]
                layer_info["dilations"] = dilations
                layer_info["pads_begin"] = pads_begin
                layer_info["pads_end"] = pads_end
                layer_info["strides"] = strides
            else:
                raise Exception("not Expected auto_pad:", data_attr["auto_pad"])
        elif layer_attr["type"] == "MaxPool":
            data_attr = layer.find("data").attrib
            if data_attr["auto_pad"] == "explicit":
                kernel = [int(d) for d in data_attr["kernel"].split(",")]
                pads_begin = [int(d) for d in data_attr["pads_begin"].split(",")]
                pads_end = [int(d) for d in data_attr["pads_end"].split(",")]
                strides = [int(d) for d in data_attr["strides"].split(",")]
                if data_attr["rounding_type"] == "ceil":
                    layer_info["ceil_mode"] = True
                else:
                    layer_info["ceil_mode"] = False
                layer_info["strides"] = strides
                layer_info["kernel"] = kernel
                layer_info["pads_begin"] = pads_begin
                layer_info["pads_end"] = pads_end
            else:
                raise Exception("not Expected auto_pad:", data_attr["auto_pad"])
        elif layer_attr["type"] == "SoftMax" or layer_attr["type"] == "Concat":
            data_attr = layer.find("data").attrib
            layer_info["axis"] = int(data_attr["axis"])
        elif layer_attr["type"] == "Clamp":
            data_attr = layer.find("data").attrib
            layer_info["min"] = data_attr["min"]
            layer_info["max"] = data_attr["max"]
        layers_info.append(layer_info)
    return layers_info


def create_model(model_path, weight_path):
    tree = ET.parse(model_path)
    root = tree.getroot()
    layers = get_layer(root)
    model_name = root.attrib["name"]

    node_def = []
    inputs = []
    outputs = []
    supported_ops = operations(weight_path)
    const_values = {}
    for i, layer in enumerate(layers):
        if layer["type"] == "Parameter":
            print(f"Input Shape: {layer['input_shape']}")
            inputs.append(supported_ops[layer["type"]].make(layer))
        elif layer["type"] == "Result":
            print(f"Output Shape: {layer['output_shape']}")
            outputs.append(supported_ops[layer["type"]].make(layer))
        else:
            if layer["type"] in supported_ops:
                if layer["type"] == "Gather":
                    node = supported_ops[layer["type"]].make(layer, const_values)
                else:
                    node = supported_ops[layer["type"]].make(layer)
                node_def.append(node)
            else:
                raise Exception("not supported layer:", layer)
        if layer["type"] == "Const":
            const_values[layer["id"]] = supported_ops[layer["type"]].value(layer)
    graph_def = helper.make_graph(
        node_def,
        model_name,
        inputs,
        outputs,
    )
    checker.check_graph(graph_def)
    model = helper.make_model(
        graph_def, producer_name="kuroko1t", producer_version="0.1"
    )
    checker.check_model(model)
    out_path = f"{model_name}.onnx"
    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(model)
    # Check the model and print Y's shape information
    checker.check_model(inferred_model)
    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"output model -> {out_path}")
