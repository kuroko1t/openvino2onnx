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

import struct

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, TensorProto, helper


class Convolution:
    def make(self, layer_info):
        node = onnx.helper.make_node(
            "Conv",
            inputs=layer_info["input_id"],
            outputs=[layer_info["id"]],
            pads=layer_info["pads_begin"] + layer_info["pads_end"],
            strides=layer_info["strides"],
            kernel_shape=layer_info["kernel_size"],
        )
        return node


class Input:
    def make(self, layer_info):
        X = helper.make_tensor_value_info(
            str(layer_info["id"]), TensorProto.FLOAT, layer_info["input_shape"]
        )
        return X


class Result:
    def make(self, layer_info):
        assert (
            len(layer_info["input_id"]) == 1
        ), f"input_num is not 1: {len(layer_info['input_id'])}"
        Y = helper.make_tensor_value_info(
            layer_info["input_id"][0], TensorProto.FLOAT, layer_info["output_shape"]
        )
        return Y


class Const:
    def __init__(self, weight_path):
        self.datatype_cfg = {
            "f32": ["f", 4],
            "f16": ["e", 2],
        }
        with open(weight_path, "rb") as f:
            self.weight = f.read()

    def read(self, layer_info):
        weight = self.weight[
            layer_info["offset"] : layer_info["offset"] + layer_info["size"]
        ]
        data_type = layer_info["element_type"]
        formatstring = "<" + self.datatype_cfg[data_type][0] * (
            len(weight) // self.datatype_cfg[data_type][1]
        )
        np_weight = np.array(list(struct.unpack(formatstring, weight))).flatten()
        if np_weight.dtype == np.float64:
            np_weight = np_weight.astype(np.float32)
        return np_weight

    def make(self, layer_info):
        np_weight = self.read(layer_info)
        node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[str(layer_info["id"])],
            value=onnx.helper.make_tensor(
                name=str(layer_info["id"]),
                data_type=onnx.TensorProto.FLOAT,
                dims=layer_info["shape"],
                vals=np_weight.flatten(),
            ),
        )
        return node


class Add:
    def make(self, layer_info):
        node = onnx.helper.make_node(
            "Add",
            inputs=layer_info["input_id"],
            outputs=[str(layer_info["id"])],
        )
        return node


class MaxPool:
    def make(self, layer_info):
        node = onnx.helper.make_node(
            "MaxPool",
            inputs=layer_info["input_id"],
            outputs=[str(layer_info["id"])],
            kernel_shape=layer_info["kernel"],
            pads=layer_info["pads_begin"] + layer_info["pads_end"],
            ceil_mode=layer_info["ceil_mode"],
            strides=layer_info["strides"],
        )
        return node


class Relu:
    def make(self, layer_info):
        node = onnx.helper.make_node(
            "Relu",
            inputs=layer_info["input_id"],
            outputs=[str(layer_info["id"])],
        )
        return node


class Softmax:
    def make(self, layer_info):
        node = onnx.helper.make_node(
            "Softmax",
            inputs=layer_info["input_id"],
            axis=layer_info["axis"],
            outputs=[str(layer_info["id"])],
        )
        return node
