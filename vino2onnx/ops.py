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


from .operation import *


def operations(weight_path):
    # Support Operations
    inp = Input()
    result = Result()
    const = Const(weight_path)
    conv = Convolution()
    add = Add()
    maxpool = MaxPool()
    relu = Relu()
    softmax = Softmax()
    multiply = Multiply()
    concat = Concat()
    shapeof = ShapeOf()
    gather = Gather()
    unsqueeze = Unsqueeze()
    reshape = Reshape()
    subtract = Subtract()
    clamp = Clamp()

    operations = {
        "Parameter": inp,
        "Result": result,
        "Const": const,
        "Convolution": conv,
        "Add": add,
        "MaxPool": maxpool,
        "ReLU": relu,
        "SoftMax": softmax,
        "Multiply": multiply,
        "Concat": concat,
        "ShapeOf": shapeof,
        "Gather": gather,
        "Unsqueeze": unsqueeze,
        "Reshape": reshape,
        "Subtract": subtract,
        "Clamp": clamp,
    }
    return operations
