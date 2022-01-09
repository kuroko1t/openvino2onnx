# vino2onnx
A tool for converting openvino models to onnx format.

## Install

```
python -m pip install -e .
```

## Example

```bash
$ vino2onnx age-gender-recognition-retail-0013
Input Shape: [1, 3, 62, 62]
Output Shape: ['1', '2', '1', '1']
Output Shape: ['1', '1', '1', '1']
output model -> age_gender.onnx
```

## Support OpenVINO OP

* Convolution
* Add
* MaxPool
* ReLU
* SoftMax
* Multiply
* Concat
* Unsqueeze
* Reshape

## License
Apache 2.0
