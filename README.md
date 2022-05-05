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

## Support Model

* [age-gender-recognition-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/age-gender-recognition-retail-0013)
* [human-pose-estimation-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
* [facial-landmarks-35-adas-0002](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/facial-landmarks-35-adas-0002)

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
* Subtract
* Clamp
* Transpose
* Elu

## License
Apache 2.0
