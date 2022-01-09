import unittest
import vino2onnx
import urllib.request
import os


class AgeGenderTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AgeGenderTests, self).__init__(*args, **kwargs)
        fp32model_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml"
        fp32weight_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin"

        urllib.request.urlretrieve(fp32model_url, os.path.basename(fp32model_url))
        urllib.request.urlretrieve(fp32weight_url, os.path.basename(fp32weight_url))
        self.model_path = os.path.splitext(os.path.basename(fp32model_url))[0]

    def test_convert(self):
        vino2onnx.create_model(self.model_path + ".xml", self.model_path + ".bin")
