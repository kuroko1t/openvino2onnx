import os
import unittest
import urllib.request

import vino2onnx


def download(model_url, weight_url):
    urllib.request.urlretrieve(model_url, os.path.basename(model_url))
    urllib.request.urlretrieve(weight_url, os.path.basename(weight_url))
    model_path = os.path.splitext(os.path.basename(model_url))[0]
    return model_path


model_urls = [
    [
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin",
    ],
    [
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin",
    ],
    [
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml",
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.bin",
    ],
]


class ConvertTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ConvertTests, self).__init__(*args, **kwargs)

    def test_convert(self):
        for url_pair in model_urls:
            model_path = download(*url_pair)
            vino2onnx.create_model(model_path + ".xml", model_path + ".bin")
