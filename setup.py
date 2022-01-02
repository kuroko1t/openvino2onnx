from setuptools import setup, find_packages

setup(
    name="vino2onnx",
    scripts=["bin/vino2onnx"],
    packages=find_packages(),
    install_requires=["onnx"],
    version="0.1",
)
