from setuptools import setup, find_packages

setup(
    name="vino2onnx",
    scripts=["bin/vino2onnx"],
    packages=[
        "onnx",
    ],
    install_requires=get_requires(),
    version="0.1",
)
