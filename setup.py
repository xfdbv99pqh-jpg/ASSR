from setuptools import setup, find_packages

setup(
    name="assr",
    version="0.1.0",
    description="Auto-Calibrated Stochastic Spectral Regularization",
    author="xfdbv99pqh-jpg",
    url="https://github.com/xfdbv99pqh-jpg/ASSR",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
