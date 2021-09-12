
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
  name="fauxgrad",
  version='0.2',
  author='Kiran Sanjeevan Cabeza',
  url='http://github.com/ksanjeevan/fauxgrad',
  author_email="ksanjeevancabeza@gmail.com",
  description="A walkthrough of a small engine for automatic differentiation",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=['fauxgrad'],
  license='MIT',
  install_requires=[
    'micrograd',
    'pytest',
    'matplotlib',
    'numpy',
    'networkx',
    'sklearn>=0.21',
    'ipython>=7.21.0'
    ],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  keywords="autodiff automatic differentiation python",
  zip_safe=False,
  python_requires=">=3.6",
)