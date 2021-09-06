
from setuptools import setup

setup(
    name="fauxgrad",
    version='0.1',
    author='Kiran Sanjeevan',
    url='http://github.com/ksanjeevan/fauxgrad',
    author_email="ksanjeevancabeza@gmail.com",
    description="A crappy automatic differentiation lilbrary",
    packages=['fauxgrad'],
    license='MIT',
    install_requires=[
        'micrograd',
        'networkx',
        'pytest'
      ],
    keywords="automatic differentiation python",
    zip_safe=False
)