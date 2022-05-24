from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup

setup(
    name='lift',
    version='0.1',
    description='Deep reinforcement learning from demonstration.',
    url='',
    author='Michael Schaarschmidt',
    author_email='mks40@cam.ac.uk',
    license='No use allowed',
    packages=['lift'],
    install_requires=[
      'numpy',
      'pymongo',
      'six',
      'rlgraph',
      'python-gflags',
      'PyYAML'
    ],
    zip_safe=False
)