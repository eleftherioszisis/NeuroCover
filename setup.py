#!/usr/bin/env python

import os
import glob


from pip.req import parse_requirements
from setuptools import find_packages, setup, Extension


NAME='neurocover'


pwd = os.path.dirname(__file__)


def scandir(directory, files=[]):

    for f in os.listdir(directory):

        f_path = os.path.join(directory, f)

        if os.path.isfile(f_path) and f_path.endswith(".pyx"):
            files.append(f_path)
        elif os.path.isdir(f_path):
            scandir(f_path, files)

    return files


setup(
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
      name=NAME,
      version='0.0',
      description='Neruon Cover scripts',
      author='Eleftherios Zisis',
      author_email='eleftherios.zisis@epfl.ch',
      packages=find_packages(),
      scripts=['apps/create_covering_animation.py'],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose']
     )

