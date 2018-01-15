#!/usr/bin/env python

import os
import glob
import numpy


from pip.req import parse_requirements
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize


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


def make_extensions(cwd):

    extensions = []

    file_paths = scandir(cwd)
    for fpath in file_paths:
         dpath = fpath.replace(os.path.sep, '.')[:-4]
         extension = Extension(dpath, [fpath], include_dirs=[numpy.get_include()])
         extensions.append(extension)
    return extensions


setup(
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
      name=NAME,
      version='0.0',
      description='Neruon Cover scripts',
      author='Eleftherios Zisis',
      author_email='eleftherios.zisis@epfl.ch',
      #install_requires=reqs,
      packages=find_packages(),
      include_dirs=[numpy.get_include()],
      ext_modules=cythonize(make_extensions(NAME)),
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose']
     )

