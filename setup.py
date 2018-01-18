#!/usr/bin/env python

import os
import glob


from pip.req import parse_requirements
from setuptools import find_packages, setup, Extension
from optparse import Option
import pip

NAME='neurocover'


pwd = os.path.dirname(__file__)

def parse_reqs(reqs_file):
    ''' parse the requirements '''
    options = Option('--workaround')
    options.skip_requirements_regex = None
    # Hack for old pip versions
    # Versions greater than 1.x have a required parameter "sessions" in
    # parse_requierements
    if pip.__version__.startswith('1.'):
        install_reqs = parse_requirements(reqs_file, options=options)
    else:
        from pip.download import PipSession  # pylint:disable=E0611
        options.isolated_mode = False
        install_reqs = parse_requirements(reqs_file,  # pylint:disable=E1123
                                          options=options,
                                          session=PipSession)

    return [str(ir.req) for ir in install_reqs]

BASEDIR = os.path.dirname(os.path.abspath(__file__))
REQS = parse_reqs(os.path.join(BASEDIR, 'requirements.txt'))

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
      install_requires= REQS,
      packages=find_packages(),
      scripts=['apps/create_covering_animation.py'],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose']
     )

