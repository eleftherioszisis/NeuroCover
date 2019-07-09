#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
      name='neurocover',
      version='0.0',
      description='Neruon Cover scripts',
      author='Eleftherios Zisis',
      author_email='eleftherios.zisis@epfl.ch',
      install_requires= ['numpy', 'scipy', 'neurom'],
      packages=find_packages(),
      scripts=['apps/create_covering_animation.py'],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose']
     )

