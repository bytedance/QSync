import os
import sys

from setuptools import setup, find_packages

name = "qsync"
version = "0.0.1"
description = "System for using mixed-bitwidth handle the heterogenous synchronization"

rootdir = os.path.abspath(os.path.dirname(__file__))
long_description = open(os.path.join(rootdir, "README.md")).read()

setup(
    name=name,
    version=version,
    description=description,

    # Long description of your library
    long_description=long_description,
    long_description_content_type = 'text/markdown',

    url='https://github.com/SpringWave1/qsync_niti_based.git',
    author='Juntao Zhao',
    author_email='juntaozh@connect.hku.hk',
    license='MIT',
    packages=find_packages(),

    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        "License :: OSI Approved :: MIT License",
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    # package_dir={"qsync": "qsync"},
    # python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
)