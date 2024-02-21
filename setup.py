import os
import subprocess

from setuptools import setup, find_packages
from typing import List

name = "qsync"
version = "0.0.1"
description = "System for using mixed-bitwidth handle the heterogenous synchronization"

ROOT_DIR = os.path.dirname(__file__)
long_description = open(os.path.join(ROOT_DIR, "README.md")).read()


def make_kernels():
    # Run make command
    print("Making the kernels, make take a bit long, if you want to check the progress, please directly run make in the root")
    print("--- Making ---")
    os.system("make")
    print("--- Done ---")

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

make_kernels()

setup(
    name=name,
    version=version,
    description=description,

    # Long description of your library
    long_description=long_description,
    long_description_content_type = 'text/markdown',

    url='https://github.com/bytedance/QSync.git',
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
    install_requires=get_requirements(),
    # package_dir={"qsync": "qsync"},
    # python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
)