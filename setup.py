# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 09:26
# @Author  : Jason
# @FileName: setup.py

from setuptools import setup
from setuptools import find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if os.path.exists("requirements.txt"):
    install_requires = open("requirements.txt").read().split("\n")
else:
    install_requires = []

setup(name="gnnex",
      version="1.0.1",
      description="draw different colors to show edges' importance",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Zhang Pin",
      author_email="zhangpin@geetest.com",
      install_requires=install_requires,
      url="https://github.com/ifkid/gnnex",
      download_url="https://pypi.org/manage/project/gnnexplainer/releases/",
      license="MIT LICENSE",
      classifiers=(
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ),
      packages=find_packages())
