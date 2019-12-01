# coding: utf-8
# !/usr/bin/env python
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/9 下午12:20
"""

from setuptools import setup


setup(
    name='dtsp',
    version='0.3',
    description='DeepLearning for Time Series Prediction Library',
    author='zhirui zhou',
    author_email='evilpsycho42@gmail.com',
    license="Apache 2.0",
    url='https://github.com/EvilPsyCHo/competition',
    packages={'dtsp', 'dtsp.dataset', 'dtsp.models', 'dtsp.modules'},
    package_data={'dtsp': ['dataset/*.csv']},
)
