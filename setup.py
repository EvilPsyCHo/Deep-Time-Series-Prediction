# coding: utf-8
# !/usr/bin/env python
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2019/11/9 下午12:20
"""

from setuptools import setup


setup(
    name='_dtsp',
    version='0.1',
    description='Time Series Prediction Library',
    author='zhirui zhou',
    author_email='evilpsycho42@gmail.com',
    url='https://github.com/EvilPsyCHo/competition',
    packages={'_dtsp', '_dtsp.dataset', '_dtsp.models'},
)
