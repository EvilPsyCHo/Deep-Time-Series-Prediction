# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/25 13:56
"""
import logging


def get_logger(name):
    logger = logging.Logger(name, level=logging.INFO)
    string = logging.StreamHandler()
    logger.addHandler(string)
    return logger
