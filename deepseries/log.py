# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/21 13:41
"""
import logging
from datetime import datetime


def get_logger(name):
    # exist_logger = logging.Logger.manager.loggerDict
    # if 'deepseries' not in exist_logger:
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    logging.basicConfig(
        # filename=os.path.join(self.log_dir, log_file),
        level=logging.INFO,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    # logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(name)
    # if len(logger.handlers) == 0:
    #     logger.addHandler(logging.StreamHandler())
    return logger
