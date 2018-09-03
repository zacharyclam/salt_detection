#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : dirs.py
# @Time     : 2018/9/3 20:35 
# @Software : PyCharm
import os


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
