#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 7:37
__author__ = 'wangwenhui'

import math
import numpy as np
from psoGrnn.grnn import train_grnn


def fitness_rmse(g,file_name):
    '''适应度函数选为均方根误差'''
    y_predicted, y_test=train_grnn(g, file_name)
    y_test=np.array(y_test)
    rmse=math.sqrt(np.mean(y_predicted-y_test)**2)
    return rmse


if __name__ == '__main__':
    rmse=fitness_rmse(30,"pipe_corrosion")
    print(rmse)