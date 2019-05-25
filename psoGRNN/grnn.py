#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/23 9:51
__author__ = 'wangwenhui'


from sklearn import preprocessing
from neupy import algorithms
import pandas as pd




def read_csv(file_name):
    df=pd.read_csv("E:\chenqing\mete_data\%s.csv"%file_name,header=None,skiprows=1,index_col=0,engine="python")
    norm_eigen=preprocessing.minmax_scale(df.iloc[:,0:4])
    norm_target=df.iloc[:,5]
    return norm_eigen,norm_target


def train_grnn(g, file_name):
    '''将样本集划分为训练集，验证集和测试集
    Parameters:
    ----------
    g: 待优化的光滑因子
    file_name: 读取的文件名
    return: 返回的是预测值与真实值
    '''
    norm_eigen,norm_target=read_csv(file_name)
    x_train=norm_eigen[:18]
    y_train=norm_target[:18]
    x_test=norm_eigen[18:24]
    y_test=norm_target[18:24]
    gn=algorithms.GRNN(std=g)
    gn.train(x_train,y_train)
    y_predicted=gn.predict(x_test)
    return y_predicted,y_test


def test_grnn(g,file_name):
    '''验证学习后的PSO_GRNN结果
    Parameters:
    ----------
    g: 优化好的光滑因子
    file_name: 读取的文件名
    return: 返回的是预测值
    '''
    norm_eigen,norm_target=read_csv(file_name)
    x_train=norm_eigen[:18]
    y_train=norm_target[:18]
    x_test=norm_eigen[24:]
    y_test=norm_target[24:]
    gn=algorithms.GRNN(std=g,verbose=False)
    gn.train(x_train,y_train)
    y_predicted=gn.predict(x_test)
    return y_predicted,y_test


if __name__ == '__main__':
    pass






