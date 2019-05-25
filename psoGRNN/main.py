#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/23 10:08
__author__ = 'wangwenhui'


import pandas as pd
import os
from matplotlib import pyplot as plt
from psoGrnn.pso import PSO
from psoGrnn.grnn import test_grnn
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams.update({'font.size': 12})

class Main:
    def __init__(self,file_name,maxgen,sizepop,record,zbest):
        '''
        :param file_name: 读取的文件名
        :param record: 记录的每次迭代的适应度函数（RMSE）值
        :param zbest: 光滑因子的最优值
        '''
        self.file_name=file_name
        self.maxgen = maxgen
        self.sizepop = sizepop
        self.record=record
        self.zbest=zbest

    def save_result(self):
        '''存储每次运算结果的最大迭代次数，种群规模，最优适应度函数值和最优光滑因子'''
        best_df=pd.DataFrame()
        best_df['file_name']=["%s.csv"%self.file_name]
        best_df['maxgen']=[self.maxgen]
        best_df['sizepop']=[self.sizepop]
        best_df['rmse']=[self.record[-1]]
        best_df['zbest']=self.zbest
        if os.path.exists("result/param.csv"):
            best_df.to_csv("result/param.csv",mode='a+',header=0,index=0)
        else:
            best_df.to_csv("result/param.csv",index=0)
        print("文件生成成功!")
        return None

    def draw_rmse(self):
        '''绘制目标函数（RMSE)随迭代次数的变化情况曲线图'''
        x=range(len(self.record))
        y=self.record
        plt.title("Iterative graph of PSO_GRNN")
        plt.plot(x,y,color='blue',label='best_value',lw=1.0)
        plt.xlabel('iteration number')
        plt.ylabel("RMSE value")
        plt.tick_params(direction='in')
        plt.legend(edgecolor='black',loc='best')
        figure_fig=plt.gcf()
        plt.rcParams['savefig.dpi']=1000
        figure_fig.savefig('result/%s_rmse.png'%self.file_name,format='png')
        print("图片保存成功！")
        return None

def draw_value():
    '''绘制预测值与真实值之间的对比图'''
    params_df=pd.read_csv("result/param.csv")
    g_index=params_df['rmse'].values.argmin()
    file_name=params_df.loc[g_index,"file_name"]
    g=params_df.loc[g_index,"zbest"]
    y_predicted, y_test=test_grnn(0.15, file_name[:-4])
    plt.title("Compare View")
    x=range(24,30)
    plt.plot(x,y_test,'-o',color='red',label='real value',lw=1.0)
    plt.plot(x,y_predicted,'-*',color='blue',label='predicted value',lw=1.0)
    plt.xlabel('pipeline_number')
    plt.ylabel('Average corrosion rate')
    plt.tick_params(direction='in')
    plt.legend(edgecolor='black', loc='best')
    figure_fig=plt.gcf()
    plt.rcParams['savefig.dpi']=1000
    figure_fig.savefig('result/%s_compare.png'%file_name,format='png')
    print("图片保存成功！")
    return None


def main():
    file_name="pipe_corrosion"
    wmax = 0.8
    wmin=0.2
    maxgen = 200
    sizepop = 80
    Vmax = 1
    Vmin = 0
    popmax = 2
    popmin = 0
    pso=PSO(wmax,wmin,maxgen,sizepop,Vmax,Vmin,popmax,popmin)
    record,zbest=pso.pso(file_name)
    print(record)
    mi=Main(file_name,maxgen,sizepop,record,zbest)
    # 保存参数结果
    mi.save_result()
    # 绘制迭代曲线图
    mi.draw_rmse()
    # 绘制预测值与真实值之间的对比图
    draw_value()


if __name__ == '__main__':
    main()


