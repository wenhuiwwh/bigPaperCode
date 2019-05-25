#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/5/1 16:24
__author__ = 'wangwenhui'

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl



class ASME:
    def __init__(self,dect_path,depth_path,predict_path,first_time,second_time,third_time):
        '''
        :param dect_path: 腐蚀薄弱管段开挖检测数据
        :param depth_path: 各管段最大检测腐蚀深度检测数据
        :param predict_path: 管段腐蚀深度预测数据
        :param first_time:第一次的检测时间点
        :param second_time:第二次的检测时间点
        :param third_time: 未来预测的时间点
        '''
        self.dect_path=dect_path
        self.depth_path=depth_path
        self.predict_path=predict_path
        self.t1=first_time
        self.t2=second_time
        self.t3=third_time

    def read_csv(self,path):
        df=pd.read_csv(path,header=None,skiprows=2,index_col=0, encoding='gbk')
        return df

    def maxcorrosion(self, d, L, d0, Ps, D=338.6, Py=2.5, ds=0.0001):
        '''
        :param d: 腐蚀缺陷的深度
        :param L: 腐蚀缺陷的长度
        :param D: 管段的直径
        :param d0: 管段的厚度
        :param Ps: 管段的屈服强度
        :param Py: 管段运行压力
        :param ds: 腐蚀缺陷深度的迭代步长值
        :return: 最大允许腐蚀深度
        '''
        k=L/d
        d+=ds
        L=k*d
        M=math.sqrt(1 + ((0.8*L**2) / (D * d0)))
        if L**2/(D * d0)<=20:
            P= ((2 * d0 * 1.1 * Ps) / D) * (1 - (2 / 3) * (d / d0) / (1 - (2 / 3) * (d / d0) * (1 / M)))
        else:
            P= ((2 * d0 * 1.1 * Ps) / D) * ((1 - (d / d0)) / (1 - ((d / d0) * (1 / M))))
        if P>Py:
            d+=ds
            L=k*d
            M=math.sqrt(1 + ((0.8*L**2) / (D * d0)))
            while P>Py:
                d+=ds
                L=k*d
                M=math.sqrt(1 + ((0.8*L**2) / (D * d0)))
                if L**2/(D * d0)<=20:
                    P= ((2 * d0 * 1.1 * Ps) / D) * (1 - (2 / 3) * (d / d0) / (1 - (2 / 3) * (d / d0) * (1 / M)))
                else:
                    P= ((2 * d0 * 1.1 * Ps) / D) * ((1 - (d / d0)) / (1 - ((d / d0) * (1 / M))))
        return d

    def allow_max_depth(self):
        '''求出最大允许腐蚀深度'''
        df=self.read_csv(self.dect_path)
        ss=pd.Series()
        index=df.index
        for i in range(len(df)):
            ss['%s' % index[i]] = self.maxcorrosion(df.iloc[i, 2], df.iloc[i, 1], df.iloc[i, 0], df.iloc[i, 3])
        return ss

    def remain_life(self):
        '''
        dect_df[1]: 管道初始壁厚
        depth_df[1]: 第一次检测时的最大腐蚀深度
        depth_df[2]: 第二次检测时的最大腐蚀深度
        corr_depth: 管道的最大允许腐蚀深度
        :return: 管道的腐蚀剩余寿命
        '''
        corr_depth=self.allow_max_depth()
        depth_df=self.read_csv(self.depth_path)
        dect_df=self.read_csv(self.dect_path)
        _a=self.t1/(self.t1-self.t2)*np.log((dect_df[1]-depth_df[1])/(dect_df[1]-depth_df[2]))
        a=(dect_df[1]-depth_df[1])/np.exp(_a)
        b=np.log((dect_df[1]-depth_df[1])/(dect_df[1]-depth_df[2]))/(self.t1-self.t2)
        remain_life=1.0/b*np.log((dect_df[1].values-corr_depth.values)/a)
        return remain_life-8

    def depth_length(self,d0=8.0):
        '''预测未来检测时间点的管道最大腐蚀深度'''
        predict_df=self.read_csv(self.predict_path)
        _a=self.t1/(self.t1-self.t2)*np.log((d0-predict_df[1])/(d0-predict_df[2]))
        a = (d0 - predict_df[1]) / np.exp(_a)
        b=np.log((d0-predict_df[1])/(d0-predict_df[2]))/(self.t1-self.t2)
        dx=d0-a*np.exp(b*self.t3)
        return a,b,dx

    def grid_view(self, a, b, his_value,future_value, pipe_num,d0=8.0):
        '''绘制管段腐蚀发展趋势图'''
        plt.xlabel("时间点/年", fontsize=14)
        plt.ylabel("最大腐蚀深度值/mm", fontsize=14)
        x=np.linspace(0.5,3,50)
        y=[d0-a*math.exp(b*i) for i in x]
        plt.plot(x, y, label="%s号管段腐蚀发展曲线" % pipe_num)
        x1 = [self.t1, self.t2, self.t3]
        y1 = [his_value.iloc[0], his_value.iloc[1], future_value]
        for i in range(len(x1)):
            plt.vlines(x1[i],0,y1[i],color='green',linestyles='--',lw=1.0)
            plt.hlines(y1[i],0,x1[i],color='green',linestyles='--',lw=1.0)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)  # 设置纵坐标刻度起始点
        plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))  # 删除x的第一个刻度
        plt.tick_params(direction='in')  # 设置坐标刻度朝里
        plt.xticks(family='Times New Roman',fontsize=12)
        plt.yticks(family='Times New Roman',fontsize=12)
        plt.scatter(x1, y1, color='red', marker='s', label="实测值")  # 实测值
        plt.legend(fontsize=12, edgecolor='black', loc='upper left')
        figure_fig = plt.gcf()
        plt.rcParams['savefig.dpi'] = 2000  # 设置图片保存分辨率
        figure_fig.savefig('pipe_No%s.png'%pipe_num, format='png')
        plt.show()
        return None

    def batch_deal(self):
        '''批量生成图片'''
        mpl.rcParams['font.sans-serif']=[u'SimHei']
        mpl.rcParams['axes.unicode_minus']=False
        a,b,dx=self.depth_length()
        future_dict={a.index[0]:0.96,a.index[1]:0.75,a.index[2]:0.42,a.index[3]:0.69}
        depth_df=self.read_csv(self.depth_path)
        for i in a.index:
            self.grid_view(a.loc[i],b.loc[i],depth_df.loc[i],future_dict[i],i)
        return '批量保存成功'

    def grid_remainlife(self,expect_life):
        mpl.rcParams['font.sans-serif']=[u'SimHei']
        mpl.rcParams['axes.unicode_minus']=False
        # 设置所有的字体大小
        mpl.rcParams.update({'font.size':22})
        plt.figure(figsize=(12,8))
        plt.hlines(expect_life, 0, 15, color='red', linestyle='--', lw=2.0)
        x = self.read_csv(self.dect_path).index
        _x=list(range(16))
        y=self.remain_life().values
        plt.plot(_x, y, '-o', mec='green', mfc='green', lw=2.0, label="各管段腐蚀剩余寿命")
        plt.ylim(ymin=0, ymax=45)
        plt.xticks(_x,x)
        plt.xlabel("管段编号")
        plt.ylabel("腐蚀剩余寿命/年")
        plt.annotate('预计剩余寿命', xy=(5, 27), xytext=(7, 32),arrowprops=dict(headwidth=8.0, facecolor='red', edgecolor='red', width=1.0))
        # 设置坐标刻度朝里
        plt.tick_params(direction='in')
        plt.xticks(family='Times New Roman')
        plt.yticks(family='Times New Roman')
        plt.legend(edgecolor='black')

        plt.rcParams['figure.dpi'] = 1000
        plt.savefig('remain_life.png', dpi=1000)
        return '图片生成成功！'

def main():
    dect_path='corr_dect.csv'
    depth_path='allow_depth.csv'
    predict_path='predict_values.csv'
    first_time=0.5
    second_time=1.0
    third_time=1.5
    expect_life=35-8
    asme=ASME(dect_path,depth_path,predict_path,first_time,second_time,third_time)
    allow_depth=asme.allow_max_depth()
    remain_life=asme.remain_life()
    predict_depth=asme.depth_length()
    asme.grid_remainlife(expect_life)



if __name__ == '__main__':
    main()



