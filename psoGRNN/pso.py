#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/23 9:06
__author__ = 'wangwenhui'

import numpy as np
import random
import math
from psoGrnn.fitness import fitness_rmse



class PSO:
    def __init__(self, wmax, wmin, maxgen, sizepop, Vmax, Vmin, popmax, popmin):
        '''初始化粒子群算法的参数
        Parameters:
         ----------
        w: 权重系数
        c1: 学习因子
        c2: 学习因子
        maxgen: 进化迭代次数
        sizepop: 种群规模
        Vmax: 最大速度
        Vmin: 最小速度
        popmax: 最大位置
        popmin: 最小位置
        '''
        self.wmax = wmax
        self.wmin = wmin
        self.maxgen = maxgen
        self.sizepop = sizepop
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.popmax = popmax
        self.popmin = popmin

    def pso(self,file_name):
        # 产生初始粒子和速度
        pop=np.random.uniform(0.1,1,size=self.sizepop)
        v=np.random.uniform(0.1,1,size=self.sizepop)
        fitness=np.array([])
        for i in pop:
            fitness=np.append(fitness,fitness_rmse(i,file_name))
        # 找最好的个体
        i=np.argmin(fitness)
        # 记录个体最优位置
        gbest=pop.copy()
        # 记录群体最优位置
        zbest=pop[i]
        # 个体最佳适应度值
        fitness_gbest=fitness.copy()
        # 全局最佳适应度值
        fitness_zbest=fitness[i]

        # 迭代寻优
        t=0
        record=np.zeros(self.maxgen)
        while t<self.maxgen:
            # 改进粒子群算法的惯性系数和学习因子
            w=self.wmax-(self.wmax-self.wmin)*math.sqrt(t/self.maxgen)
            c1=2*math.sqrt(math.cos(math.pi/2*(t/self.maxgen)))
            c2=2*math.sqrt(1-math.cos(math.pi/2*(t/self.maxgen)))
            # 速度更新
            v=w*v+c1*random.random()*(gbest-pop)+c2*np.random.random()*(zbest-pop)
            # 限制速度
            v[v>self.Vmax]=self.Vmax
            v[v<self.Vmin]=self.Vmin
            # 位置更新
            pop=pop+0.1*v
            # 限制位置
            pop[pop>self.popmax]=self.popmax
            pop[pop<self.popmin]=self.popmin
            # 计算适应度
            for i in range(len(pop)):
                fitness[i]=fitness_rmse(pop[i], file_name)
            # 个体最优位置更新
            idx1=fitness<fitness_gbest
            fitness_gbest[idx1]=fitness[idx1]
            gbest[idx1]=pop[idx1]
            # 群体最优更新
            idx2=np.argmin(fitness)
            if fitness[idx2]<fitness_zbest:
                zbest=pop[idx2]
                fitness_zbest=fitness[idx2]
            # 记录群体最优位置的变化
            record[t]=fitness_zbest
            t=t+1
        # 结果分析
        return record,zbest


