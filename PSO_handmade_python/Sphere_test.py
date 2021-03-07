#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: Sphere_test.py
@Abstract: Test for PSO
@Time: 2021/03/08 00:04:27
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from PSOKW import PSOKW

def new_penalty_func(k):
    return math.sqrt(k)

def objective(x):
    #Process Flow Sheeting Problem
    res = np.zeros_like(x[:,1])
    for i in range(9):
        res = res + (x[:,i])**2
    return res

def c1(x):
    return x-100

def c2(x):
    return -100-x

constraints = []
num_runs = 30

iter_w_recorder = []
pso_recorder = []
pso_recorder_iter = []
#Run several time to get mean

for _ in range(num_runs):
    pso = PSOKW(dim=10, fitness=objective, constraints=constraints, population_size=50)
    pso.c1 = 1.5
    pso.c2 = 2.5
    pso.h = new_penalty_func
    pso.init_Population(low=-100, high=100) # Bound of Union of xi
    pso.solve()

    fit = pso.fit
    # obj fitness
    x = pso.gbest.reshape((1,-1))
    obj_fit = objective(x)
    pso_recorder.append(obj_fit)
    pso_recorder_iter.append(pso.record_fit)
pso_recorder = [float(i) for i in pso_recorder]
best_index = pso_recorder.index(min(pso_recorder))
pso_recorder = np.log(pso_recorder)
pso_recorder_iter = np.log(pso_recorder_iter)


plt.figure(figsize=(15,6))
plt.subplot(121)
plt.grid(linestyle='-.')
plt.title('Different Results under Multiple Runs')
plt.plot(np.arange(1,len(pso_recorder)+1,1),pso_recorder, linewidth=1,linestyle='-',marker='.', color='darkgreen')
plt.xticks([1,5,10,15,20,25,30])
plt.xlabel('Run times', fontsize=12)
plt.ylabel('Logarithmic Fitness Value', fontsize=12)

plt.subplot(122)
plt.grid(linestyle='-.')
plt.title('Optimization Process for\nBest Result among Multiple Runs')
plt.plot(np.arange(1,len(pso_recorder_iter[best_index])+1,1),(pso_recorder_iter[best_index]), linewidth=1, color='darkcyan', label='Best Fitness')
plt.xlabel('Number of Generation', fontsize=12)
plt.ylabel('Logarithmic Fitness Value', fontsize=12)
plt.suptitle('Sphere Function\nBast Known Result = 0, Best Result by this Algorithm = '+str(np.exp(pso_recorder[best_index])), fontsize=14)
plt.show()