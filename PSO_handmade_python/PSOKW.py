#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: PSOKW.py
@Abstract: 
@Time: 2021/03/05 19:24:54
@Requirements: PSO of K(w*v+c1*rand1*(pbest-x)+c2*rand2*(gbest-x)) with constraints
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''


import numpy as np
import math
import random
import matplotlib.pyplot as plt

class PSOKW:

    def __init__(self,
        population_size=50,
        max_iter=1000,
        dim=10,
        fitness=None,
        constraints=None):
        '''
        Particle Swarm Optimization Constraint Optimization
        Args:
            population_size (int): The number of particles
            max_iter (int): Max iteration
            dim (int): Dimension of solution
            fitness (callable function): Fitness function
            constraints (list): Standard Constraints
        '''
        self.c1 = 1.5 #cognition component
        self.c2 = 2.5 #social component
        self.w = 1 # Gradually reduce to 0.1
        self.kai = 0.7 #Capital K
        self.vmax = 4 # max velocity
        self.population_size = population_size
        self.max_iter = max_iter
        self.dim = dim
        self.record_pos = []
        self.record_fit = []
        self.w_recorder = []

        # pso parameters 
        self.X = np.zeros((self.population_size, self.dim)) #Size: N*D
        self.V = np.zeros((self.population_size, self.dim))
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.population_size)
        self.fit = float("inf")
        self.iter = 1

        self.constraints = constraints
        if constraints is not None:
            for cons in constraints:
                if not callable(cons):
                    raise Exception("Invalid Constraint")
        if not callable(fitness):
            raise Exception("Invalid Fitness Function")
        self.sub_fitness = fitness
    
    # power of penalty function 
    def gamma(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore >= 1] = 2
        result[qscore < 1] = 1 
        return result
    
    # multi-assignment function
    def theta(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore < 0.001] = 1
        result[qscore <= 0.1] = 1
        result[qscore <= 1] = 10
        result[qscore > 1] = 30
        return result

    # relative violated function
    def q(self, g):
        return np.maximum(0, g)
    
    # penalty score 
    def h(self, k):
        return k * math.sqrt(k)
    
    # penalty factor
    def H(self, x):
        res = 0
        for cons_func in self.constraints:
            qscore = self.q(cons_func(x))
            if len(qscore.shape) == 1 or qscore.shape[1] == 1:
                qscore = qscore.reshape((-1, 1))
                res += self.theta(qscore) * np.power(qscore, self.gamma(qscore))
            else:
                for i in range(qscore.shape[1]):
                    qscorei = qscore[:, i].reshape((-1, 1))
                    res += self.theta(qscorei) * \
                        np.power(qscorei, self.gamma(qscorei))
        return res 

    def fitness(self, x, k):
        '''fitness function + penalty function'''
        obj = self.sub_fitness(x).reshape((-1,1))
        #obj = obj.reshape((-1, 1))
        return obj + self.h(k) * self.H(x)
    
    def init_Population(self, low=0, high=1):  
        '''Initialization'''
        self.X = np.random.uniform(size=(self.population_size, self.dim), low=low, high=high)
        self.V = np.random.uniform(size=(self.population_size, self.dim))
        self.pbest = self.X 
        self.p_fit = self.fitness(self.X, 1)
        best = np.min(self.p_fit)
        best_idx = np.argmin(self.p_fit)
        if best < self.fit:
            self.fit = best 
            self.gbest = self.X[best_idx] 
    
    def solve(self):  
        '''Solving process'''
        fitness = []  
        w_step = (self.w - 0.1) / self.max_iter
        for k in range(1, self.max_iter+1):  
            tmp_obj = self.fitness(self.X, k) 

            # Update pbest
            stack = np.hstack((tmp_obj.reshape((-1, 1)), self.p_fit.reshape((-1, 1))))
            best_arg = np.argmin(stack, axis=1).ravel().tolist()
            self.p_fit = np.minimum(tmp_obj, self.p_fit)
            X_expand = np.expand_dims(self.X, axis=2)
            p_best_expand = np.expand_dims(self.pbest, axis=2)
            concat = np.concatenate((X_expand, p_best_expand), axis=2)
            self.pbest = concat[range(0, len(best_arg)), :, best_arg]

            # Update fitness and gbest
            best = np.min(self.p_fit)
            best_idx = np.argmin(self.p_fit)
            if best < self.fit:
                self.fit = best 
                self.gbest = self.X[best_idx]

            # Update velocity

            
            '''
            # Update by particle
            for i in range(self.population_size):  
                self.V[i] = self.w*self.V[i] + self.c1*random.random()*(self.pbest[i] - self.X[i]) + \
                             self.c2*random.random()*(self.gbest - self.X[i])  
                self.X[i] = self.X[i] + self.V[i] 
            '''
            rand1 = np.random.random(size=(self.population_size, self.dim))
            rand2 = np.random.random(size=(self.population_size, self.dim))
            # Update by swarm
            self.V = self.kai * (self.w*self.V + self.c1*rand1*(self.pbest - self.X) + \
                        self.c2*rand2*(self.gbest - self.X))
            self.V[self.V > self.vmax] = self.vmax
            self.V[self.V < -self.vmax] = -self.vmax
            
            self.X = self.X + self.V  
            fitness.append(self.fit)  
            #record w
            self.w_recorder.append(self.w)

            self.w -= w_step

            #record
            self.record_fit.append(self.fit)
            self.record_pos.append(self.X[best_idx])


        return fitness 

    def fitness(self, x, k):
        '''fitness function + penalty function'''
        obj = self.sub_fitness(x).reshape((-1,1))
        #obj = obj.reshape((-1, 1))
        return obj + self.h(k) * self.H(x)
    
    def init_Population(self, low=0, high=1):  
        '''Initialization'''
        self.X = np.random.uniform(size=(self.population_size, self.dim), low=low, high=high)
        self.V = np.random.uniform(size=(self.population_size, self.dim))
        self.pbest = self.X 
        self.p_fit = self.fitness(self.X, 1)
        best = np.min(self.p_fit)
        best_idx = np.argmin(self.p_fit)
        if best < self.fit:
            self.fit = best 
            self.gbest = self.X[best_idx] 
    
    def solve(self):  
        '''Solving process'''
        fitness = []  
        w_step = (self.w - 0.1) / self.max_iter
        for k in range(1, self.max_iter+1):  
            tmp_obj = self.fitness(self.X, k) 

            # Update pbest
            stack = np.hstack((tmp_obj.reshape((-1, 1)), self.p_fit.reshape((-1, 1))))
            best_arg = np.argmin(stack, axis=1).ravel().tolist()
            self.p_fit = np.minimum(tmp_obj, self.p_fit)
            X_expand = np.expand_dims(self.X, axis=2)
            p_best_expand = np.expand_dims(self.pbest, axis=2)
            concat = np.concatenate((X_expand, p_best_expand), axis=2)
            self.pbest = concat[range(0, len(best_arg)), :, best_arg]

            # Update fitness and gbest
            best = np.min(self.p_fit)
            best_idx = np.argmin(self.p_fit)
            if best < self.fit:
                self.fit = best 
                self.gbest = self.X[best_idx]

            # Update velocity

            
            '''
            # Update by particle
            for i in range(self.population_size):  
                self.V[i] = self.w*self.V[i] + self.c1*random.random()*(self.pbest[i] - self.X[i]) + \
                             self.c2*random.random()*(self.gbest - self.X[i])  
                self.X[i] = self.X[i] + self.V[i] 
            '''
            rand1 = np.random.random(size=(self.population_size, self.dim))
            rand2 = np.random.random(size=(self.population_size, self.dim))
            # Update by swarm
            self.V = self.kai * (self.w*self.V + self.c1*rand1*(self.pbest - self.X) + \
                        self.c2*rand2*(self.gbest - self.X))
            self.V[self.V > self.vmax] = self.vmax
            self.V[self.V < -self.vmax] = -self.vmax
            
            self.X = self.X + self.V  
            fitness.append(self.fit)  
            #record w
            self.w_recorder.append(self.w)

            self.w -= w_step

            #record
            self.record_fit.append(self.fit)
            self.record_pos.append(self.X[best_idx])


        return fitness 