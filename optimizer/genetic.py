#! /usr/bin/env python3
import numpy as np
import math
from utils import otimization
from point import Point


class Genetic(object):

    def __init__(self, func, params):
        """ the Genetic Algorithm method

        :param func: objective function object
        :param params: dict, tuning parameter, name: (min, max)

        ---
        Genetic(
             lambda x: x[0]**2 + 5 * x[1],
             {
                 "x1": (1, 5),
                 "x2": (0, 3),
             }
        )

        """

        self.initialized = False
        self.otm = otimization(func, params)

    def initialize(self, init_params):
        """ Inialize first simplex point

        :param init_params(list):

        """
        assert len(init_params) == (self.otm.dim +
                                    1), "Invalid the length of init_params"
        for param in init_params:
            p = Point(self.dim)
            p.p = np.array(param, dtype=np.float32)
            self.simplex.append(p)
        self.initlized = True

    def optimize(self, n_iter=25, minimize=True, pop_len=16, mating_size=8, mutation_ratio=0.01):
        """ Minimize or maximize the objective function.

        :param n_iter: the number of iterations for the nelder_mead method
        :param minimize: True to minimize and False to Maximize
        :param pop_len: the population size
        :param mating_size: mating pool size
        :param mutation_ratio: chance of a mutation to occur on each gene

        """

        self.otm._coef = 1 if minimize else -1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt()

    def _opt(self):
        self.otm.header()

        if not self.initialized:
            self.pop = self.otm.initialize(self.pop_len)
            self.initialized = True

        for p in self.pop:
            p.v = self.otm.func_impl(p.p)

        for i in range(self.n_iter):
            self.pop = self.otm.sort(self.pop)

            parents = self.pop[:self.mating_size]

            offsprings = []

            for j in range(len(parents)//2):
                locus = np.random.randint(self.otm.dim-1)+1
                p1 = Point(self.otm.dim)
                p2 = Point(self.otm.dim)
                a = list(parents[j].p)
                b = list(parents[j+1].p)
                c = a[:locus]+b[locus:]
                d = b[:locus]+a[locus:]
                p1.p = np.array(c)
                p2.p = np.array(d)
                offsprings.append(p1)
                offsprings.append(p2)

            for offspring in offsprings:
                for j in range(self.otm.dim):
                    if np.random.random() < self.mutation_ratio:
                        offspring.p[j] = ((self.otm.p_max[j] - self.otm.p_min[j]) *
                                          np.random.random() + self.otm.p_min[j])
                offspring.v = self.otm.func_impl(offspring.p)

            self.pop[len(parents):] = offsprings

        self.pop = self.otm.sort(self.pop)
        self.otm.print_best(self.pop[0])
