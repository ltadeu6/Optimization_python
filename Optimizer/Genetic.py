#! /usr/bin/env python3
import numpy as np
import math
from Utils import otimization
from Point import Point

class Genetic(object):

    def __init__(self, func, params):
        self.initialized = False
        self.otm = otimization(func, params)

    def otimize(self, n_iter=25, minimize=True, pop_len=16, mating_size=8, mutation_ratio=0.01):
        if minimize:
            self.otm._coef = 1
        else:
            self.otm._coef = -1
        variables = locals()
        for k,v in variables.items():
            setattr(self, k, v)
        self._opt()


    def _opt(self):
        self.otm.header()

        if not self.initialized:
            self.pop = self.otm.initialize(self.pop_len)

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

def main():
    def senoide (x):
        return sum(t + 5 * math.sin(5 * t) + 2 * math.cos(3 * t) for t in x)

    params = {
            "x1":["real", (0,10)],
            "x2":["real", (0,10)],
            }

    ga = Genetic(senoide, params)
    ga.otimize(n_iter = 25000, minimize=False, mutation_ratio = 0.5)

if __name__ == "__main__":
    main()
