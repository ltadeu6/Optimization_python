#! /usr/bin/env python3
import math
import numpy as np
from Utils import otimization
from Point import Point

class NelderMead(object):

    def __init__(self, func, params):
        self.initialized = False
        self.otm = otimization(func, params)

    def otimize(self, n_iter=20, minimize=True, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
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
            self.simplex = self.otm.initialize(self.otm.dim + 1)
        for p in self.simplex:
            p.v = self.otm.func_impl(p.p) #calcula objetivo nos vértices

        for i in range(self.n_iter):
            self.simplex = sorted(self.simplex, key=lambda p: p.v)

            p_centroide = self._centroid()

            p_refletido = self._refletir(p_centroide)


            if p_refletido < self.simplex[0]:
                p_expandido = self._expandir(p_centroide)

                if p_expandido < p_refletido:
                    self.simplex[-1] = p_expandido
                else:
                    self.simplex[-1] = p_refletido
                continue
            elif p_refletido > self.simplex[-2]:
                if p_refletido <= self.simplex[-1]:
                    p_contraido = self._outside(p_centroide)
                    if p_contraido < p_refletido:
                        self.simplex[-1] = p_contraido
                    else:
                        self.simplex[-1] = p_refletido
                    continue
                elif p_refletido > self.simplex[-1]:
                    p_contraido = self._inside(p_centroide)
                    if p_contraido < self.simplex[-1]:
                        self.simplex[-1] = p_contraido
                        continue
                # encolher

                for j in range(len(self.simplex) - 1):
                    p = Point(self.otm.dim)
                    p.p = self.simplex[0].p + self.gamma_s * (self.simplex[j+1].p - self.simplex[0].p)
                    p.v = self.otm.func_impl(p.p)
                    self.simplex[j+1] = p
            else:
                self.simplex[-1] = p_refletido

        self.simplex = self.otm.sort(self.simplex)
        self.otm.print_best(self.simplex[0])

    def _inside(self, p_c):
        return self._generate_point(p_c, self.delta_ic)

    def _outside(self, p_c):
        return self._generate_point(p_c, self.delta_oc)

    def _expandir(self, p_c):
        return self._generate_point(p_c, self.delta_e)

    def _refletir(self, p_c):
        return self._generate_point(p_c, self.delta_r)

    def _generate_point(self, p_c, x_coef):
        p = Point(self.otm.dim)
        p.p = p_c.p + x_coef * (p_c.p - self.simplex[-1].p)
        p.v = self.otm.func_impl(p.p)
        return(p)

    def _centroid (self):
        p_c = Point(self.otm.dim)
        x_sum = []
        for p in self.simplex[:-1]:
            x_sum.append(p.p)
        p_c.p = np.mean(x_sum, axis=0)
        return p_c

def main():

    def parabola (x):
        return sum(t**2 for t in x)
    def senoide (x):
        return sum(t + 5 * math.sin(5 * t) + 2 * math.cos(3 * t) for t in x)

    func = senoide
    params = {
        "x1": ["real", (0, 10)],
        "x2": ["real", (0,10)],
    }

    nm = NelderMead(func, params)
    nm.otimize(n_iter=25)

if __name__ == "__main__":
    main()
