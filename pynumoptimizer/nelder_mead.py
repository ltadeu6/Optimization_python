#! /usr/bin/env python3
import numpy as np
from pynumoptimizer.utils import otimization
from pynumoptimizer.point import Point


class NelderMead(object):

    def __init__(self, func, params):
        """ the Nelder-Mead method

        :param func: objective function object
        :param params: dict, tuning parameter, name: (min, max)

        ---
        NelderMead(
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

        assert len(init_params) == (
            self.otm.dim + 1), "Invalid the length of init_params"
        for param in init_params:
            p = Point(self.otm.dim)
            p.x = np.array(param, dtype=np.float32)
            self.simplex.append(p)
        self.initlized = True

    def optimize(self, n_iter=20, minimize=True, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
        """ Minimize or maximize the objective function.

        :param n_iter: the number of iterations for the nelder_mead method
        :param minimize: True to minimize and False to Maximize
        :param delta_r: the parameter of reflect
        :param delta_e: the parameter of expand
        :param delta_ic: the parameter of inside contraction
        :param delta_oc: the parameter of outside contraction
        :param gamma_s: the parameter of shrink

        """

        self.otm._coef = 1 if minimize else -1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt()

    def _opt(self):
        self.otm.header()

        if not self.initialized:
            self.simplex = self.otm.initialize(self.otm.dim + 1)
            self.initialized = True
        for p in self.simplex:
            p.v = self.otm.func_impl(p.p)

        for i in range(self.n_iter):
            self.simplex = sorted(self.simplex, key=lambda p: p.v)

            p_centroid = self._centroid()

            p_reflected = self._reflect(p_centroid)

            if p_reflected < self.simplex[0]:
                p_expanded = self._expand(p_centroid)

                if p_expanded < p_reflected:
                    self.simplex[-1] = p_expanded
                else:
                    self.simplex[-1] = p_reflected
                continue
            elif p_reflected > self.simplex[-2]:
                if p_reflected <= self.simplex[-1]:
                    p_contracted = self._outside(p_centroid)
                    if p_contracted < p_reflected:
                        self.simplex[-1] = p_contracted
                    else:
                        self.simplex[-1] = p_reflected
                    continue
                elif p_reflected > self.simplex[-1]:
                    p_contracted = self._inside(p_centroid)
                    if p_contracted < self.simplex[-1]:
                        self.simplex[-1] = p_contracted
                        continue

                for j in range(len(self.simplex) - 1):
                    p = Point(self.otm.dim)
                    p.p = self.simplex[0].p + self.gamma_s * \
                        (self.simplex[j+1].p - self.simplex[0].p)
                    p.v = self.otm.func_impl(p.p)
                    self.simplex[j+1] = p
            else:
                self.simplex[-1] = p_reflected

        self.simplex = self.otm.sort(self.simplex)
        self.otm.print_best(self.simplex[0])

    def _inside(self, p_c):
        return self._generate_point(p_c, self.delta_ic)

    def _outside(self, p_c):
        return self._generate_point(p_c, self.delta_oc)

    def _expand(self, p_c):
        return self._generate_point(p_c, self.delta_e)

    def _reflect(self, p_c):
        return self._generate_point(p_c, self.delta_r)

    def _generate_point(self, p_c, x_coef):
        p = Point(self.otm.dim)
        p.p = p_c.p + x_coef * (p_c.p - self.simplex[-1].p)
        p.v = self.otm.func_impl(p.p)
        return(p)

    def _centroid(self):
        p_c = Point(self.otm.dim)
        x_sum = []
        for p in self.simplex[:-1]:
            x_sum.append(p.p)
        p_c.p = np.mean(x_sum, axis=0)
        return p_c
