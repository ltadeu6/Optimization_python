#! /usr/bin/env python3
import numpy as np
import math
from Point import Point

class otimization(object):
    def __init__(self, func, params):
        self.func = func
        self.dim = len(params)
        self.names = []
        self.n_eval = 0
        self.p_types = []
        self.p_min = []
        self.p_max =[]
        self.obj=[]
        self._parse_minmax(params)

    def _parse_minmax(self, params):
        for name, values in params.items():
            self.names.append(name)
            self.p_types.append(values[0])
            self.p_min.append(values[1][0])
            self.p_max.append(values[1][1])

    def sort(self, obj):
        return sorted(obj, key=lambda p: p.v)

    def print_best(self, best):
        best.v = best.v * self._coef
        print("\nBest Point: {}".format(best))

    def header(self):
        print("{:>5} | {} | {:>15}".format(
            "Eval",
            " | ".join(["{:>15}".format(name) for name in self.names]),
            "ObjVal"
        ))
        print("-" * (20 + self.dim * 20))

    def func_impl(self, x):
        objval, valid = None, True
        for i, t in enumerate (x):
            if t < self.p_min[i] or t > self.p_max[i]:
                objval = float("inf")
                valid = False
        if valid:
            x = [int(np.round(x_t)) if p_t == "integer" else x_t for p_t, x_t in zip(self.p_types, x)]
            objval = self._coef * self.func(x)

        print("{:5d} | {} | {:>15.5f}".format(
            self.n_eval,
            " | ".join(["{:>15.5f}".format(t) for t in x]),
            self._coef * objval
        ))

        self.n_eval += 1
        return(objval)

    def initialize(self, obj_len):
        for i in range(obj_len):
            p = Point(self.dim)
            init_val = [(m2 - m1) * np.random.random() + m1 for m1, m2 in zip(self.p_min, self.p_max)]
            p.p = np.array(init_val)
            self.obj.append(p)
        return self.obj
