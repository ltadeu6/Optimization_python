#! /usr/bin/env python3
import numpy as np
from pynumoptimizer.utils import otimization
from pynumoptimizer.point import Point


class PSO(object):

    def __init__(self, func, params):
        """ the Particle particle swarm optimization (PSO) method

        :param func: objective function object
        :param params: dict, tuning parameter, name: (min, max)

        ---
        Pso(
             lambda x: x[0]**2 + 5 * x[1],
             {
                 "x1": (1, 5),
                 "x2": (0, 3),
             }
        )

        """

        self.otm = otimization(func, params)

    def optimize(self, n_iter=500, minimize=True, swarmsize=15, omega=0.5, phip=0.5, phig=0.5):
        """ Minimize or maximize the objective function.

        :param n_iter: The number of iterations for the method
        :param minimize: True to minimize and False to Maximize
        :param swarmsize: The number of particles
        :param omega: Particle velocity scaling factor
        :param phip: scaling factor to search away from the particle's best known position
        :param phig: Scaling factor to search away from the swarm's best known position

        """

        self.otm._coef = 1 if minimize else -1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt()

    def _opt(self):
        self.otm.header()

        particles = self.otm.initialize(self.swarmsize)

        globalbest = particles[0]

        p_min = np.array(self.otm.p_min)
        p_max = np.array(self.otm.p_max)
        p_range = np.abs(p_max - p_min)

        for p in particles:
            p.v = self.otm.func_impl(p.p)
            p.vel = (np.random.rand(self.otm.dim) - 0.5 *
                     np.ones(self.otm.dim)) * p_range
            p.bestposition = p.p
            p.bestvalue = p.v
            if p < globalbest:
                globalbest = p

        for i in range(self.n_iter):

            for p in particles:
                p.vel = self.omega * p.vel + self.phig * np.random.rand(self.otm.dim) * (
                    p.bestposition - p.p) - self.phip * np.random.rand(self.otm.dim) * (globalbest.p - p.p)
                p.p = p.p + p.vel
                p.v = self.otm.func_impl(p.p)
                if p.v < p.bestvalue:
                    p.bestposition = p.p
                    p.bestvalue = p.v
                    if p < globalbest:
                        globalbest = p
        self.otm.print_best(globalbest)
