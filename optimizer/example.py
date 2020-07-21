#! /usr/bin/env python3
from nelder_mead import NelderMead
from genetic import Genetic
import argparse
import math

def main(algorithm):
    def senoide (x):
        return sum(t + 5 * math.sin(5 * t) + 2 * math.cos(3 * t) for t in x)

    params = {
            "x1":["real", (0,10)],
            "x2":["real", (0,10)],
            }
    if algorithm == "Genetic":
        ga = Genetic(senoide, params)
        ga.optimize(n_iter = 250, minimize=False, mutation_ratio = 0.5)

    elif algorithm == "NelderMead":
        nm = NelderMead(senoide, params)
        nm.optimize(n_iter=250, minimize=False)

if __name__ == "__main__":

    algorithms = ["Genetic", "NelderMead"]

    parser = argparse.ArgumentParser(description="Demonstrates an otimizing algorithm")

    parser.add_argument('algorithm', choices=algorithms, metavar=('Algorithm'), help='Chooses the algorithm. Allowed values: '+', '.join(algorithms)+'.')

    args = parser.parse_args()

    main(args.algorithm)
