#! /usr/bin/env python3
import numpy as np


class Point(object):
    def __init__(self, dim):
        self.p = np.zeros(dim)
        self.v = 0

    def __str__(self):
        return "Params: {}, ObjValue: {}".format(", ".join(["{:>10.5f}".format(p) for p in self.p]),
                                                 self.v)

    def __eq__(self, rhs):
        return self.v == rhs.v

    def __lt__(self, rhs):
        return self.v < rhs.v

    def __le__(self, rhs):
        return self.v <= rhs.v

    def __gt__(self, rhs):
        return self.v > rhs.v

    def __ge__(self, rhs):
        return self.v >= rhs.v


def main():
    p = Point(3)
    print(p)


if __name__ == "__main__":
    main()
