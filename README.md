# Python numeric optimization
Creates an environment for the development of numerical optimization codes in addition to some known methods already implemented

## Dependencies
numpy

# Usage


``` python

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

```
