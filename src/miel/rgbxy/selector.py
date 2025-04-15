import math
from random import Random

from matplotlib import pyplot as plt

from miel.rgbxy.genome import RGBXYGenome


class RGBXYGenomeSelector:
    def __init__(self):
        self.rng = Random(0)
        self.population = None

    def pre_select(self, population):
        self.population = list(population)
        return self.rng.choices(self.population, k=3)

    def select(self, population: list[RGBXYGenome]) -> list[float]:
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        preferences = None

        def onclick(event):
            nonlocal preferences
            dists = [math.sqrt((event.xdata - ind.genome.x) ** 2
                               + (event.ydata - ind.genome.y) ** 2)
                     for ind in population]
            ranks = [
                (j, *t) for j, t in enumerate(
                    sorted([(i, d) for i, d in enumerate(dists)],
                           key=lambda _t: _t[1])
                )
            ]
            ranks = [t[0] for t in sorted(ranks, key=lambda _t: _t[1])]
            preferences = [-2 * r / (len(dists) - 1) + 1 for r in ranks]
            plt.close(fig=fig)
        fig.canvas.mpl_connect('button_press_event', onclick)

        x, y, c = zip(*[(g.x, g.y, (g.r, g.g, g.b, .1))
                        for ind in self.population if (g := ind.genome)])
        ax.scatter(x=x, y=y, c=c)

        x, y, c = zip(*[(g.x, g.y, (g.r, g.g, g.b))
                        for ind in population if (g := ind.genome)])
        ax.scatter(x=x, y=y, c=c)

        plt.show(block=True)
        assert preferences is not None
        return preferences
