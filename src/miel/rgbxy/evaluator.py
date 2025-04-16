import math
import warnings
from random import Random
from typing import Iterable, Optional

from matplotlib import pyplot as plt

from ..api import Evaluator, Individual
from .genome import RGBXYGenome


class RGBXYGenomeEvaluator(Evaluator[RGBXYGenome, None]):
    def __init__(self):
        self.rng = Random(0)
        self.population = None

    def pre_select(self, population):
        self.population = list(population)
        return self.rng.choices(self.population, k=3)

    def evaluate(self, individuals: Iterable[Individual[RGBXYGenome, None]]) -> None:
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        preferences: Optional[list[float]] = None

        def onclick(event):
            nonlocal preferences
            dists = [math.sqrt((event.xdata - ind.genotype.x) ** 2
                               + (event.ydata - ind.genotype.y) ** 2)
                     for ind in individuals]
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
                        for ind in self.population if (g := ind.genotype)])
        ax.scatter(x=x, y=y, c=c)

        x, y, c = zip(*[(g.x, g.y, (g.r, g.g, g.b))
                        for ind in individuals if (g := ind.genotype)])
        ax.scatter(x=x, y=y, c=c)

        plt.show(block=True)
        if preferences is not None:
            for individual, preference in zip(individuals, preferences):
                individual.preference = preference
        else:
            warnings.warn("Error while getting human preferences: null list")
