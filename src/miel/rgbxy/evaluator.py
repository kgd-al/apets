import math
from abc import ABC, abstractmethod
from typing import Iterable, Callable

import matplotlib
from matplotlib import pyplot as plt

from .genome import RGBXYGenome
from ..api import Evaluator, Individual

matplotlib.use('qtagg')


def _eq_dist(lhs, rhs):
    return math.sqrt(sum(
        (a - b)**2 for a, b in zip(lhs, rhs)
    ))


def _distance_based_preference(individuals, dist: Callable):
    assert len(individuals) > 1, f"Cannot make a choice with {len(individuals)} individuals"
    dists = [dist(ind) for ind in individuals]
    ranks = [
        (j, *t) for j, t in enumerate(
            sorted([(i, d) for i, d in enumerate(dists)],
                   key=lambda _t: _t[1])
        )
    ]
    ranks = [t[0] for t in sorted(ranks, key=lambda _t: _t[1])]

    preferences = [round(-2 * r / (len(ranks) - 1) + 1) for r in ranks]
    for individual, preference in zip(individuals, preferences):
        individual.preference = preference


class ScatterEval(Evaluator[RGBXYGenome, None], ABC):
    def __init__(self, headless):
        super().__init__()

        self.headless = headless
        self.stage = 0

        fig, ax = plt.subplots()
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')

        self.fig, self.ax = fig, ax
        self.plots = [self.ax.scatter([], [])]
        self.fig.suptitle(f"Initialisation")

        if not headless:
            plt.show(block=False)
            plt.pause(.01)

    def plot(self, individuals: Iterable[Individual[RGBXYGenome, None]], kwargs, stage):
        if stage != 2:
            for plot in self.plots:
                plot.remove()
            self.plots = []

        if (stage & 1) != 0:
            if (population := kwargs.get("population", None)) is not None:
                x, y, c = zip(*[(g.x, g.y, (g.r, g.g, g.b, .1))
                                for ind in population if (g := ind.genome)])
                self.plots.append(self.ax.scatter(x=x, y=y, c=c))

            x, y, c = zip(*[(g.x, g.y, (g.r, g.g, g.b))
                            for ind in individuals if (g := ind.genotype)])
            self.plots.append(self.ax.scatter(x=x, y=y, c=c))

            self._subplots()

            self.fig.suptitle(f"Stage {self.stage}: {self._label()}")

        if (stage & 2) != 0:
            for ind in individuals:
                if (p := ind.preference) != 0:
                    self.plots.append(self.ax.annotate(
                        f"{p:+g}", (ind.genotype.x, ind.genotype.y),
                    ))

        if not self.headless:
            self.fig.canvas.draw()
            plt.pause(.01)

        if (folder := kwargs.get("odir", None)) is not None and ((stage & 2) != 0):
            self.fig.savefig(folder.joinpath(f"stage_{self.stage}.png"))

        if (stage & 1) != 0:
            self.stage += 1

    def _subplots(self):
        pass

    @abstractmethod
    def _label(self): pass


class RGBXYAutoEvaluator(ScatterEval):
    def __init__(self, target: RGBXYGenome):
        super().__init__(headless=True)
        assert isinstance(target, RGBXYGenome)
        self.target = target

    def dist(self, ind):
        return _eq_dist(ind.genotype, self.target)

    def _label(self):
        return (f"Target: {(self.target.x, self.target.y)}#"
                + "".join(
                    f"{int(255*v):02X}"
                    for v in [self.target.r, self.target.g, self.target.b]))

    def _subplots(self):
        self.plots.append(
            self.ax.scatter([self.target.x], [self.target.y],
                            c=[(self.target.r, self.target.g, self.target.b)],
                            edgecolors='black',
                            marker='X')
        )

    def evaluate(self, individuals: Iterable[Individual], kwargs):
        _distance_based_preference(individuals, self.dist)
        self.plot(individuals, kwargs, stage=3)


class RGBXYGenomeEvaluator(ScatterEval):
    def __init__(self):
        super().__init__(headless=False)

    def _label(self): return "Human"

    def evaluate(self,
                 individuals: Iterable[Individual[RGBXYGenome, None]],
                 kwargs) -> None:

        self.plot(individuals, kwargs, stage=1)

        def onclick(event):
            _distance_based_preference(
                individuals,
                lambda ind: _eq_dist((ind.genotype.x, ind.genotype.y),
                                     (event.xdata, event.ydata))
            )
            self.fig.canvas.stop_event_loop()

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.fig.canvas.start_event_loop(timeout=-1)

        self.plot(individuals, kwargs, stage=2)
