import math
from dataclasses import dataclass
from random import Random
from typing import ClassVar

from abrain.neat.evolver import EvaluationResult


@dataclass
class RGBXYGenome:
    a: ClassVar[float] = 0
    s: ClassVar[float] = .01

    r: float = 0
    g: float = 0
    b: float = 0
    x: float = 0
    y: float = 0

    def __iter__(self):
        return iter([self.r, self.g, self.b, self.x, self.y])

    @staticmethod
    def range(f): return -1 if f in "xy" else 0, 1

    @classmethod
    def random(cls, rng: Random):
        return RGBXYGenome(*[rng.uniform(*cls.range(f)) * cls.a for f in "rgbxy"])

    def mutate(self, rng: Random):
        f = rng.choice("rgbxy")
        setattr(self, f,
                max(
                    self.range(f)[0],
                    min(
                        getattr(self, f) + rng.gauss(0, self.s),
                        1
                    )))

    @staticmethod
    def crossover(lhs: 'RGBXYGenome', rhs: 'RGBXYGenome', rng: Random):
        return RGBXYGenome(*[rng.choice([a, b]) for a, b in zip(lhs, rhs)])

    @staticmethod
    def distance(lhs: 'RGBXYGenome', rhs: 'RGBXYGenome'):
        return math.sqrt(sum((a - b)**2 for a, b in zip(rhs, lhs)))

    @staticmethod
    def evaluate(ind: 'RGBXYGenome') -> float:
        r, g, b = ind.r, ind.g, ind.b
        a = math.atan2(math.sqrt(3) * (g - b) / 2, .5 * (2*r - g - b))
        l = max(r, g, b)

        x, y = math.cos(a), math.sin(a)
        res = -math.sqrt((ind.x - x) ** 2 + (ind.y - y) ** 2) + l - 1
        return res
