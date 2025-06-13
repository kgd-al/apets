import math
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import ClassVar


@dataclass
class RGBXYGenome:
    a: ClassVar[float] = .2
    s: ClassVar[float] = .1
    fields: ClassVar[str] = "rgbxy"

    r: float = 0
    g: float = 0
    b: float = 0
    x: float = 0
    y: float = 0

    @staticmethod
    def from_string(s: str):
        return RGBXYGenome(*[float(x) for x in s.replace(",", " ").split()])

    def __str__(self):
        return (
            f"{(self.x, self.y)}#"
            + "".join(
                f"{int(255*v):02X}"
                for v in [self.r, self.g, self.b])
        )

    def __iter__(self):
        return iter([self.r, self.g, self.b, self.x, self.y])

    @staticmethod
    def __range(f): return -1 if f in "xy" else 0, 1

    @classmethod
    def random(cls, rng: Random):
        return RGBXYGenome(*[rng.uniform(*cls.__range(f)) * cls.a for f in cls.fields])

    def mutate(self, rng: Random):
        n = rng.randint(1, len(self.fields))
        fields = rng.choices(self.fields, k=n)
        for f in fields:
            setattr(self, f,
                    max(
                        self.__range(f)[0],
                        min(
                            getattr(self, f) + rng.gauss(0, self.s / n),
                            1
                        )))

    # def mutate(self, rng: Random):
    #     f = rng.choice("rgbxy")
    #     setattr(self, f,
    #             max(
    #                 self.range(f)[0],
    #                 min(
    #                     getattr(self, f) + rng.gauss(0, self.s),
    #                     1
    #                 )))

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

    @staticmethod
    def features(ind: 'RGBXYGenome'):
        return ind.x, ind.y

    @classmethod
    @lru_cache
    def fitness_range(cls):
        return (
            cls.evaluate(RGBXYGenome(1, .749, 0, -1, -1)),
            cls.evaluate(RGBXYGenome(1, 0, 0, 1, 0))
        )

    @classmethod
    def features_range(cls):
        return (-1, 1), (-1, 1)


