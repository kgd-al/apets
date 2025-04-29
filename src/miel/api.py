from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Iterable, Optional, Tuple, Any, Dict, get_args

Genotype = TypeVar("Genotype")
Phenotype = TypeVar("Phenotype")
EA = TypeVar("EA")


@dataclass
class Individual(Generic[Genotype, Phenotype]):
    genotype: Genotype
    phenotype: Optional[Phenotype]

    fitness: list[float] = field(default=None)
    stats: dict[str, float] = field(default=None)
    preference: float = field(default=0)


class Evolver(ABC, Generic[Genotype, Phenotype, EA]):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, n) -> None:
        """Let evolution run "freely" for n steps

        TODO N steps is n generations? what about map elite? or other steady states
        """
        pass

    @abstractmethod
    def select(self) -> Tuple[Iterable[Individual[Genotype, Phenotype]], Dict[str, Any]]:
        pass

    @abstractmethod
    def feedback(self, individuals: Iterable[Individual[Genotype, Phenotype]]):
        """Use human preferences to bias evolution their way"""
        pass


class Evaluator(ABC, Generic[Genotype, Phenotype]):
    def __init_subclass__(cls) -> None:
        evaluator_base = [
            c for c in cls.__orig_bases__
            if (o := getattr(c, '__origin__', None)) and o is Evaluator
        ]
        assert len(evaluator_base) == 1, f"Error getting evaluator 'template' arguments"
        cls.GENOTYPE, cls.PHENOTYPE = get_args(evaluator_base[0])
        # print(f"Evaluator.__init_subclass__({cls}): {cls.GENOTYPE=}, {cls.PHENOTYPE=}")

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, individuals: Iterable[Individual[Genotype, Phenotype]], kwargs) -> None:
        """
        Proceed to (human) evaluation of preferences and stores them in the appropriate field

        :param individuals: individuals to show to the user
        :param kwargs: additional arguments to use (as returned by :meth:`~Evolver.select`)
        """
        pass
