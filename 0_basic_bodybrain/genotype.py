"""Genotype class."""

from dataclasses import dataclass
from random import Random

import multineat
import numpy as np
from abrain.core.genome import GIDManager

from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeV2
from revolve2.modular_robot import ModularRobot
from abrain import Genome as BodyOrBrainGenome, ANN3D
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain import BrainInstance, Brain
from body import develop as develop_body
from brain import develop as develop_brain


@dataclass
class Genotype:
    """A genotype for a body and brain using CPPN."""

    body: BodyOrBrainGenome
    brain: BodyOrBrainGenome

    # TODO Single rng and GID manager
    # rng: Random
    # gid_manager: GIDManager

    @dataclass
    class Data:
        body: BodyOrBrainGenome.Data
        brain: BodyOrBrainGenome.Data

        def __init__(self, seed=None):
            self.body = BodyOrBrainGenome.Data.create_for_generic_cppn(
                inputs=4, outputs=["bsgm", "id"],
                seed=seed
            )
            self.brain = BodyOrBrainGenome.Data.create_for_eshn_cppn(
                dimension=3,
                seed=seed
            )

    @classmethod
    def random(cls, data: Data) -> 'Genotype':
        return Genotype(
            body=BodyOrBrainGenome.random(data.body),
            brain=BodyOrBrainGenome.random(data.brain)
        )

    def mutated(self, data: Data) -> 'Genotype':
        return Genotype(
            body=self.body.mutated(data.body),
            brain=self.brain.mutated(data.brain)
        )

    @classmethod
    def crossover(cls, lhs, rhs, data: Data) -> 'Genotype':
        body = BodyOrBrainGenome.crossover(lhs.body, rhs.body, data.body)
        brain = BodyOrBrainGenome.crossover(lhs.brain, rhs.brain, data.brain)
        return Genotype(body, brain)

    def develop(self) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """

        body = self.develop_body()
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)

    def develop_body(self) -> BodyV2:
        return develop_body(self.body)

    def develop_brain(self, body: BodyV2) -> Brain:
        return develop_brain(self.brain, body)
