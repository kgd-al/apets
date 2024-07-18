"""Genotype class."""

from dataclasses import dataclass

from abrain import Genome as BodyOrBrainGenome
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain import Brain

from body import DefaultBodyPlan
from brain import develop as develop_brain
from config import BODY_BRAIN_MUTATION_RATIO


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
        if data.body.rng.random() < BODY_BRAIN_MUTATION_RATIO:
            return Genotype(
                body=self.body.mutated(data.body),
                brain=self.brain.copy()
            )
        else:
            return Genotype(
                body=self.body.copy(),
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
        return DefaultBodyPlan.develop(self.body)

    def develop_brain(self, body: BodyV2) -> Brain:
        return develop_brain(self.brain, body)
