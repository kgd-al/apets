"""Genotype class."""
from copy import deepcopy

import json
import pprint
from dataclasses import dataclass

from abrain import Genome as BodyOrBrainGenome
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain import Brain

from body import DefaultBodyPlan
from brain import develop as develop_brain
from config import Config


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
        config: Config

        def __init__(self, config, seed=None):
            self.config = config
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

    def mutate(self, data: Data) -> None:
        if data.body.rng.random() < data.config.body_brain_mutation_ratio:
            self.body.mutate(data.body)
        else:
            self.brain.mutate(data.brain)

    def mutated(self, data: Data) -> 'Genotype':
        clone = deepcopy(self)
        clone.mutate(data)
        return clone

    @classmethod
    def crossover(cls, lhs, rhs, data: Data) -> 'Genotype':
        body = BodyOrBrainGenome.crossover(lhs.body, rhs.body, data.body)
        brain = BodyOrBrainGenome.crossover(lhs.brain, rhs.brain, data.brain)
        return Genotype(body, brain)

    def develop(self, config: Config) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """

        body = self.develop_body(config)
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)

    def develop_body(self, config: Config) -> BodyV2:
        return DefaultBodyPlan.develop(self.body,
                                       inputs=config.cppn_body_inputs,
                                       outputs=config.cppn_body_outputs)

    def develop_brain(self, body: BodyV2) -> Brain:
        return develop_brain(self.brain, body)

    @classmethod
    def distance(cls, lhs: 'Genotype', rhs: 'Genotype') -> float:
        return (.5 * BodyOrBrainGenome.distance(lhs.body, rhs.body)
                + .5 * BodyOrBrainGenome.distance(lhs.brain, rhs.brain))

    def print_json(self):
        pprint.pprint(dict(
            body=self.body.to_json(),
            brain=self.brain.to_json()
        ))
