"""Genotype class."""
from copy import deepcopy

import json
import pprint
from dataclasses import dataclass
from typing import Tuple

from abrain import Genome as BodyOrBrainGenome
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain import Brain

from abrain.neat.evolver import Evolver
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
                seed=seed,
                with_lineage=False
            )
            self.brain = BodyOrBrainGenome.Data.create_for_eshn_cppn(
                dimension=3,
                seed=seed,
                with_lineage=True
            )

    @property
    def id(self): return self.brain.id

    @staticmethod
    def random(data: Data) -> 'Genotype':
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
        clone.brain.update_lineage(data.body, [self.brain])
        return clone

    @staticmethod
    def crossover(lhs, rhs, data: Data) -> 'Genotype':
        body = BodyOrBrainGenome.crossover(lhs.body, rhs.body, data.body)
        brain = BodyOrBrainGenome.crossover(lhs.brain, rhs.brain, data.brain)
        brain.update_lineage(data.brain, [lhs.brain, rhs.brain])
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

    @staticmethod
    def distance(lhs: 'Genotype', rhs: 'Genotype') -> float:
        return (.5 * BodyOrBrainGenome.distance(lhs.body, rhs.body)
                + .5 * BodyOrBrainGenome.distance(lhs.brain, rhs.brain))

    @staticmethod
    def neat_interface(data: 'Genotype.Data') -> Evolver.Interface:
        return Evolver.Interface(
            g_class=Genotype,
            data=dict(data=data),
        )

    def to_json(self):
        return dict(
            body=self.body.to_json(),
            brain=self.brain.to_json()
        )

    def to_file(self, path, data: dict = None):
        with open(path, "wt") as f:
            dct = self.to_json()
            if data is not None:
                dct.update(data)
            json.dump(dct, f)

    @staticmethod
    def from_file(path) -> Tuple['Genotype', dict]:
        with open(path, "rt") as f:
            j = json.load(f)
            genome = Genotype(
                BodyOrBrainGenome.from_json(j.pop("body")),
                BodyOrBrainGenome.from_json(j.pop("brain")),
            )

            return genome, j

    def print_json(self):
        pprint.pprint(self.to_json())
