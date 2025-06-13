"""Genotype class."""
import logging
from copy import deepcopy

import json
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

from abrain import Genome as CPPNGenome
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain import Brain

from abrain.neat.evolver import Evolver
from apets.hack.body import DefaultBodyPlan, AABB
from apets.hack.brain import develop as develop_brain
from apets.hack.config import Config


@dataclass
class Genotype:
    """A genotype for a body and brain using CPPN."""

    body: CPPNGenome
    stem: CPPNGenome
    brain: CPPNGenome

    # TODO Single rng and GID manager
    # rng: Random
    # gid_manager: GIDManager

    @dataclass
    class Data:
        body: CPPNGenome.Data
        stem: CPPNGenome.Data
        brain: CPPNGenome.Data
        config: Config

        _mutation_weights: list[float]

        def __init__(self, config, seed=None):
            self.config = config
            self.body = CPPNGenome.Data.create_for_generic_cppn(
                inputs=4, outputs=["ssgm", "id"], labels="x,y,z,d,M,A",
                seed=seed,
                with_lineage=False
            )
            self.stem = CPPNGenome.Data.create_for_generic_cppn(
                inputs=6, outputs=["bsgm", "step", "step"],
                labels="x_0,y_0,z_0,x_1,y_1,z_1,W,LEO,CPG",
                with_input_bias=True,
                seed=seed,
                with_lineage=False
            )
            self.brain = CPPNGenome.Data.create_for_eshn_cppn(
                dimension=3,
                seed=seed,
                with_lineage=True
            )

            weights = [config.body_mutate_weight, config.stem_mutate_weight, config.brain_mutate_weight]
            self._mutation_weights = [w / sum(weights) for w in weights]

        @property
        def mutation_weights(self) -> list[float]: return self._mutation_weights

    @property
    def id(self): return self.brain.id

    @staticmethod
    def random(data: Data) -> 'Genotype':
        return Genotype(
            body=CPPNGenome.random(data.body),
            stem=CPPNGenome.random(data.stem),
            brain=CPPNGenome.random(data.brain)
        )

    def mutate(self, data: Data) -> None:
        i = data.body.rng.choices(range(3), data.mutation_weights, k=1)[0]
        if i == 0:
            self.body.mutate(data.body)
        elif i == 1:
            self.stem.mutate(data.stem)
        elif i == 2:
            self.brain.mutate(data.brain)
        else:
            raise RuntimeError(f"Invalid mutation type: n{i}")

    def mutated(self, data: Data) -> 'Genotype':
        clone = deepcopy(self)
        clone.mutate(data)
        clone.brain.update_lineage(data.body, [self.brain])
        return clone

    @staticmethod
    def crossover(lhs, rhs, data: Data) -> 'Genotype':
        body = CPPNGenome.crossover(lhs.body, rhs.body, data.body)
        stem = CPPNGenome.crossover(lhs.stem, rhs.stem, data.stem)
        brain = CPPNGenome.crossover(lhs.brain, rhs.brain, data.brain)
        brain.update_lineage(data.brain, [lhs.brain, rhs.brain])
        return Genotype(body, stem, brain)

    def develop(self, config: Config, with_labels=False, _id: int = 0) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :returns: The created robot.
        """

        body = self.develop_body(config)
        brain = self.develop_brain(body=body, config=config, with_labels=with_labels, _id=_id)
        return ModularRobot(body=body, brain=brain)

    def develop_body(self, config: Config) -> BodyV2:
        return DefaultBodyPlan.develop(self.body,
                                       camera=config.vision)

    def develop_brain(self, body: BodyV2, config: Config, with_labels=False, _id: int = 0) -> Brain:
        return develop_brain(body, self.stem, self.brain, config=config,
                             _id=_id, with_labels=with_labels)

    @staticmethod
    def distance(lhs: 'Genotype', rhs: 'Genotype') -> float:
        return (
            CPPNGenome.distance(lhs.body, rhs.body)
            + CPPNGenome.distance(lhs.stem, rhs.stem)
            + CPPNGenome.distance(lhs.brain, rhs.brain)
        ) / 3.

    @staticmethod
    def neat_interface(data: 'Genotype.Data') -> Evolver.Interface:
        return Evolver.Interface(
            g_class=Genotype,
            data=dict(data=data),
        )

    def to_json(self):
        return dict(
            body=self.body.to_json(),
            stem=self.stem.to_json(),
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
            if "genotype" in j:
                j_ = j["genotype"]
            elif "genome" in j:
                j_ = j["genome"]
            else:
                j_ = j
            genome = Genotype(
                CPPNGenome.from_json(j_.pop("body")),
                CPPNGenome.from_json(j_.pop("stem")),
                CPPNGenome.from_json(j_.pop("brain")),
            )

            return genome, j

    def print_json(self):
        pprint.pprint(self.to_json())

    def to_dot(self, path: Path, ext: str, data: Data):
        brain = path.parent.joinpath(f"{path.stem}_brain.{ext}")
        if self.brain.to_dot(data.brain, brain, ext, title="Brain"):
            logging.info(f"Rendered brain CPPN to {brain}")
        else:
            logging.error(f"Failed to render brain CPPN to {brain}")

        stem = path.parent.joinpath(f"{path.stem}_stem.{ext}")
        if self.stem.to_dot(data.stem, stem, ext, title="Stem"):
            logging.info(f"Rendered stem CPPN to {stem}")
        else:
            logging.error(f"Failed to render stem CPPN to {stem}")

        body = path.parent.joinpath(f"{path.stem}_body.{ext}")
        if self.body.to_dot(data.body, body, ext, title="Body"):
            logging.info(f"Rendered body CPPN to {body}")
        else:
            logging.error(f"Failed to render body CPPN to {body}")
