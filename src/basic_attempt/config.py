"""Configuration parameters for this example."""
from dataclasses import dataclass
from typing import Annotated

from abrain.neat.config import Config as NEATConfig


@dataclass
class Config(NEATConfig):
    simulation_duration: Annotated[int, "Number of seconds per simulation"] = 5

    body_brain_mutation_ratio: (
        Annotated)[float, "Probability of mutating the body, otherwise brain"] = 0.1

    cppn_body_inputs: Annotated[str, "Inputs provided to the body's CPPN"] = "x,y,z,d"
    cppn_body_outputs: Annotated[str, "Outputs computed by the body's CPPN"] = "b,a"

