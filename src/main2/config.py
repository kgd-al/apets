"""Configuration parameters for this example."""
from dataclasses import dataclass
from enum import auto, Enum
from typing import Annotated, Optional, Tuple

from strenum import UppercaseStrEnum

from abrain.neat.config import Config as NEATConfig


class TaskType(Enum):
    MOVE_STOP = auto()
    STOP_MOVE = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()


@dataclass
class Config(NEATConfig):
    simulation_duration: Annotated[int, "Number of seconds per simulation"] = 5

    initial_distance_threshold: float = 1.5  # Overridden from NEATConfig

    body_mutate_weight: Annotated[float, "Weight for body mutations"] = 1
    stem_mutate_weight: Annotated[float, "Weight for stem mutations"] = 9
    brain_mutate_weight: Annotated[float, "Weight for brain mutations"] = 10

    # cppn_body_inputs: Annotated[str, "Inputs provided to the body's CPPN"] = "x,y,z,d"
    # cppn_body_outputs: Annotated[str, "Outputs computed by the body's CPPN"] = "b,a"

    vision: Annotated[Optional[tuple[int, int]],
                      ("Resolution of the front-facing camera "
                       "or None for no vision")] = None
