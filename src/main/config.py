"""Configuration parameters for this example."""
from dataclasses import dataclass
from enum import auto
from typing import Annotated, Optional, Tuple

from strenum import UppercaseStrEnum

from abrain.neat.config import Config as NEATConfig


class ExperimentType(UppercaseStrEnum):
    LOCOMOTION = auto()
    FOLLOW = auto()         #! Not implemented
    PUNCH_ONCE = auto()
    PUNCH_AHEAD = auto()
    PUNCH_BACK = auto()
    PUNCH_THRICE = auto()
    PUNCH_TOGETHER = auto()
    PUNCH_FOLLOW = auto()   #! Not implemented


EXPERIMENT_DURATIONS = {
    ExperimentType.LOCOMOTION: 5,
    ExperimentType.PUNCH_ONCE: 5,
    ExperimentType.PUNCH_AHEAD: 5,
    ExperimentType.PUNCH_BACK: 10,
    ExperimentType.PUNCH_THRICE: 20,
    ExperimentType.PUNCH_TOGETHER: 20,
    ExperimentType.FOLLOW: 20,
    ExperimentType.PUNCH_FOLLOW: 20,
}


@dataclass
class Config(NEATConfig):
    experiment: Annotated[ExperimentType, "Experiment to perform"] = None
    centered_ball: Annotated[bool, "Whether the ball is facing the robot"] = False
    simulation_duration: Annotated[Optional[int],
                                   ("Number of seconds per simulation"
                                    " (defaults to experiment-specific value)")] = None

    initial_distance_threshold: float = 1.5  # Overridden from NEATConfig

    body_brain_mutation_ratio: (
        Annotated)[float, "Probability of mutating the body, otherwise brain"] = 0.1

    cppn_body_inputs: Annotated[str, "Inputs provided to the body's CPPN"] = "x,y,z,d"
    cppn_body_outputs: Annotated[str, "Outputs computed by the body's CPPN"] = "b,a"

    vision: Annotated[Optional[tuple[int, int]],
                      ("Resolution of the front-facing camera "
                       "or None for no vision")] = None
