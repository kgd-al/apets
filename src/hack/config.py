"""Configuration parameters for this example."""
from dataclasses import dataclass
from enum import auto, Enum
from typing import Annotated, Optional, Tuple, ClassVar

from strenum import UppercaseStrEnum

from abrain import Config as ABrainConfig
from abrain.neat.config import Config as NEATConfig, ConfigBase


class TaskType(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()


@dataclass
class Config(NEATConfig):
    experiment: Annotated[str, "Description of the experiment"] = (
        "CPG - ANN Hybrid using sensory CPGs\n"
        "CPGs follow the traditional topology (2-neighborhood)"
        "ANN is not connected, control signals come from the evaluator, all hinges are cpgs"
        "Morphological evolution is on"
    )
    experiment_version: Annotated[str, "Shorthand for the incremental experiment version"] = "0.0.0"

    monotask: Annotated[bool, "Whether to evolve with individual tasks (move, rotate...) or just move"] = True

    scale_hinges: Annotated[bool, "Whether to scale hinges coordinates or use raw values"] = True

    simulation_duration: Annotated[int, "Number of seconds per simulation"] = 5

    initial_distance_threshold: float = 1.5  # Overridden from NEATConfig

    body_mutate_weight: Annotated[float, "Weight for body mutations"] = 9#1
    stem_mutate_weight: Annotated[float, "Weight for stem mutations"] = 9
    brain_mutate_weight: Annotated[float, "Weight for brain mutations"] = 0#10

    # cppn_body_inputs: Annotated[str, "Inputs provided to the body's CPPN"] = "x,y,z,d"
    # cppn_body_outputs: Annotated[str, "Outputs computed by the body's CPPN"] = "b,a"

    vision: Annotated[Optional[tuple[int, int]],
                      ("Resolution of the front-facing camera "
                       "or None for no vision")] = None

    generate_plots: Annotated[bool, "Auto-generate fitness and species plots"] = True
    generate_movie: Annotated[bool, "Auto-generate movie of champion's behavior"] = True

    abrain_config: ClassVar[ABrainConfig]  # TODO: Needs rework in abrain. Static fields are bad
