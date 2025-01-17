"""Configuration parameters for this example."""
import ast
from abc import ABC
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import get_args, get_origin, Union, Annotated, Optional

from abrain.neat.evolver import NEATConfig


@dataclass
class ConfigBase(ABC):

    @classmethod
    def __fields(cls):
        return [field for field in fields(cls) if get_origin(field.type) is Annotated]

    @classmethod
    def populate_argparser(cls, parser):
        for field in cls.__fields():
            a_type = field.type.__args__[0]
            t_args = get_args(a_type)
            f_type, str_type = a_type, None
            default = field.default
            action = "store"

            if get_origin(a_type) is Union and type(None) in t_args:
                f_type = t_args[0]
            elif a_type == bool:
                f_type = ast.literal_eval
                str_type = bool

            if not str_type:
                str_type = f_type

            assert str_type, (
                f"Invalid user type {str_type} " f"(from {a_type=} {f_type=}"
            )

            help_msg = (
                f"{'.'.join(field.type.__metadata__)}"
                f" (default: {default},"
                f" type: {str_type.__name__})"
            )
            parser.add_argument(
                f"--{field.name}".replace("_", "-"),
                action=action,
                dest=f"{field.name}",
                default=default,
                metavar="V",
                type=f_type,
                help=help_msg,
            )

    @classmethod
    def from_argparse(cls, namespace):
        data = cls()
        for field in cls.__fields():
            f_name = f"{field.name}"
            attr = None
            if (
                    hasattr(namespace, f_name)
                    and (maybe_attr := getattr(namespace, f_name)) is not None
            ):
                attr = maybe_attr
            if attr is not None:
                setattr(data, field.name, attr)
        if post_init := getattr(data, "_post_init", None):  # pragma: no branch
            post_init(allow_unset=True)
        return data


@dataclass
class Config(ConfigBase):
    threads: Annotated[int, "Number of concurrent evaluations"] = None
    overwrite: Annotated[bool, "Do we allow running in an existing folder?"] = False

    simulation_duration: Annotated[float, "Number of seconds per simulation"] = 5

    body_brain_mutation_ratio: (
        Annotated)[float, "Probability of mutating the body, otherwise brain"] = 0.1

    seed: Annotated[Optional[int], "RNG seed (set from time if none)"] = None

    population_size: Annotated[int, "Population size (duh)"] = 10
    generations: Annotated[int, "Number of generations (double duh)"] = 10
    species: Annotated[int, "Target number of species (for NEAT diversity)"] = 8
    distance_threshold: Annotated[float, "Initial genetic distance to differentiate between species"] = .1

    data_root: Annotated[Optional[Path], "Where to store the generated data"] = None

    cppn_body_inputs: Annotated[str, "Inputs provided to the body's CPPN"] = "x,y,z,d"
    cppn_body_outputs: Annotated[str, "Outputs computed by the body's CPPN"] = "b,a"

    neat: NEATConfig = NEATConfig()
