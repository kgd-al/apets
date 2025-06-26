""" Contains an out-of-the-box exemple of verbose callback relying on
Tensorboard.

Provided as-is without *any* guarantee of functionality or fitness for a
particular purpose
"""

import logging
import math
import pprint
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Mapping, Any

import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import (
    Image,
    HParam,
    TensorBoardOutputFormat, Figure,
)
from stable_baselines3.common.vec_env.base_vec_env import tile_images


logger = logging.getLogger(__name__)


def _recurse_avg_dict(dicts: List[Dict], root_key=""):
    avg_dict = defaultdict(list)
    for d in dicts:
        for k, v in _recurse_dict(d, root_key):
            avg_dict[k].append(v)
    return {k: np.average(v) for k, v in avg_dict.items()}


def _recurse_dict(dct, root_key):
    for k, v in dct.items():
        current_key = f"{root_key}/{k}" if root_key else k
        if isinstance(v, dict):
            for k_, v_ in _recurse_dict(v, current_key):
                yield k_, v_
        else:
            yield current_key, v


def maybe_convert(v):
    if isinstance(v, Path):
        return str(v)
    else:
        return v


class TensorboardCallback(BaseCallback):
    def __init__(
        self,
        log_trajectory_every: int = 0,
        verbose: int = 0,
        max_timestep: int = 0,
        prefix: str = "",
        multi_env: bool = False,
        args: Mapping[str, Any] = None,
    ):
        super().__init__(verbose=verbose)
        self.args = args

        self.log_trajectory_every = log_trajectory_every
        fmt = (
            "{:d}"
            if max_timestep == 0
            else "{:0" + str(math.ceil(math.log10(max_timestep - 1))) + "d}"
        )
        if prefix:
            if not prefix.endswith("_"):
                prefix += "_"
            fmt = prefix + fmt
        self.prefix = prefix
        self.img_format = fmt
        self.multi_env = multi_env

        self.last_stats: Optional = None

    @staticmethod
    def _rewards(env):
        dd_rewards = defaultdict(list)
        for d in [e.__dict__ for e in env.env_method("atomic_rewards")]:
            for key, value in d.items():
                dd_rewards[key].append(value)
        reward_strings = []
        for k, dl in dd_rewards.items():
            a, s = np.average(dl), np.std(dl)
            rs = f"{k[0]}={a:g}"
            if s != 0:
                rs += f"+/-{s:g}"
            reward_strings.append(rs)
        return " ".join(reward_strings)

    def _on_training_start(self) -> None:
        assert isinstance(self.parent, EvalCallback)

        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

        writer = self.tb_formatter.writer

        if self.num_timesteps > 0:
            return

        params = set(self.training_env.get_attr("parameters"))
        assert len(params) == 1, f"Non-uniform parameters:\n{pprint.pformat(params)}"
        params = next(iter(params))

        policy = self.model.policy
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "policy": policy.__class__.__name__,
            "learning rate": self.model.learning_rate,
            # "body": params.label,
            "train_envs": self.training_env.num_envs,
            "eval_envs": self.parent.eval_env.num_envs,
            "duration": params.simulation_time,
        }
        if self.args is not None:
            hparam_dict.update({f"config/{k}": maybe_convert(v) for k, v in self.args.items()})

        metric_dict = {
            f"eval/{k}": v for k, v in
            self.parent.eval_env.env_method("infos")[0].items()
        }

        if isinstance(self.model, PPO):
            hparam_dict.update({
                "param/" + attr: getattr(self.model, attr)
                for attr in [
                    "n_steps", "batch_size", "gae_lambda", "gamma"
                ]
            })

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

        logger.info(f"Policy: {policy}")
        folder = Path(self.logger.dir)
        folder.mkdir(exist_ok=True)
        with open(folder.joinpath("policy.str"), "w") as f:
            f.write(str(policy) + "\n")

        writer.add_text(
            "policy", str(policy).replace("\n", "<br/>").replace(" ", "&nbsp;")
        )

    def _on_step(self) -> bool:
        self.log_step(False)
        return True

    def _print_trajectory(self, env, key, name):
        images = env.env_method("plot_trajectory_as_image")

        big_image = tile_images(images)

        self.logger.record(
            f"infos/{key}_traj",
            Image(big_image, "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )
        folder = Path(self.logger.dir).joinpath("trajectories")
        folder.mkdir(exist_ok=True)
        pil_img = PIL.Image.fromarray(big_image)
        pil_img.save(folder.joinpath(f"{key}_{name}.png"))

    def _plot_trajectory(self, env):
        assert env.num_envs == 1
        figure = env.env_method("plot_trajectory")[0]
        self.logger.record("infos/traj", Figure(figure, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        plt.close(figure)

    def log_step(self, final: bool):
        assert isinstance(self.parent, EvalCallback)
        env = self.parent.eval_env

        assert env.num_envs == 1
        for key, value in next(iter(env.env_method("previous_infos"))).items():
            self.logger.record_mean(f"eval/{key}", value)

        print_trajectory = final or (
            self.log_trajectory_every > 0
            and (self.n_calls % self.log_trajectory_every) == 0
        )

        if print_trajectory:
            self._plot_trajectory(env)

        self.logger.dump(self.model.num_timesteps)

        if final:
            self.logger.close()
