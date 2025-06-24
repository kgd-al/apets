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
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here,
        # should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

        # print("[kgd-debug]", self.training_env.env_method("infos"))
        # print("[kgd-debug]", self.training_env.get_attr("reward"))

        writer = self.tb_formatter.writer
        # if self.multi_env:
        #     writer.add_text(
        #         "train/rewards",
        #         self.prefix + ": " + self._rewards(self.training_env),
        #     )
        #     writer.add_text(
        #         "eval/rewards",
        #         self.prefix + ":" + self._rewards(self.parent.eval_env),
        #     )

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
        # if not self.multi_env:
        #     hparam_dict["rewards"] = self._rewards(self.training_env)

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

        print(hparam_dict)
        # for k, v in hparam_dict.items():
        #     assert any(isinstance(v, t) for t in [int, float, str, bool]), f"{k}: {v} ({type(v)})"

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

        # dummy_inputs = \
        #     policy.obs_to_tensor(policy.observation_space.sample())[0]
        # writer.add_graph(policy, dummy_inputs, use_strict_trace=False)
        #
        # graph = to_dot(policy)
        # graph.render(folder.joinpath("policy"), format='pdf', cleanup=True)
        # graph.render(folder.joinpath("policy"), format='png', cleanup=True)
        # # noinspection PyTypeChecker
        # writer.add_image(
        #     "policy",
        #     np.asarray(PIL.Image.open(BytesIO(graph.pipe(format='jpg')))),
        #     dataformats="HWC", global_step=0)

    def _on_step(self) -> bool:
        self.log_step(False)
        return True

    def _print_trajectory(self, env, key, name):
        images = env.env_method(
            "plot_trajectory_as_image"#, verbose=True, cb_side=0, square=True
        )

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
        # logger.info(
        #     f"[kgd-debug] Logging tensorboard data at time"
        #     f" {self.num_timesteps} {self.model.num_timesteps} ({final=})")

        assert isinstance(self.parent, EvalCallback)
        env = self.parent.eval_env

        # eval_infos = env_attr(env, "last_infos")
        # for key, value in _recurse_avg_dict(eval_infos, "infos").items():
        #     self.logger.record_mean(key, value)
        assert env.num_envs == 1
        for key, value in next(iter(env.env_method("infos"))).items():
            self.logger.record_mean(f"eval/{key}", value)

        print_trajectory = final or (
            self.log_trajectory_every > 0
            and (self.n_calls % self.log_trajectory_every) == 0
        )

        if print_trajectory:
            t_str = "final" if final else self.img_format.format(self.num_timesteps)
            # self._print_trajectory(env, "eval", t_str)
            self._plot_trajectory(env)

        self.logger.dump(self.model.num_timesteps)

        # if final:
        #     train_env = self.training_env
        #     env_method(train_env, "log_trajectory", True)
        #
        #     logger.info("Final log step. Storing performance on training env")
        #     r = evaluate_policy(model=self.model, env=train_env)
        #
        #     env_method(train_env, "log_trajectory", False)
        #
        #     t_str = "final" if final else self.img_format.format(self.num_timesteps)
        #     self._print_trajectory(train_env, "train", t_str)
        #
        #     eval_infos = _recurse_avg_dict(eval_infos, "eval")
        #     train_infos = _recurse_avg_dict(env_attr(train_env, "last_infos"), "train")
        #
        #     self.last_stats = {
        #         "train/reward": np.average(r),
        #         "eval/reward": self.parent.best_mean_reward,
        #     }
        #     self.last_stats.update(eval_infos)
        #     self.last_stats.update(train_infos)

        if final:
            self.logger.close()
