import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from apets.hack.evaluator import MoveForwardFitness
from apets.hack.rl.callbacks import EvalCallback, TensorboardCallback

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain.dummy import BrainDummy
from revolve2.simulation.simulator import RecordSettings
from revolve2.standards import modular_robots_v2
from env import make, make_vec


TRAINERS = {
    "ppo": PPO
}

ROBOTS = [
    name.split("_")[0] for name, val in modular_robots_v2.__dict__.items()
    if callable(val) and "_" in name
]


def robot_body(name: Optional[str] = None):
    if name is None or name == "empty":
        return BodyV2()
    else:
        return modular_robots_v2.get(name)


def file_name(args):
    # folder = args.trainer.lower() + "_" + args.body
    # if args.rotated:
    #     folder += '45'
    # folder += f"_S{args.seed}"
    #
    # return Path(args.output_folder or "tmp/rl/").joinpath(folder).joinpath("model.zip")
    return Path(args.output_folder or "tmp/rl/").joinpath("model.zip")


def env_kwargs(args, _rerun, _render, _log, _backup):
    robot = ModularRobot(body=robot_body(args.body), brain=BrainDummy())
    reward_func = MoveForwardFitness(
        reward=args.reward,
        rerun=_rerun, render=_render,
        log_trajectory=_log, backup_trajectory=_backup,
        log_reward=_log,
    )
    return dict(
        rerun=_rerun,
        render=_render,
        robot=robot,
        rotated=args.rotated,
        reward_function=reward_func,
        simulation_time=args.simulation_time,
    )


def train(args, model_file):
    print("Training", model_file)
    n = args.threads or os.cpu_count()
    vec_env = make_vec(n=n,
                       **env_kwargs(
                           args, _rerun=False, _render=False, _log=False, _backup=False),
                       label=args.body + "-train",
                       vec_env_cls=SubprocVecEnv)
    test_env = make_vec(n=1,
                        **env_kwargs(
                            args, _rerun=True, _render=False, _log=True, _backup=True),
                        label=args.body + "-eval",
                        vec_env_cls=SubprocVecEnv)

    folder = model_file.parent

    budget = args.timesteps
    tb_callback = TensorboardCallback(
        log_trajectory_every=1,  # Eval callback (below)
        max_timestep=budget,
        multi_env=n > 1,
        args=vars(args)
    )
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=folder,
        log_path=folder,
        eval_freq=max(100, budget // (10 * n)),
        verbose=1,
        n_eval_episodes=1,
        deterministic=True,
        callback_after_eval=tb_callback,
    )

    # Define and Train the agent
    trainer = TRAINERS[args.trainer]
    model = trainer(
        "MlpPolicy", vec_env, device="cpu",
        # From https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml#L201
        # Based on the ant-v0

        normalize_advantage=True,
        n_steps=256,
        batch_size=32,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=2.5e-4,
    )

    model.set_logger(configure(str(folder), ["csv", "tensorboard"]))
    model.learn(
        total_timesteps=budget, progress_bar=True, callback=eval_callback,
    )

    tb_callback.log_step(final=True)
    model.save(model_file)

    args.simulation_time = 15
    args.rerun = ''
    args.movie = True

    rerun(args, model_file)

    return model


def rerun(args, model_file):
    print("Re-evaluating", model_file)
    render = not args.headless
    _env_kwargs = env_kwargs(
        args, _rerun=True, _render=render, _log=True, _backup=False)
    if args.movie:
        _env_kwargs["record_settings"] = RecordSettings(
            video_directory=model_file.parent,
            overwrite=True,
            fps=25,
            width=480, height=480,

            camera_id=2
        )

    env = make(**_env_kwargs, start_paused=render)
    check_env(env)

    model = PPO.load(model_file, device="cpu")

    model_time, steps = 0, 0

    obs, infos = env.reset()
    if render:
        env.render()

    while not env.done:
        model_start = time.time()
        action, _states = model.predict(obs, deterministic=True)
        model_time += time.time() - model_start
        steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()

    env.reward_function.do_plots(model_file.parent)

    print("Final reward:", env.cumulative_reward, "(truncated)" if env.truncated else "")

    modules = [m for m in model.policy.mlp_extractor.policy_net.modules() if isinstance(m, nn.Linear)]

    summary = {
        "arch": "mlp",
        "depth": len(modules),
        "width": modules[0].out_features if len(modules) > 0 else np.nan,
        "reward": args.reward,
        "run": args.seed,
        "body": args.body + ("45" if args.rotated else ""),
        "params": sum(p.numel() for p in model.policy.parameters()),
        "tps": args.simulation_time,
    }
    print(args.simulation_time, model_time)
    summary.update(env.infos())
    print(summary)
    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    print(summary)

    summary.to_csv(model_file.parent.joinpath("summary.csv"))

    # time.sleep(1)  # Ugly but prevents errors when quitting the window


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Generic")
    group.add_argument("-T", "--simulation-time", default=10, type=float)
    group.add_argument("-R", "--reward", required=True,
                       type=lambda _str: _str.lower(),
                       choices=[v.name.lower() for v in MoveForwardFitness.RewardType])

    group = parser.add_argument_group("Training")
    group.add_argument("--body", default="spider",
                       help=f"Morphology to use for the robot", choices=ROBOTS)
    group.add_argument("--trainer", default="ppo",
                       help="Name of the trainer to use", choices=TRAINERS.keys())
    group.add_argument("-r", "--rotated", default=False, action="store_true",
                       help="Whether the front of the robot is rotated by 45 degrees")
    group.add_argument("-t", "--timesteps", default=10_000, type=int,
                       help="Total number of (control) timesteps")
    group.add_argument("-s", "--seed", default=0, type=int,
                       help="Seed for the RNG Gods")
    group.add_argument("-o", "--output-folder", default=None, type=Path,
                       help="Where to store the model and associated data")
    group.add_argument("--threads", default=None, type=int,
                       help="Number of parallel environments to use")
    group.add_argument("--overwrite", default=False, action="store_true",)

    group.add_argument("--policy", choices=["mlp", "cpg"], default=None,
                       help="Policy architecture to use. See corresponding help sections for"
                            " parameters")

    group = parser.add_argument_group("MLP Policy")
    group.add_argument("--depth", type=int, help="Number of layers in the MLP")
    group.add_argument("--width", type=int, help="Number of neurons per layer")

    group = parser.add_argument_group("Evaluation")
    group.add_argument("--rerun", type=str, default=None,
                       const='', nargs='?')
    group.add_argument("--movie", default=False, action="store_true",)
    group.add_argument("--headless", default=False, action="store_true",)

    args = parser.parse_args()

    model_file = Path(args.rerun or file_name(args))
    _rerun = model_file.exists() and not args.overwrite

    log_folder = model_file.parent
    assert log_folder != os.getcwd(), "Not writing in root folder. Please provide a nested destination"
    if log_folder.exists():
        if args.overwrite:
            shutil.rmtree(log_folder)
        elif args.rerun is None:
            raise ValueError(f"Log folder '{log_folder}' already exists and overwriting was not requested")
    log_folder.mkdir(parents=True, exist_ok=True)

    if _rerun:
        rerun(args, model_file)

    else:
        train(args, model_file)


if __name__ == '__main__':
    main()
