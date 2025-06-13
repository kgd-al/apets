import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from apets.hack.evaluator import MoveForwardFitness
from apets.hack.rl.callbacks import EvalCallback, TensorboardCallback

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain.dummy import BrainDummy
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
    folder = args.trainer.lower() + "_" + args.body
    if args.rotated:
        folder += '45'
    folder += f"_S{args.seed}"

    return Path(args.output_folder or "tmp/rl/").joinpath(folder).joinpath("model.zip")


def train(args, model_file, env_kwargs):
    print("Training", model_file)
    n = args.threads or os.cpu_count()
    vec_env = make_vec(n=n,
                       **env_kwargs,
                       vec_env_cls=SubprocVecEnv)

    folder = model_file.parent

    budget = args.timesteps
    tb_callback = TensorboardCallback(
        log_trajectory_every=1,  # Eval callback (below)
        max_timestep=budget,
        multi_env=n > 1
    )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=folder,
        log_path=folder,
        eval_freq=max(100, budget // (10 * n)),
        verbose=1,
        n_eval_episodes=1,
        callback_after_eval=tb_callback,
    )

    # Define and Train the agent
    trainer = TRAINERS[args.trainer]
    model = trainer("MlpPolicy", vec_env, device="cpu")

    model.set_logger(configure(str(folder), ["csv", "tensorboard"]))
    model.learn(total_timesteps=budget, progress_bar=True, callback=eval_callback)

    model.save(model_file)

    return model


def rerun(args, model_file, reward_fn, env_kwargs):
    print("Re-evaluating", model_file)
    env = make(rerun=True, **env_kwargs, start_paused=True)
    check_env(env)

    model = PPO.load(model_file, device="cpu")
    print(model.policy)

    obs, infos = env.reset()
    env.render()
    while not env.done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

    reward_fn.do_plots()

    print("Final reward:", env.cumulative_reward, "(truncated)" if env.truncated else "")

    # time.sleep(1)  # Ugly but prevents errors when quitting the window


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Generic")
    group.add_argument("-T", "--simulation-time", default=10, type=float)

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

    group = parser.add_argument_group("Evaluation")
    group.add_argument("--rerun", type=Path, default=None)

    args = parser.parse_args()

    rotated = args.rotated

    model_file = args.rerun or file_name(args)
    _rerun = model_file.exists() and not args.overwrite

    log_folder = model_file.parent
    assert log_folder != os.getcwd(), "Not writing in root folder. Please provide a nested destination"
    if log_folder.exists():
        if args.overwrite:
            shutil.rmtree(log_folder)
        else:
            raise ValueError("Log folder already exists and overwriting was not requested")
    log_folder.mkdir(parents=True, exist_ok=True)

    robot = ModularRobot(body=robot_body(args.body), brain=BrainDummy())
    reward_func = MoveForwardFitness(robot, render=_rerun, rerun=_rerun, log_folder=log_folder)
    env_kwargs = dict(
        robot=robot,
        rotated=rotated,
        reward_function=reward_func,
        simulation_time=args.simulation_time,
    )

    if _rerun:
        rerun(args, model_file, reward_func, env_kwargs)

    else:
        train(args, model_file, env_kwargs)


if __name__ == '__main__':
    main()
