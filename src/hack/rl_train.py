import argparse
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from evaluator import MoveForwardFitness
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.v2 import BodyV2
from revolve2.modular_robot.brain.dummy import BrainDummy
from revolve2.standards import modular_robots_v2
from rl_env import make, make_vec

TRAINERS = {
    "ppo": PPO
}


def robot_body(name: Optional[str] = None):
    if name is None or name == "empty":
        return BodyV2()
    else:
        return modular_robots_v2.get(name)


def file_name(args):
    path = args.trainer.lower() + "_" + args.body
    if args.rotated:
        path += '45'
    path += f"_S{args.seed}"
    path += ".zip"
    return Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", default="spider")
    parser.add_argument("--trainer", default="ppo")
    parser.add_argument("-r", "--rotated", default=False, action="store_true")
    parser.add_argument("-t", "--timesteps", default=10_000, type=int)
    parser.add_argument("-T", "--simulation-time", default=10, type=float)
    parser.add_argument("-s", "--seed", default=0, type=int)

    parser.add_argument("--rerun", type=Path, default=None)

    args = parser.parse_args()

    name = args.body
    rotated = args.rotated
    trainer = PPO

    model_file = args.rerun or file_name(args)
    rerun = model_file.exists()

    log = None
    if rerun:
        log = model_file.with_suffix("")
        log.mkdir(parents=True, exist_ok=True)

    robot = ModularRobot(body=robot_body(name), brain=BrainDummy())
    reward_func = MoveForwardFitness(robot, render=rerun, log=log)
    env_kwargs = dict(
        robot=robot,
        rotated=rotated,
        reward_function=reward_func,
        simulation_time=args.simulation_time,
    )

    if not rerun:
        print("Training", model_file)
        vec_env = make_vec(n=6,
                           **env_kwargs,
                           vec_env_cls=SubprocVecEnv)

        # Define and Train the agent
        model = trainer("MlpPolicy", vec_env, device="cpu")
        model.learn(total_timesteps=args.timesteps, progress_bar=True)

        model.save(model_file)

    else:
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

        reward_func.do_plots()

        print("Final reward:", env.cumulative_reward)

        # time.sleep(1)  # Ugly but prevents errors when quitting the window


if __name__ == '__main__':
    main()
