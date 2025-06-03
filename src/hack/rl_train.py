import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from hack.evaluator import MoveForwardFitness
from hack.rl_env import make, make_vec
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.dummy import BrainDummy
from revolve2.standards.modular_robots_v2 import spider_v2


def main():
    model_file = Path("ppo_spider.zip")
    rerun = model_file.exists()

    robot = ModularRobot(body=spider_v2(), brain=BrainDummy())
    env_kwargs = dict(
        robot=robot,
        rotated=False,
        reward_function=MoveForwardFitness(robot),
        simulation_time=60,
    )

    if not rerun:
        vec_env = make_vec(n=6,
                           **env_kwargs,
                           vec_env_cls=SubprocVecEnv)

        # Define and Train the agent
        model = PPO("MlpPolicy", vec_env, device="cpu")
        model.learn(total_timesteps=1000000, progress_bar=True)

        model.save(model_file)

    else:
        env = make(rerun=True, **env_kwargs)
        check_env(env)

        model = PPO.load(model_file, device="cpu")

        obs, infos = env.reset()
        while not env.done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

        time.sleep(1)  # Ugly but prevents errors when quitting the window


if __name__ == '__main__':
    main()
