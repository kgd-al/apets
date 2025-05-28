from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from hack.evaluator import MoveForwardFitness
from hack.rl_env import make
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.dummy import BrainDummy
from revolve2.standards.modular_robots_v2 import spider_v2

# Instantiate the env
robot = ModularRobot(body=spider_v2(), brain=BrainDummy())
env = make(robot=robot, rerun=False, rotated=True, reward_function=MoveForwardFitness(robot))
print(env)
check_env(env)

# Define and Train the agent
model = PPO("MLPPolicy", env).learn(total_timesteps=1000, progress_bar=True)
