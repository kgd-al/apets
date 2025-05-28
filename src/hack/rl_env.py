from hack.evaluator import Evaluator
from hack.local_simulator import LocalSimulator, GymSimulator


def make(robot, rerun, rotated, reward_function):
    scene = Evaluator.scene(robot, rerun, rotated)
    simulator = GymSimulator(
        scene=scene,
        fitness_function=reward_function,

        headless=not rerun,
        start_paused=False,
        simulation_time=10,
        record_settings=None
    )

    return simulator
