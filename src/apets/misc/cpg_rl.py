import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, BrainCpgInstance

path = Path("remote/showcase/bests/cma_kernels_cpg-6_run-3.revolve_controller.pkl")

with open(path, "rb") as f:
    brain: BrainCpgNetworkStatic = pickle.load(f)

print(brain)
with np.printoptions(precision=2, floatmode="fixed", suppress=True, linewidth=200):
    print(brain._weight_matrix)

revolve_brain: BrainCpgInstance = brain.make_instance()

fig, axes = plt.subplots(8, 1, sharex=True, sharey=True,
                         figsize=(8, 8))

h2i = {h: i for i, h in revolve_brain._output_mapping}
class Logger(ModularRobotControlInterface):
    def __init__(self, channels):
        self.data = {c: [[] for _ in range(len(h2i))] for c in channels}
        self.current_data = None

    def set_channel(self, channel):
        self.current_data = self.data[channel]

    def set_active_hinge_target(self, active_hinge: ActiveHinge, target: float) -> None:
        self.current_data[h2i[active_hinge]].append(target)

logger = Logger(["revolve_cpg", "factorized_cpg"])

start, end, count = 0, 5, 1000
dt = (end - start)/count
linspace = np.linspace(start, end, count)

logger.set_channel("revolve_cpg")
for _ in linspace:
    revolve_brain.control(dt, None, logger)

logger.set_channel("factorized_cpg")
a, s, t = revolve_brain._weight_matrix, brain._initial_state, dt
for _ in linspace:
    s += ((a @ a @ a @ a @ s * t**4) / 24
          + (a @ a @ a @ s * t**3) / 6
          + (a @ a @ s * t**2)
          + (a @ s * t))
    for i, x in enumerate(s[:8]):
        logger.current_data[i].append(x)

for c in logger.data.keys():
    for i, ax in enumerate(axes.flat):
        ax.plot(linspace, logger.data[c][i])

fig.tight_layout()
fig.savefig("revolve_cpg.pdf", bbox_inches='tight')
