import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TABLEAU_COLORS

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkStatic, BrainCpgInstance


class Logger:
    def __init__(self, cpg: BrainCpgInstance):
        self.current_data = None
        self.h2i = {h: i for i, h in cpg._output_mapping}


class RevolveLogger(Logger, ModularRobotControlInterface):
    def __init__(self, cpg: BrainCpgInstance):
        super().__init__(cpg)
        self.data = {}
        self.channels = [[], []]

    def set_channel(self, channel):
        self.data[channel] = [[] for _ in range(len(self.h2i))]
        self.current_data = self.data[channel]

        for i, sub_channel in enumerate(channel):
            if sub_channel not in self.channels[i]:
                self.channels[i].append(sub_channel)

    def set_active_hinge_target(self, active_hinge: ActiveHinge, target: float) -> None:
        self.current_data[self.h2i[active_hinge]].append(float(target))


class SimpleLogger(Logger):
    def __init__(self, cpg: BrainCpgInstance):
        super().__init__(cpg)
        self.current_data = [[] for _ in range(len(self.h2i))]


class CPG:
    cpg_factory: BrainCpgNetworkStatic
    cpg: BrainCpgInstance

    def __init__(self):
        self.A = self.cpg._weight_matrix.copy()
        self.state = self.cpg_factory._initial_state.copy()

    def reset(self):
        self.__init__()

    def step(self, dt, logger):
        self._step(dt, logger)
        for j, x in enumerate(self.state[:8]):
            logger.current_data[j].append(x)

    def _step(self, dt, logger): pass


class RevolveBaseline(CPG):
    name = "revolve_baseline"

    def __init__(self):
        super().__init__()
        self.__cpg = self.cpg_factory.make_instance()

    def reset(self): self.__cpg._state = self.cpg_factory._initial_state.copy()

    def step(self, dt, logger):
        self.__cpg.control(dt, None, logger)


class RevolveFactorized(CPG):
    name = "revolve_factorized"

    def _step(self, dt, logger):
        a, s, t = self.A, self.state, dt
        s += (
            (a @ a @ a @ a @ s * t ** 4) / 24
            + (a @ a @ a @ s * t ** 3) / 6
            + (a @ a @ s * t ** 2)
            + (a @ s * t)
        )
        self.state = s.clip(-1, 1)


class RevolveLowCost(CPG):
    name = "revolve_low-cost"

    def _step(self, dt, logger):
        a, s, t = self.A, self.state, dt
        self.state = np.clip(s + a @ s * t, -1, 1)


class TensorCPGBase(CPG):
    def __init__(self):
        super().__init__()
        self.A = torch.tensor(self.A, requires_grad=True)
        self.state = torch.tensor(self.state, requires_grad=False)

    def step(self, dt, logger):
        self._step(dt, logger)
        for j, x in enumerate(self.state[:8].detach().numpy().tolist()):
            logger.current_data[j].append(x)


class TensorCPG(TensorCPGBase):
    name = "tensor_cpg"

    def _step(self, dt, logger):
        a, s, t = self.A, self.state, dt
        A1 = torch.matmul(a, s)
        A2 = torch.matmul(a, (s + dt / 2 * A1))
        A3 = torch.matmul(a, (s + dt / 2 * A2))
        A4 = torch.matmul(a, (s + dt * A3))
        s = s + dt / 6 * (A1 + 2 * (A2 + A3) + A4)
        self.state = s.clip(-1, 1)


class TensorLowCost(TensorCPGBase):
    name = "tensor_low-cost"

    def _step(self, dt, logger):
        self.state = (self.state + self.A @ self.state * dt).clip(-1, 1)


def plot_frequencies(file):
    logger = Logger(CPG.cpg)

    rng = np.random.RandomState(seed=0)

    freqs = [20, 50, 100, 1000] + [f"20~{s}s" for s in [.5, 1, 2]]
    linspaces = {}

    for freq in freqs:
        start, end = 0, 4

        if not isinstance(freq, int):
            dt = 1 / int(freq.split("~")[0])
        else:
            dt = 1 / freq

        linspace = np.linspace(start, end, int((end - start) / dt))

        if isinstance(freq, int):
            def t_t(_): return dt

        else:
            linspace += rng.normal(loc=0, scale=dt*float(freq.split("~")[1][:-1]),
                                   size=linspace.shape)
            linspace[0], linspace[-1] = start, end
            linspace = linspace.clip(start, end)
            linspace.sort()

            _dt = linspace[1:] - linspace[:-1]
            def t_t(_i): return _dt[_i]

        linspaces[freq] = linspace

        for cpg in [
            RevolveBaseline(), RevolveFactorized(), RevolveLowCost(),
            TensorCPG(), TensorLowCost()
        ]:
            cpg.reset()
            logger.set_channel((freq, cpg.name))
            for _l, _x in zip(logger.current_data, cpg.state):
                _l.append(float(_x))
            for i, _li in enumerate(linspace[1:]):
                cpg.step(t_t(i), logger)

    keys = logger.channels[1]
    n = len(keys)

    with PdfPages(file) as pdf:
        fig, axes = plt.subplots(8, len(freqs), sharex=True, sharey=True,
                                 figsize=(16, 9))

        handles = {}
        for k, c in enumerate(keys):
            for j, ax_row in enumerate(axes):
                for i, (freq, ax) in enumerate(zip(freqs, ax_row)):
                    handles[c], = ax.plot(linspaces[freq], logger.data[(freq, c)][j],
                                          label=c, linestyle=(k/2, (n/2, n/2)),
                                          color=list(TABLEAU_COLORS.values())[k])

        for ax in axes.flat:
            ax.grid(True)

        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")

        for freq, ax in zip(freqs, axes[0]):
            ax.set_title(f"{freq} Hz")

        fig.legend(handles=handles.values(), ncols=len(handles),
                   loc='outside upper center', bbox_to_anchor=(0.5, 1.025))

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')

        fig, axes = plt.subplots(8, len(keys), sharex=True, sharey=True,
                                 figsize=(16, 9))

        handles = {}
        for k, freq in enumerate(freqs):
            for j, ax_row in enumerate(axes):
                for i, (c, ax) in enumerate(zip(keys, ax_row)):
                    handles[freq], = ax.plot(
                        linspaces[freq], logger.data[(freq, c)][j],
                        label=f"{freq} Hz", linestyle=(k/2, (n/2, n/2)),
                        color=list(TABLEAU_COLORS.values())[k])

        for ax in axes.flat:
            ax.grid(True)

        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")

        for key, ax in zip(keys, axes[0]):
            ax.set_title(key)

        fig.legend(handles=handles.values(), ncols=len(handles),
                   loc='outside upper center', bbox_to_anchor=(0.5, 1.025))

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')

    print("Generated plots")


def plot_learning(file):
    dt = 1 / 20
    logger = SimpleLogger(CPG.cpg)

    cpg = TensorCPG()
    cpg.step(dt, logger)

    print(logger.current_data)
    print(cpg.A.grad)
    print(cpg.state)
    cpg.state.backward()
    # cpg.A.backward()
    print(cpg.A.grad_fn)
    print(cpg.A.grad)


if __name__ == "__main__":
    outfile = Path("cpg_rl.pdf")

    path = Path("remote/showcase/bests/cma_kernels_cpg-6_run-3.revolve_controller.pkl")

    with open(path, "rb") as f:
        brain: BrainCpgNetworkStatic = pickle.load(f)

    with np.printoptions(precision=2, floatmode="fixed", suppress=True, linewidth=200):
        print(brain._weight_matrix)

    CPG.cpg_factory = brain
    CPG.cpg = brain.make_instance()

    if len(sys.argv) > 1:
        flag = int(sys.argv[1])
        if (flag & 1) > 0:
            plot_frequencies(outfile)

        if (flag & 2) > 0:
            plot_learning(outfile)
