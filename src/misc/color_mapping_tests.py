import logging
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
from matplotlib.pyplot import colormaps
from seaborn import heatmap
from seaborn_image import imshow

n = 64
annot = False

data = [
    [(r, g, b, 1) for r, g, b in [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [.5, .5, .5]
    ]],
    *[colormaps[cm](np.linspace(0, 1, n))
      for cm in ["hsv", "hot"]]
]
mappings = [
    ("Lossy", lambda r, g, b: -1 * r + g),
    ("Better", lambda r, g, b: -1 * r + max(g, b)),
    ("Non-ternary", lambda r, g, b: max(-1, min(-r + g - .33 * b, 1))),
]

samples = len(data[0])+(len(data)-1)*n
fig, axes_arr = plt.subplots(nrows=len(mappings)+1, ncols=len(data), sharex=False, sharey=False,
                             figsize=[.5*samples, .8*len(mappings)],
                             width_ratios=[len(data[i])/samples for i in range(len(data))])


def plot(_axes, _name, map_fn, plot_fn, **kwargs):
    for i, ax in enumerate(_axes):
        plot_fn(map_fn(data[i]), **kwargs, ax=ax)
    _axes[0].text(-0.01, 0.5, _name, va="center", ha="right", transform=_axes[0].transAxes)


plot(axes_arr[0], "Truth", lambda _l: [_l], imshow, cbar=False)

cmap = LinearSegmentedColormap.from_list("vision", [[1, 0, 0], [0, 0, 0], [0, 1, 0]])

for axes, (name, mapping) in zip(axes_arr[1:], mappings):
    plot(axes, name, lambda l: [[mapping(r, g, b) for r, g, b, _ in l]], heatmap,
         cmap=cmap, annot=annot, cbar=False, vmin=-1, vmax=1)
    for i, _ax in enumerate(axes):
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_aspect("equal")

fig.tight_layout()
fig.show()
