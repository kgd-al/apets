#!/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.pyplot import colormaps
from seaborn import heatmap

n = 64
annot = False

tested_colormaps = ["Manual", "hsv", "viridis", "inferno", "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"]

data = [
    [(r, g, b, 1) for r, g, b in [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0], [.5, .5, .5],
        [1, .9, .9], [.9, 1, .9], [.9, .9, 1], [.9, .7, .7], [.7, .9, .7], [.7, .7, .9]
    ]],
    *[colormaps[cm](np.linspace(0, 1, n))
      for cm in tested_colormaps if cm != "Manual"],
]
a = .25
mappings = [
    ("Lossy", lambda r, g, b: -1 * r + g),
    # ("Better", lambda r, g, b: -1 * r + max(g, b)),
    ("Non-ternary", lambda r, g, b: max(-1, min(-r + g - .33 * b, 1))),
    ("Better?", lambda r, g, b: max(-1, min(- 2 * r + 1.5 * g + .25 * b, 1))),
    # ("Better?", lambda r, g, b: max(-1, min(- r + g - a*(1 - max(r, g, b) + min(r, g, b)) * max(r, g, b), 1))),
]

cmap = LinearSegmentedColormap.from_list("vision", [[1, 0, 0], [0, 0, 0], [0, 1, 0]])

with PdfPages("color_mappings.pdf") as pdf:
    for name, mapping in mappings:
        fig = plt.figure(figsize=[6, 2 * len(data)])
        outer_grid = GridSpec(nrows=2*len(data), ncols=1, hspace=.5)

        for i, colors in enumerate(data):
            #     for ax, (name, _) in zip(axes, [("Truth", None)] + mappings):
            #         ax.text(0.5, 1.05, name, va="bottom", ha="center", transform=ax.transAxes)

            inner_grid = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=outer_grid[i],
                                                 hspace=0.1)
            ax0, ax1 = axes = [plt.subplot(cell) for cell in inner_grid]
            ax0.imshow([colors], aspect="auto")

            ax0.text(0.5, 1.05, tested_colormaps[i], va="bottom", ha="center", transform=ax0.transAxes)
            ax0.text(-0.01, .5, "Truth", va="center", ha="right", transform=ax0.transAxes)
            ax1.text(-0.01, .5, name, va="center", ha="right", transform=ax1.transAxes)

            heatmap(
                [[mapping(r, g, b) for r, g, b, _ in colors]],
                ax=ax1,
                cmap=cmap, annot=(i == 0) or annot, cbar=False, vmin=-1, vmax=1,
                annot_kws={"fontsize": 8}
            )
            for _ax in axes:
                _ax.set_xticks([])
                _ax.set_yticks([])
                # _ax.set_aspect("equal")

        pdf.savefig(fig, bbox_inches="tight")
