import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas import Series

parser = argparse.ArgumentParser("Sumarizes summary.csv files")
parser.add_argument("root", type=Path, nargs="+")
args = parser.parse_args()

df = pd.concat(
    pd.read_csv(f, index_col=0)
    for root in args.root
    for f in root.glob("**/summary.csv")
)

print(df.columns)
print(df)

print(df.loc[[df.kernels.idxmax(), df.speed.idxmax()]][["arch", "neighborhood", "width", "depth", "kernels", "speed"]])

print("Plotting...")

sns.set_style("darkgrid")
pdf_file = args.root[0].joinpath("summary.pdf")
with PdfPages(pdf_file) as pdf:
    g = sns.scatterplot(df, x="params", y="depth")
    plt.xscale('log', base=10)
    pdf.savefig(g.figure, bbox_inches="tight")
    plt.close()

    g = sns.scatterplot(df, x="width", y="depth")
    plt.xscale('log', base=2)
    pdf.savefig(g.figure, bbox_inches="tight")
    plt.close()

    def hue(overview):
        def fmt_rl(_s):
            trainer = _s.split("/")[4]
            if trainer == "rlearn":
                trainer = "ppo"
            return trainer
        return Series(name="arch" + ("" if overview else "-detailed"),
                      data=(
            df.arch
            + ("" if overview else df.depth.map(lambda f: str(int(f)) if not np.isnan(f) else ""))
            + "-" + df.index.map(fmt_rl)
        ))

    def plot(x, y, base=10, **kwargs):
        _args = dict(x=x, y=y, hue=hue(kwargs.pop("overview", True)), col="reward",
                     kind='line', marker='o',
                     err_style="bars", errorbar="sd")

        _args.update(kwargs)
        g = sns.relplot(df, **_args)
        plt.xscale('log', base=base)

        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    for overview in [True, False]:
        g = sns.scatterplot(df, x="speed", y="kernels", hue=hue(overview), style="reward")
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        for k in ["speed", "kernels"]:
            plot(x="params", y=k, overview=overview)
            plot(x="params", y=k, errorbar=("pi", 100), err_style="band", overview=overview)
    # plot(x="width", y="speed", base=2)
    plot(x="params", y="tps")
    # plot(x="width", y="tps", base=2)


print("Generated", pdf_file)
